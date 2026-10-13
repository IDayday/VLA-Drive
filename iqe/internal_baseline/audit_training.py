"""Read-only evidence from the completed, selected own expert/scorer experiment.

The fitting-set result is a diagnostic, never a checkpoint-selection metric.
No optimizer or scoring simulation is constructed here.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
import json
import os
import subprocess
import numpy as np
import torch
from iqe.contracts import require
from iqe.io import read_json, atomic_json, atomic_bytes, file_hash, digest
from iqe.data.manifests import load_scenes
from .sampling import ExpertSampler, TwoStreamSampler
from .training import load_bank, make_scorer, predict_scores, training_device
from .evaluation import selection_metrics, argmax_base_tie


def replay_consumption(sampler, rows):
    counts = {}
    for row in rows:
        entries = sampler.peek(128)
        require([e['scene_id'] for e in entries] == row['sample_ids'], 'actual consumed sequence differs from locked sampler')
        for entry in entries:
            stream = entry.get('stream', entry.get('bucket'))
            counts.setdefault(stream, Counter())[entry['scene_id']] += 1
        sampler.consume(128)
    return counts


def summarize_counts(counts):
    return {name: {'total_exposures': sum(c.values()), 'unique_scenes': len(c),
                   'minimum_exposures_per_seen_scene': min(c.values()),
                   'maximum_exposures_per_seen_scene': max(c.values()),
                   'per_scene_exposures': dict(c)} for name, c in counts.items()}


def read_steps(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    require([r['optimizer_step'] for r in rows] == list(range(1, len(rows)+1)), 'duplicate/missing optimizer steps')
    return rows


def selected_reference_audit(config, bank, arrays, scenes):
    """Fixed eight stage_val scenes; rescore the actual selected trajectory alone."""
    from .bank import protocol
    decisions = {r['scene_id']: r for r in read_json(config.root/'stage_val_decisions.json')['rows']}
    rows = [r for r in bank['rows'] if r['role'] == 'stage_val'][:8]
    proto = protocol(config)
    root = config.root/'selected_single_reference'
    atomic_json(root/'FIXED_SCENES.json', {'scene_ids': [r['scene_id'] for r in rows],
                'rule': 'first eight in pre-score fixed stage_val manifest, not selected by expert wins'}, immutable=True)
    expert_hash = read_json(config.root/'expert/result.json')['termination']['best_snapshot']['path']
    from iqe.evaluation.retention import module_hash
    from .binding import load_head
    head = load_head(config)
    head.load_state_dict(torch.load(expert_hash, weights_only=True, map_location='cpu'), strict=True)
    expert_hash = module_hash(head)
    base_hash = read_json(config.root/'BASE_BINDING.json')['checkpoint_hash']
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(repo), proto['navsim_root'], proto['nuplan_root'])),
               CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    def one(row):
        scene = scenes[row['scene_id']]
        choice = decisions[scene.scene_id]['choices']['B3_ScratchScorer_argmax']
        candidate = read_json(config.root/'candidates'/f'expert_{choice}'/(expert_hash if choice else base_hash)/(scene.scene_id+'.json'))
        from iqe.contracts import CandidateRecord, strict_record
        require(strict_record(CandidateRecord, candidate).key == row['expert_key' if choice else 'base_key'], 'selected candidate join changed')
        request = {'candidate': candidate, 'scene': asdict(scene), 'protocol_hash': proto['protocol_hash'],
                   'navsim_root': proto['navsim_root'], 'source_hash': proto['navsim_source_hash'], 'config': proto['config'],
                   'seed': int(digest({'scene_id': scene.scene_id, 'seed': 42, 'repetition': 0})[:8], 16), 'eval_seed': 42, 'repetition': 0}
        req, response = root/(scene.scene_id+'.request.json'), root/(scene.scene_id+'.response.json')
        atomic_json(req, request, immutable=True)
        done = subprocess.run([config.scoring_python, '-m', 'iqe.scoring.worker', str(req), str(response)],
                              cwd=repo, env=env, capture_output=True, text=True, timeout=300)
        require(done.returncode == 0, done.stderr[-2000:])
        actual = read_json(response)
        require(actual['score_valid'], 'invalid selected official score')
        index = next(i for i,r in enumerate(bank['rows']) if r['scene_id'] == scene.scene_id)
        require(actual['total_score_01'] == arrays['scores'][index,choice], 'selected total differs from bank')
        require(all(actual['named_components'][c] == arrays['components'][index,choice,j] for j,c in enumerate(bank['components'])),
                'selected components differ from bank')
        return {'scene_id': scene.scene_id, 'selected_expert': choice, 'score_01': actual['total_score_01'], 'exact_equal': True}
    with ThreadPoolExecutor(max_workers=4) as pool:
        evidence = list(pool.map(one, rows))
    result = {'status': 'PASS', 'fresh_official_single_trajectory_calls': len(rows), 'evidence': evidence,
              'protocol_hash': proto['protocol_hash'], 'no_joint_candidate_normalization': True}
    atomic_json(config.root/'SELECTED_REFERENCE_AUDIT.json', result, immutable=True)
    return result


def audit(config, device='cuda:0'):
    from .budget import training_budget
    budget = training_budget(config)
    device = training_device(device)
    torch.set_num_threads(2)
    bank, arrays = load_bank(config)
    scenes = {s.scene_id: s for s in load_scenes(config.root/'scenes.json')}
    manifest = read_json(config.root/'EXPERT_MANIFEST.json')
    q = read_json(config.root/'COMPARISON_DISTRIBUTION.json')
    erows, srows = (read_steps(config.root/k/'steps.jsonl') for k in ('expert', 'scorer'))
    require(len(erows) == budget['expert_updates'] and len(srows) == budget['scorer_updates'], 'fixed budget incomplete')
    es = ExpertSampler(manifest['hard'], manifest['ordinary'], config.natural_seed, manifest['hash'],
                       manifest['sampling']['hard'])
    ss = TwoStreamSampler(q['scene_ids'], q['weights'], natural_seed=config.natural_seed,
                          comparison_seed=config.comparison_seed, pointwise_updates=budget['pointwise_updates'],
                          bank_hash=file_hash(config.root/'PAIR_BANK.json'))
    ec, sc = replay_consumption(es, erows), replay_consumption(ss, srows)
    require(ss.q_hash == q['q_hash'], 'replayed comparison identity mismatch')
    exposure = {'expert': summarize_counts(ec), 'scorer': summarize_counts(sc),
                'sampler_sequence_matches_all_actual_updates': True, 'final_scorer_sampler': ss.state_dict()}
    atomic_json(config.root/'EXPOSURE_AUDIT.json', exposure, immutable=True)

    model = make_scorer(config).to(device).eval()
    result = read_json(config.root/'scorer/result.json')
    require(result['status'] == 'COMPLETE' and result['best_checkpoint_hash'] == file_hash(config.root/'scorer/best.pt'), 'selected scorer mismatch')
    model.load_state_dict(torch.load(config.root/'scorer/best.pt', weights_only=True, map_location=device), strict=True)
    from .partition import training_ids
    allowed_fit = training_ids(config, list(scenes.values()), 'scorer_fit')
    ix = [i for i, r in enumerate(bank['rows']) if r['scene_id'] in allowed_fit]
    rows = [scenes[bank['rows'][i]['scene_id']] for i in ix]
    predicted, _ = predict_scores(config, model, rows, arrays['trajectories'][ix], device)
    fit = selection_metrics(arrays['scores'][ix], arrays['components'][ix], argmax_base_tie(predicted),
                            [s.source_log_id for s in rows], bootstrap_repetitions=config.bootstrap_repetitions,
                            bootstrap_seed=config.bootstrap_seed)
    atomic_json(config.root/'MAIN_SCORER_FIT_DIAGNOSTIC.json', {'selection': fit, 'checkpoint_selection_used_fit': False,
                'fit_subrole': 'scorer_fit' if config.research_protocol == 'two_candidate_v1' else 'incremental_fit',
                'scorer_hash': file_hash(config.root/'scorer/best.pt'), 'rows': len(ix)}, immutable=True)
    delta = arrays['trajectories'][:, 1, :, :2] - arrays['trajectories'][:, 0, :, :2]
    distances = np.linalg.norm(delta, axis=-1)
    yaw = arrays['trajectories'][:, 1, :, 2]-arrays['trajectories'][:, 0, :, 2]
    yaw = np.abs(np.arctan2(np.sin(yaw), np.cos(yaw)))
    diversity = {}
    for role in sorted({r['role'] for r in bank['rows']}):
        mask = np.array([r['role'] == role for r in bank['rows']])
        diversity[role] = {'n': int(mask.sum()), 'pair_ADE_m': float(distances[mask].mean()),
                           'pair_FDE_m': float(distances[mask, -1].mean()), 'periodic_yaw_difference_rad': float(yaw[mask].mean()),
                           'exact_identical_trajectories': int((arrays['trajectories'][mask, 0] == arrays['trajectories'][mask, 1]).all((1, 2)).sum())}
    atomic_json(config.root/'CANDIDATE_GEOMETRY.json', diversity, immutable=True)
    selected_reference = selected_reference_audit(config, bank, arrays, scenes)

    def training_summary(kind, rows):
        val = {p.stem: read_json(p) for p in sorted((config.root/kind).glob('validation_*.json'))}
        timings = np.array([r['compute_update_seconds'] for r in rows])
        gradients = {str(r['optimizer_step']): r['gradients'] for r in rows if r['gradients']}
        require(all(np.isfinite(r['loss']) and np.isfinite(r['gradient_norm']) for r in rows), 'nonfinite training history')
        return {'updates': len(rows), 'global_batch': 128, 'micro_batch': config.expert_micro_batch if kind == 'expert' else config.scorer_micro_batch,
                'normal_update_seconds': {'mean': float(timings.mean()), 'p50': float(np.median(timings)), 'p95': float(np.quantile(timings, .95))},
                'first_loss': rows[0]['loss'], 'last_loss': rows[-1]['loss'], 'last_loss_terms': rows[-1]['losses'],
                'finite_loss_and_gradient_all_steps': True, 'gradient_evidence': gradients,
                'validation': val, 'selected_result': read_json(config.root/kind/'result.json')}
    summary = {'expert': training_summary('expert', erows), 'scorer': training_summary('scorer', srows),
               'effective_training_budget': budget,
               'main_scorer_fit': fit, 'exposure_summary': {k: {s: {n: v for n, v in c.items() if n != 'per_scene_exposures'}
                    for s, c in exposure[k].items()} for k in ('expert', 'scorer')},
               'natural_and_comparison_losses_separate': True, 'comparison_second_gap_weight': False,
               'VLM_calls_during_cached_head_training': 0, 'external_scorer_weights_loaded': False}
    summary['selected_reference'] = selected_reference
    atomic_json(config.root/'TRAINING_AUDIT.json', summary, immutable=True)
    text = (f"# Own scorer training evidence\n\nPointwise: {budget['pointwise_updates']} updates of 128 natural scenes. "
            f"Then {budget['scorer_updates']-budget['pointwise_updates']} updates of 64 natural + 64 gap-sampled comparison scenes when comparison signal exists. "
            f"Continuous {budget['scorer_updates']}-step scheduler; no external initialization. "
            "Fitting-set selection is diagnostic, not generalization evidence. Actual per-scene/stream exposure, including absent-comparison fallback, is in EXPOSURE_AUDIT.json.\n\n```json\n"
            +json.dumps(summary, indent=2)+"\n```\n")
    atomic_bytes(config.root/'SCORER_TRAINING_REPORT.md', text.encode())
    return {'status': 'PASS', 'fit_diagnostic': fit, 'exposures': summary['exposure_summary']}


if __name__ == '__main__':
    import argparse
    from .config import load_config
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--device', default='cuda:0')
    a = p.parse_args(); print(json.dumps(audit(load_config(a.config), a.device)))
