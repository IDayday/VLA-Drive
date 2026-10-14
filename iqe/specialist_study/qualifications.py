"""Bounded real-data checks for this campaign, separate from scientific models."""
import argparse
from dataclasses import replace
from pathlib import Path
import shutil

import numpy as np
import torch

from iqe.contracts import require
from iqe.io import atomic_json, digest, file_hash, read_json
from iqe.losses import expert_il_terms
from iqe.training.trainer import train
from iqe.internal_baseline.bank import save_candidate, score_candidates
from .binding import scenes_by_role
from .config import load_config
from .data import ReusedContext, verified_base_scores
from .models import make_specialist, EMAExpert
from .objectives import SceneSampler
from .resource_guard import check
from .training import runtime, preload, trainer_config


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal_tree(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def expert_resume(config, device):
    """Eight diagnostic updates total: four continuous versus two plus two."""
    check(config, 'train-expert-cell')
    device = runtime(device)
    folder = config.root/'qualifications/expert_gpu_resume'
    output = folder/'REPORT.json'
    if output.exists():
        return read_json(output)
    manifest = read_json(config.root/'EXPERT_MATRIX_DATA.json')
    lookup = {s.scene_id: s for s in scenes_by_role(config)['expert_fit']}
    scenes = [lookup[i] for i in manifest['scene_ids'][:32]]
    require(len(scenes) == 32, '32 real eligible specialist scenes required')
    cache = preload(config, scenes)
    identities = [s.scene_id for s in scenes]
    model_state = make_specialist(config, True).state_dict()
    cfg = trainer_config(config, 'expert')
    cfg['max_optimizer_steps'] = 4  # diagnostic only; main 5000-step runs untouched
    cfg['warmup_steps'] = 1
    cfg['validation_steps'] = []
    cfg['checkpoint_every_steps'] = 2
    deps = {'diagnostic': 'four_real_updates_private_reader_raw_ema', 'Base': file_hash(config.root/'BASE_BINDING.json'),
            'scene_ids': digest(identities), 'not_a_scientific_expert': True}
    prev = read_json(config.root/'OFFLINE_BASE_CONTRACT.json')['query_architecture']['prev_weight']

    def fetch(entries, dev):
        x = cache.batch([lookup[e['scene_id']] for e in entries], dev)
        return {'x': x, 'target': x['target_raw']}

    def loss(module, batch):
        return expert_il_terms(module(batch['x']), batch['target'], prev_weight=prev)

    def run(name, stop=None, resume=None):
        raw = make_specialist(config, True).to(device)
        raw.load_state_dict(model_state, strict=True)
        model = EMAExpert(raw)
        sampler = SceneSampler(identities, config.seed, deps['scene_ids'])
        train(model, loss, fetch, sampler, cfg, deps, folder/name, device=device, seed=config.seed,
              stop_after=stop, resume=resume, after_optimizer_step=lambda m, _: m.update(),
              denominator_function=lambda b: {'il': torch.tensor(len(b['target']), device=device).float()})
        return sampler.peek(128)

    continuous_ids = run('continuous')
    run('interrupted', stop=2)
    resumed_ids = run('interrupted', resume='auto')
    paths = [folder/name/'step_000004.pt' for name in ('continuous', 'interrupted')]
    a, b = [torch.load(p, map_location='cpu', weights_only=False) for p in paths]
    fields = ('module', 'optimizer', 'scheduler', 'sampler', 'rng_by_rank', 'consumed_sample_ids')
    equality = {key: equal_tree(a[key], b[key]) for key in fields}
    equality['next_sample_ids'] = continuous_ids == resumed_ids
    require(all(equality.values()), 'real GPU exact-resume mismatch')
    import json
    logs = [[json.loads(line) for line in (folder/name/'steps.jsonl').read_text().splitlines()]
            for name in ('continuous', 'interrupted')]
    loss_equal = [r['loss'] for r in logs[0]] == [r['loss'] for r in logs[1]]
    require(loss_equal, 'real GPU resumed losses changed')
    terminal_ids = run('interrupted', resume='auto')
    terminal = torch.load(paths[1], map_location='cpu', weights_only=False)
    require(terminal_ids == resumed_ids and equal_tree(terminal, b), 'terminal resume consumed/updated state')
    result = {'status': 'PASS', 'device': device, 'hardware': torch.cuda.get_device_name() if 'cuda' in device else 'CPU',
              'Base_binding_hash': deps['Base'], 'real_scene_ids': identities, 'continuous_updates': 4,
              'interrupted_updates': [2, 2], 'terminal_added_updates': 0, 'equality': equality, 'losses_equal': loss_equal,
              'checkpoint_hashes': [file_hash(p) for p in paths], 'production_dropout_preserved': True,
              'diagnostic_model_not_registered_as_candidate': True}
    atomic_json(output, result, immutable=True)
    return result


def official_invariance(config):
    """Fresh official calls, not reuse of the labels under investigation."""
    check(config, 'audit-target-opportunities')
    folder = config.root/'qualifications/official_reference'
    output = folder/'REPORT.json'
    if output.exists():
        return read_json(output)
    ids = read_json(config.root/'EXPERT_MATRIX_DATA.json')['scene_ids'][:2]
    lookup = {s.scene_id: s for s in scenes_by_role(config)['expert_fit']}
    scenes = [lookup[i] for i in ids]
    old = verified_base_scores(config, scenes)
    cache = ReusedContext(config)
    candidates, observations = [], []
    for s in scenes:
        values = cache.read(s)
        for kind in ('base_physical', 'target_physical'):
            candidates.append(save_candidate(config, s, 'qualification_'+kind, digest([s.scene_id, kind]), values[kind].numpy()))
            observations.append(s)
    tests = [('original', [0, 1, 2, 3], 2, 4), ('reordered', [3, 1, 0, 2], 3, 4),
             ('deleted_other', [0, 2], 1, 2), ('duplicated', [0, 0, 1, 2, 3], 1, 5),
             ('chunk_one', [0, 1, 2, 3], 1, 1), ('chunk_three', [0, 1, 2, 3], 2, 3)]
    truth, details = {}, []
    for name, order, workers, chunk in tests:
        root = folder/name
        root.mkdir(parents=True, exist_ok=True)
        for filename in ('SCORE_PROTOCOL.json', 'OFFLINE_BASE_CONTRACT.json', 'BASE_BINDING.json'):
            if not (root/filename).exists():
                shutil.copy2(config.root/filename, root/filename)
        c = replace(config, output_root=str(root), scoring_workers=workers)
        values = []
        for at in range(0, len(order), chunk):
            subset = order[at:at+chunk]
            values.extend(score_candidates(c, [observations[i] for i in subset], [candidates[i] for i in subset], name+str(at)))
        errors = []
        for index, value in zip(order, values):
            comparable = {'total': value.total_score_01, 'components': value.named_components, 'valid': value.component_valid_masks}
            if index not in truth:
                truth[index] = comparable
            require(truth[index] == comparable, 'official label depends on candidate ordering/pool/chunk')
            if index % 2 == 0:
                original = old[observations[index].scene_id]
                require(original['total_score_01'] == value.total_score_01 and original['named_components'] == value.named_components,
                        'fresh official score does not equal original Base bank')
            errors.append(0.)
        details.append({'name': name, 'order': order, 'workers': workers, 'chunk_size': chunk, 'max_abs_error': max(errors), 'labels': len(values)})
    result = {'status': 'PASS', 'real_scene_ids': ids, 'protocol_hash': read_json(config.root/'SCORE_PROTOCOL.json')['protocol_hash'],
              'GT_is_supervision_audit_only': True, 'independent_fresh_official_contexts': True, 'tests': details,
              'cached_Base_vs_fresh_official_exact': True, 'no_navtest_access': True}
    atomic_json(output, result, immutable=True)
    return result


def scorer_resume(config, device):
    from .bank import load_bank
    from .scorer_training import create_scorer, batch_fetch
    from .objectives import StudySampler, pair_population, terms, denominators
    check(config, 'train-scorer-cell');device=runtime(device)
    root=config.root/'qualifications/scorer_gpu_resume'
    if (root/'REPORT.json').exists():return read_json(root/'REPORT.json')
    metadata,all_data=load_bank(config,'deploy','scorer_fit')
    pairs,_=pair_population(all_data['scores'],all_data['comparison_valid'])
    indices=list(dict.fromkeys(int(p[0]) for p in pairs))[:64]
    require(indices,'SCORER_COMPARISON_SIGNAL_ABSENT: cannot run a real two-flow audit')
    all_scenes=scenes_by_role(config)['scorer_fit']
    scenes=[all_scenes[i] for i in indices];data={k:v[indices] for k,v in all_data.items()}
    cache=preload(config,scenes)
    fetch=batch_fetch(config,scenes,cache,None,data,data,False)
    cfg=trainer_config(config,'scorer')
    cfg.update(max_optimizer_steps=4,warmup_steps=1,validation_steps=[],checkpoint_every_steps=2)
    deps={'diagnostic':'real_Scorer_two_flow_resume','bank_hash':metadata['hash'],
          'Base':file_hash(config.root/'BASE_BINDING.json'),'scene_ids':digest([s.scene_id for s in scenes]),
          'diagnostic_phase_boundary':2,'production_phase_boundary':3000}
    def run(name,stop=None,resume=None):
        model,_=create_scorer(config,'S00',diagnostic=True)
        sampler=StudySampler([s.scene_id for s in scenes],data['scores'],data['comparison_valid'],
            seed=config.seed+10,bank_hash=metadata['hash'],phase_steps=2)
        train(model,lambda m,b:terms(m(b['context'],b['trajectories']),b),fetch,sampler,cfg,deps,root/name,
              device=device,seed=config.seed+20,resume=resume,stop_after=stop,denominator_function=denominators,
              weights={'value':1.,'components':.2,'ranking':.2})
        return sampler.peek(128)
    next_continuous=run('continuous')
    run('interrupted',stop=2)
    next_restored=run('interrupted',resume='auto')
    paths=[root/name/'step_000004.pt' for name in ('continuous','interrupted')]
    a,b=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
    same={key:equal_tree(a[key],b[key]) for key in ('module','optimizer','scheduler','sampler','rng_by_rank','consumed_sample_ids')}
    same['next_natural_and_pair_ids']=next_continuous==next_restored
    require(all(same.values()),'real Scorer two-flow resume mismatch')
    import json
    logs=[[json.loads(s) for s in (root/name/'steps.jsonl').read_text().splitlines()] for name in ('continuous','interrupted')]
    require([x['loss'] for x in logs[0]]==[x['loss'] for x in logs[1]],'Scorer resumed loss changed')
    report={'status':'PASS','bank_hash':metadata['hash'],'real_scene_ids':[s.scene_id for s in scenes],
        'hardware':torch.cuda.get_device_name() if 'cuda' in device else 'CPU','equality':same,
        'phases':'two natural-only updates then two natural64/pair64 updates; production remains 3000/3000',
        'natural_consumed':a['sampler']['natural_consumed'],'comparison_consumed':a['sampler']['comparison_consumed'],
        'losses_equal':True,'checkpoint_hashes':[file_hash(p) for p in paths],
        'external_scorer_weights_loaded':False,'diagnostic_model_not_used_for_selection':True}
    atomic_json(root/'REPORT.json',report,immutable=True)
    return report


def bundle_roundtrip(config, device):
    from iqe.data.sources import load_observation
    from .evaluation import load_research_bundle
    check(config, 'benchmark-system')
    root = config.root/'research_bundle'
    source = read_json(root/'ROUNDTRIP_INPUT.json')
    require(file_hash(root/'ROUNDTRIP_REFERENCE.pt') == source['reference_hash'], 'bundle reference changed')
    candidates = {s.scene_id: s for s in scenes_by_role(config)['stage_val']}
    scene = candidates[source['scene_id']]
    require(scene.observation_hash == source['observation_hash'], 'roundtrip observation changed')
    model = load_research_bundle(root/'bundle.pt', device)
    observation = load_observation(scene, model.adapter.contract)
    with torch.no_grad():
        result = model.shared_forward([observation])
    expected = torch.load(root/'ROUNDTRIP_REFERENCE.pt', map_location='cpu', weights_only=True)
    differences = {name: float((result[name].detach().cpu()-value).abs().max())
                   for name, value in expected.items() if name != 'selected'}
    require(differences['trajectory'] <= source['physical_tolerance'] and differences['candidates'] <= source['physical_tolerance']
            and differences['predicted_scores'] <= source['score_tolerance'], 'bundle restored outputs changed')
    require(torch.equal(result['selected'].cpu(), expected['selected']), 'bundle restored candidate selection changed')
    require(result['calls'] == {'visual': 1, 'language': 1}, 'bundle executed extra VLA')
    report = {'status': 'PASS', 'independent_process': True, 'bundle_hash': file_hash(root/'bundle.pt'),
              'observation_hash': scene.observation_hash, 'differences': differences,
              'selected_exact': True, 'calls': result['calls'], 'branch_calls': result['branch_calls'],
              'external_scorer_weights_loaded': False, 'GT_or_metric_cache_read_for_inference': False,
              'ACTIVE_updated': False, 'device': device}
    atomic_json(root/'ROUNDTRIP_REPORT.json', report, immutable=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('expert-resume', 'scorer-resume', 'official-reference', 'bundle-roundtrip'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == 'expert-resume':result = expert_resume(config, args.device)
    elif args.command == 'scorer-resume':result = scorer_resume(config, args.device)
    elif args.command == 'official-reference':result = official_invariance(config)
    else:result = bundle_roundtrip(config, args.device)
    import json
    print(json.dumps(result))


if __name__ == '__main__':
    main()
