"""Paired old G16/K10 SDE and original ten-step ODE on locked diagnostics."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

import torch

from scripts.analysis.flow_safe_improvement.build_candidate_banks import publish_scene
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import SFT_SHA256, file_sha, fingerprint
from starVLA.rl.safe_flow_improvement.sampling import load_cached_policy, sample_scene
from starVLA.rl.safe_flow_improvement.scoring_queue import ScoringClient


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--population', required=True)
    p.add_argument('--cohorts', required=True)
    p.add_argument('--reference-scores', required=True)
    p.add_argument('--additional-reference', required=True)
    p.add_argument('--queue', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--behavior')
    p.add_argument('--behavior-sha256')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after-for-qualification', type=int)
    p.add_argument('--config', default='configs/flow_grpo/frozen_action_head_batched_single8_epoch1.yaml')
    p.add_argument('--source-config', default='/mnt/project/DriveDreamer-Policy-flow-grpo/artifacts/action-only-source/checkpoint_code/action-only-checkpoints-v1/configs/frozen-visual-step100000-config.yaml')
    a = p.parse_args()
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    configure_numerics()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    population, cohorts = [json.loads(Path(x).read_text()) for x in (a.population, a.cohorts)]
    if cohorts['population_identity'] != population['identity_sha256']:
        raise ValueError('Diagnostic population changed')
    tokens = cohorts['diagnostic_tokens']
    assert len(tokens) == 512 and set(tokens) <= set(population['train_tokens'])
    queue = json.loads((Path(a.queue)/'manifest.json').read_text())
    scorer = ScoringClient(a.queue, queue['protocol_sha256'])
    policy, store, cfg, sft, behavior = load_cached_policy(a.config, a.source_config, behavior=a.behavior, behavior_sha256=a.behavior_sha256)
    from starVLA.rl.safe_flow_improvement import sampling, policy_adapter
    identity = {'schema': 'flow_safe_D1_paired_sampler_v1', 'population_identity': population['identity_sha256'],
                'cohorts_sha256': cohorts['sha256'], 'tokens': tokens, 'SFT_sha256': SFT_SHA256,
                'behavior_sha256': behavior, 'feature_identity': store.identity_sha, 'protocol_sha256': queue['protocol_sha256'],
                'paired_root_seeds': population['diagnostic_root_seeds'], 'initial_roots': 'shared_float32_gaussian',
                'SDE_innovation_stream': 'independent_from_initial_roots',
                'ODE_single_candidate_forwards': True, 'SDE_G_K_rho_chunk': [16, 10, .8, 1],
                'source_files': {Path(x).name: file_sha(x) for x in (__file__, sampling.__file__, policy_adapter.__file__)}}
    identity['sha256'] = fingerprint(identity)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    with (out/'manifest.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (out/'manifest.json').exists():
            if json.loads((out/'manifest.json').read_text()) != identity:
                raise ValueError('D1 model/data/roots/source changed')
            if not a.resume and (out/f'rank{rank}.json').exists():
                raise ValueError('D1 requires explicit resume')
        else:
            atomic_json(out/'manifest.json', identity)
    lease = (out/f'rank{rank}.lock').open('a'); fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = dict(status='RUNNING', identity_sha256=identity['sha256'], rank=rank, world_size=world,
                  new_scenes=0, reused_scenes=0, actual_new_scores=0, reused_score_candidates=0,
                  GPU_generation_seconds=0., velocity_forward_calls=0)
    started = time.monotonic()
    for token in tokens[rank::world]:
        dst = out/'scenes'/token[:2]/(token+'.pt')
        if dst.with_suffix('.json').exists():
            seal = json.loads(dst.with_suffix('.json').read_text())
            if seal['identity_sha256'] != identity['sha256'] or seal['payload_sha256'] != file_sha(dst):
                raise ValueError('D1 committed sample changed')
            status['reused_scenes'] += 1
            continue
        log = population['token_logs'][token]
        fixed_reference = None
        for root in (a.reference_scores, a.additional_reference):
            path = Path(root)/'scenes'/token[:2]/(token+'.pt')
            if path.with_suffix('.json').exists():
                seal = json.loads(path.with_suffix('.json').read_text())
                if seal['payload_sha256'] != file_sha(path):
                    raise ValueError('Permanent reference changed')
                fixed_reference = torch.load(path, weights_only=True, map_location='cpu')
                break
        if fixed_reference is None:
            if behavior != SFT_SHA256:
                raise ValueError('Run the SFT diagnostic first to establish missing permanent references')
            rs = sample_scene(policy, store, token, population['reference_seeds'], mode='ode', role='reference', act_norm=int(sft.datasets.vla_data.act_norm))
            rs['behavior_sha256'] = SFT_SHA256
            feedback, _ = scorer.score(token, log, rs['requested_trajectories'].numpy(), caller_identity='permanent_SFT_reference_'+population['identity_sha256'], request_key='reference_4')
            fixed_reference = {'token': token, 'sample': rs, 'feedback': feedback}
            ref_id = fingerprint(['permanent_SFT_reference', population['identity_sha256'], SFT_SHA256, population['reference_seeds']])
            publish_scene(Path(a.additional_reference)/'scenes'/token[:2]/(token+'.pt'), fixed_reference, ref_id)
            status['velocity_forward_calls'] += 40
        if fixed_reference['sample']['behavior_sha256'] != SFT_SHA256:
            raise ValueError('Diagnostic reference is not the fixed SFT')
        samples = {}
        for mode in ('ode', 'legacy_sde'):
            t0 = time.monotonic()
            sample = sample_scene(policy, store, token, population['diagnostic_root_seeds'], mode=mode, role='sampler_gap', act_norm=int(sft.datasets.vla_data.act_norm))
            torch.cuda.synchronize()
            elapsed = time.monotonic()-t0
            status['GPU_generation_seconds'] += elapsed; status['velocity_forward_calls'] += 160
            feedback, budget = scorer.score(token, log, sample['requested_trajectories'].numpy(), caller_identity=identity['sha256'], request_key=mode)
            samples[mode] = {'sample': sample, 'feedback': feedback, 'budget': budget, 'generation_seconds': elapsed}
        if not torch.equal(samples['ode']['sample']['initial_noise'], samples['legacy_sde']['sample']['initial_noise']):
            raise ValueError('D1 roots are not paired')
        row = {'token': token, 'log_name': log, 'identity_sha256': identity['sha256'], 'samples': samples,
               'fixed_SFT_reference_feedback': fixed_reference['feedback'],
               'diagnostic_subpopulation': 'random' if token in population['diagnostic_random_tokens'] else 'targeted'}
        publish_scene(dst, row, identity['sha256'])
        status['new_scenes'] += 1
        status.update(actual_new_scores=scorer.actual_new_scores, reused_score_candidates=scorer.reused_scores, wall_seconds=time.monotonic()-started)
        atomic_json(out/f'rank{rank}.json', status)
        print(json.dumps(status), flush=True)
        if a.stop_after_for_qualification and status['new_scenes'] >= a.stop_after_for_qualification:
            status['status'] = 'QUALIFICATION_WINDOW_COMPLETE_FULL_POPULATION_PENDING'
            break
    else:
        status['status'] = 'ASSIGNED_POPULATION_COMPLETE'
    status['wall_seconds'] = time.monotonic()-started
    atomic_json(out/f'rank{rank}.json', status)


if __name__ == '__main__':
    main()
