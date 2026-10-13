"""Bounded real-head ODE batching profile with canonical per-candidate checks.

Only exploration/bank throughput can use a passing chunk. Deployment and
reference sampling remain the original single-candidate call. This does not
qualify legacy SDE logprob batching or a new training precision profile.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import SAFETY_METRICS, file_sha, fingerprint, seed_for
from starVLA.rl.safe_flow_improvement.policy_adapter import encode_condition, sample_ode, decode_request
from starVLA.rl.safe_flow_improvement.sampling import load_cached_policy
from starVLA.rl.safe_flow_improvement.scoring_queue import ScoringClient


@torch.no_grad()
def grouped(policy, condition, roots, chunk):
    values = []
    for start in range(0, len(roots), chunk):
        noise = roots[start:start+chunk]
        values.append(sample_ode(policy, condition.expand(len(noise), -1, -1), noise))
    return torch.cat(values)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--population', required=True)
    p.add_argument('--queue', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--config', default='configs/flow_grpo/frozen_action_head_batched_single8_epoch1.yaml')
    p.add_argument('--source-config', default='/mnt/project/DriveDreamer-Policy-flow-grpo/artifacts/action-only-source/checkpoint_code/action-only-checkpoints-v1/configs/frozen-visual-step100000-config.yaml')
    a = p.parse_args()
    configure_numerics()
    torch.cuda.set_device(0)
    population = json.loads(Path(a.population).read_text())
    tokens = population['calibration_tokens'][:32]
    assert set(tokens) <= set(population['train_tokens'])
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    registration = {'source_sha256': file_sha(__file__), 'tokens_sha256': fingerprint(tokens),
                    'population_identity': population['identity_sha256'], 'scenes': 32, 'candidates': 16,
                    'chunks': [1, 4, 8, 16], 'normalized_action_max_abs_tolerance': 1e-5,
                    'requested_pose_max_abs_tolerance': 1e-4,
                    'official_raw_score_max_abs_tolerance': 1e-5,
                    'NC_DAC_TTC_require_identical': True,
                    'scope': 'Exploration and fixed-bank ODE only; no change to deployment/reference or legacy SDE'}
    registration['sha256'] = fingerprint(registration)
    atomic_json(out / 'registration.json', registration)
    q = json.loads((Path(a.queue) / 'manifest.json').read_text())
    scorer = ScoringClient(a.queue, q['protocol_sha256'])
    policy, store, _, sft, _ = load_cached_policy(a.config, a.source_config)
    rows = []
    with torch.no_grad():
        for token in tokens:
            raw = store.get(token)['raw'].cuda()
            condition = encode_condition(policy, raw)
            roots = torch.randn((16, 8, 4), dtype=raw.dtype, device='cuda',
                                generator=torch.Generator(device='cuda').manual_seed(seed_for('batch_qualification_root', 42, token)))
            baseline, base_request, base_feedback = None, None, None
            for chunk in registration['chunks']:
                torch.cuda.synchronize(); started = time.monotonic()
                actions = grouped(policy, condition, roots, chunk)
                torch.cuda.synchronize(); elapsed = time.monotonic()-started
                actions = actions.float().cpu().numpy()
                request = decode_request(actions, act_norm=int(sft.datasets.vla_data.act_norm))
                feedback, budget = scorer.score(token, population['token_logs'][token], request,
                                                caller_identity=registration['sha256'], request_key=f'chunk{chunk}')
                if chunk == 1:
                    baseline, base_request, base_feedback = actions, request, feedback
                row = {'token': token, 'chunk': chunk, 'seconds': elapsed, 'velocity_forward_calls': 10*((16+chunk-1)//chunk),
                       'max_abs_action': float(np.max(np.abs(actions-baseline))),
                       'max_abs_request': float(np.max(np.abs(request-base_request))),
                       'max_abs_score': max(abs(x['metrics']['score']-y['metrics']['score']) for x, y in zip(feedback, base_feedback)),
                       'safety_components_identical': all(x['metrics'][k] == y['metrics'][k] for x, y in zip(feedback, base_feedback) for k in SAFETY_METRICS),
                       'budget': budget}
                rows.append(row)
                atomic_json(out / 'measurements.json', rows)
            print(json.dumps({'completed_scenes': len(rows)//4, 'requested_scenes': len(tokens)}), flush=True)
    summaries = {}
    for chunk in registration['chunks']:
        data = [r for r in rows if r['chunk'] == chunk]
        passed = all(r['max_abs_action'] <= 1e-5 and r['max_abs_request'] <= 1e-4 and r['max_abs_score'] <= 1e-5 and r['safety_components_identical'] for r in data)
        summaries[str(chunk)] = {'passed': passed, 'seconds': sum(r['seconds'] for r in data),
                                 'action_max_abs': max(r['max_abs_action'] for r in data),
                                 'request_max_abs': max(r['max_abs_request'] for r in data),
                                 'score_max_abs': max(r['max_abs_score'] for r in data)}
    passing = [c for c in registration['chunks'] if summaries[str(c)]['passed']]
    selected = min(passing, key=lambda c: summaries[str(c)]['seconds'])
    result = {'status': 'COMPLETE', 'registration': registration, 'summaries': summaries,
              'selected_exploration_chunk': selected, 'deployment_chunk': 1,
              'actual_scoring_calls': scorer.actual_new_scores, 'reference_SDE_training_numeric_qualification': 'NOT_CHANGED_NOT_CLAIMED'}
    atomic_json(out / 'report.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
