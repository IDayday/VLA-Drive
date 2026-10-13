"""Check the actual 7/12/15 candidate batch sizes under the fixed ODE tolerances."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.analysis.flow_safe_improvement import qualify_rollout_batch as original


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('population', 'queue', 'output'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    original.configure_numerics(); torch.cuda.set_device(0)
    pop = json.loads(Path(a.population).read_text()); tokens = pop['calibration_tokens'][:32]
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    registration = dict(scenes=32, baseline_candidates=16, effective_batch_sizes=[7, 12, 15],
                        tokens_sha256=original.fingerprint(tokens), population_identity=pop['identity_sha256'],
                        source_sha256=original.file_sha(__file__), grouped_source_sha256=original.file_sha(original.__file__),
                        normalized_action_max_abs_tolerance=1e-5, requested_pose_max_abs_tolerance=1e-4,
                        official_raw_score_max_abs_tolerance=1e-5, NC_DAC_TTC_require_identical=True,
                        reason='The earlier check used chunks 1/4/8/16; actual draws per call in D/P/O are 7/12/15.',
                        qualification_scope='SFT actual exploration shapes; deployment remains chunk1; no changed training layout')
    registration['sha256'] = original.fingerprint(registration)
    original.atomic_json(out/'registration.json', registration)
    queue = json.loads((Path(a.queue)/'manifest.json').read_text())
    scorer = original.ScoringClient(a.queue, queue['protocol_sha256'])
    policy, store, _, sft, _ = original.load_cached_policy(
        'configs/flow_grpo/frozen_action_head_batched_single8_epoch1.yaml',
        '/mnt/project/DriveDreamer-Policy-flow-grpo/artifacts/action-only-source/checkpoint_code/action-only-checkpoints-v1/configs/frozen-visual-step100000-config.yaml')
    rows = []
    with torch.no_grad():
        for token in tokens:
            raw = store.get(token)['raw'].cuda(); condition = original.encode_condition(policy, raw)
            roots = torch.randn((16, 8, 4), device='cuda', dtype=raw.dtype,
                                generator=torch.Generator(device='cuda').manual_seed(original.seed_for('batch_qualification_root', 42, token)))
            reference = original.grouped(policy, condition, roots, 1).float().cpu().numpy()
            requests = original.decode_request(reference, act_norm=int(sft.datasets.vla_data.act_norm))
            feedback, _ = scorer.score(token, pop['token_logs'][token], requests, caller_identity=registration['sha256'], request_key='single16')
            for count in registration['effective_batch_sizes']:
                torch.cuda.synchronize(); started = time.monotonic()
                actions = original.grouped(policy, condition, roots[:count], 16).float().cpu().numpy()
                torch.cuda.synchronize(); seconds = time.monotonic()-started
                request = original.decode_request(actions, act_norm=int(sft.datasets.vla_data.act_norm))
                scored, budget = scorer.score(token, pop['token_logs'][token], request, caller_identity=registration['sha256'], request_key=f'effective{count}')
                rows.append(dict(token=token, effective_batch=count, seconds=seconds,
                                 max_abs_action=float(np.abs(actions-reference[:count]).max()),
                                 max_abs_request=float(np.abs(request-requests[:count]).max()),
                                 max_abs_score=max(abs(x['metrics']['score']-y['metrics']['score']) for x, y in zip(scored, feedback[:count])),
                                 safety_identical=all(x['metrics'][k] == y['metrics'][k] for x, y in zip(scored, feedback[:count]) for k in original.SAFETY_METRICS), budget=budget))
            original.atomic_json(out/'measurements.json', rows)
    summary = {}
    for count in registration['effective_batch_sizes']:
        selected = [r for r in rows if r['effective_batch'] == count]
        summary[str(count)] = dict(passed=all(r['max_abs_action']<=1e-5 and r['max_abs_request']<=1e-4 and r['max_abs_score']<=1e-5 and r['safety_identical'] for r in selected),
                                   action_max_abs=max(r['max_abs_action'] for r in selected), request_max_abs=max(r['max_abs_request'] for r in selected),
                                   score_max_abs=max(r['max_abs_score'] for r in selected), safety_identical=all(r['safety_identical'] for r in selected))
    report = dict(status='PASS' if all(x['passed'] for x in summary.values()) else 'FAIL_STOP_AFFECTED_SHAPES',
                  registration=registration, summaries=summary, actual_scoring_calls=scorer.actual_new_scores)
    original.atomic_json(out/'report.json', report); print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
