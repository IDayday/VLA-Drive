"""Exactly four CPU-only replays of the immutable failed profile group.

Read-only hooks retain native LQR inputs/outputs. No canonical source changes,
no model loading, no score/trajectory repair, and no extra simulation probes.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import torch

from starVLA.model.modules.st_world.consequence import (
    ConsequenceEngine, ConsequenceRequest, ConsequenceSpec, array_sha256,
    file_sha256, protocol_identity,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or any(os.environ.get(k) != "1" for k in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise RuntimeError("Require CPU only and inherited BLAS1")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "ADMITTED.json").exists():
        raise RuntimeError("Four-call diagnostic already admitted; no automatic re-scoring")
    original = Path('/mnt/project/st-world-policy-20261016/profiles/flow_official_runtime_8gpu_v1')
    run = original / 'runs/flow_official_execution_profile240_v1'
    ledger = run / 'official_feedback/d8019cf3e07dcf8c79e099cd8ca9205b0a058b72b85f16f8f270b6dfed3f4af9'
    receipt = json.loads((ledger / 'attempt_00/COMPLETE.json').read_text())
    result_file = Path(receipt['result']['path'])
    if file_sha256(result_file) != receipt['result']['sha256']:
        raise RuntimeError('Original result checksum mismatch')
    all_rows = torch.load(result_file, map_location='cpu', weights_only=False)
    rows = all_rows[56:60]
    if any(r['identity']['scene_id'] != 'd40a2f8b287e527e' for r in rows):
        raise RuntimeError('Unexpected failed scene')
    feedback = json.loads((original / 'CONFIG.json').read_text())['feedback']
    spec = ConsequenceSpec(**{k: feedback[k] for k in
        ('navsim_root', 'maps_root', 'cache_roots', 'campaign_id', 'score_protocol_hash')})
    protocol = protocol_identity(spec.navsim_root)
    if protocol['sha256'] != spec.score_protocol_hash:
        raise RuntimeError('Must replay exact original CPU scorer environment/protocol')
    from starVLA.model.modules.st_world import consequence
    registered = {
        'purpose': 'engineering_root_cause_only', 'registered_unix': time.time(),
        'max_official_calls': 4, 'cpu_workers': 1, 'gpus': 0, 'blas_threads': 1,
        'source_file': str(Path(consequence.__file__).resolve()),
        'source_sha256': file_sha256(Path(consequence.__file__)),
        'script_sha256': file_sha256(Path(__file__)),
        'original_result': receipt['result'], 'original_requests_sha256': file_sha256(ledger / 'REQUESTS.json'),
        'policy_version': '162', 'original_indices': [56, 57, 58, 59],
        'requests': [r['identity'] for r in rows], 'protocol_sha256': protocol['sha256'],
    }
    (out / 'ADMITTED.json').write_text(json.dumps(registered, indent=2, sort_keys=True) + '\n')
    records, traces, references = [], [], []
    for old in rows:
        # Fresh instances prove independence from the previous worker's scene.
        engine = ConsequenceEngine(spec)
        request = ConsequenceRequest(**old['identity'], trajectory=old['requested_trajectory'], observe=True)
        tracker = engine.simulator._tracker
        trace = {'track': [], 'lateral': [], 'solve': [], 'reference_profile': []}
        current = {'t': None}
        original_track = tracker.track_trajectory
        original_lateral = tracker._lateral_lqr_controller
        original_solve = tracker._solve_one_step_lateral_lqr
        original_profile = tracker._compute_reference_velocity_and_curvature_profile
        def track(iteration, next_iteration, states):
            current['t'] = iteration.index
            result = original_track(iteration, next_iteration, states)
            trace['track'].append({'t': iteration.index, 'states': states.copy(), 'commands': result.copy()})
            return result
        def lateral(initial, velocity, curvature):
            result = original_lateral(initial, velocity, curvature)
            trace['lateral'].append({'t': current['t'], 'initial': initial.copy(),
                'velocity': velocity.copy(), 'curvature': curvature.copy(), 'output': result.copy()})
            return result
        def solve(**kwargs):
            result = original_solve(**kwargs)
            trace['solve'].append({'t': current['t'], **{k:v.copy() for k,v in kwargs.items()}, 'output': result.copy()})
            return result
        def profile(iteration):
            result = original_profile(iteration)
            trace['reference_profile'].append({'t': iteration.index,
                'velocity': result[0].copy(), 'curvature': result[1].copy()})
            return result
        tracker.track_trajectory = track
        tracker._lateral_lqr_controller = lateral
        tracker._solve_one_step_lateral_lqr = solve
        tracker._compute_reference_velocity_and_curvature_profile = profile
        result = engine.score(request, '162')
        if not result['valid'] or result['cost']['scoring_calls'] != 1:
            raise RuntimeError(f'Diagnostic failed: {result.get("error")}')
        records.append(result)
        traces.append(trace)
        references.append(engine.scorer._states[0].copy())
    with (out / 'FOUR_REPLAYS.pkl').open('xb') as stream:
        pickle.dump({'records':records, 'traces':traces, 'reference_states':references},stream,protocol=4)
    reports=[]
    for i,(old,new) in enumerate(zip(rows,records)):
        diff=references[i]-references[0]
        different=np.argwhere(references[i] != references[0])
        reports.append({'candidate':i,'metrics_exact_to_original':old['metrics']==new['metrics'],
            'executed_candidate_exact':np.array_equal(old['executed_states'],new['executed_states']),
            'canonical_arrays_exact':all(np.array_equal(old['canonical_arrays'][k],new['canonical_arrays'][k]) for k in old['canonical_arrays']),
            'context_exact':old['context_hash']==new['context_hash'],
            'reference_identity_exact_to_original':old['reference_identity']==new['reference_identity'],
            'reference_executed_sha256':array_sha256(references[i]),
            'reference_max_abs_diff_from0':float(np.max(np.abs(diff))),
            'reference_different_cells':different.tolist(),
            'regular_controller_batch_sizes':[len(r['output']) for r in traces[i]['lateral']]})
    (out/'SUMMARY.json').write_text(json.dumps({'status':'FOUR_REPLAYS_COMPLETE','calls':4,
        'reports':reports,'payload_sha256':file_sha256(out/'FOUR_REPLAYS.pkl')},indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'FOUR_REPLAYS_COMPLETE','calls':4,'reports':reports}))


if __name__ == '__main__':
    main()
