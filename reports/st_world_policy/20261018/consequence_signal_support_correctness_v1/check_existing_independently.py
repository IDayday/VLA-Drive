"""Independent bounded replay of the one approved, existing C1 result ledger.

Reads RESULTS.pt exactly once. No model, checkpoint, image, GT, metric cache,
manifest, sampler, scoring backend, CUDA computation, or experiment dispatch.
Source functions are loaded via selected AST definitions only; author's main
and source import side effects are never executed.
"""
from __future__ import annotations

import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import resource
import subprocess
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
resource.setrlimit(resource.RLIMIT_CPU, (110, 110))
T0 = time.monotonic()
import numpy as np
import torch
torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
LEAD = Path('/mnt/project/VLA-Drive-st-world-policy-20261016')
AUTHOR_ROOT = 'reports/st_world_policy/20261018/consequence_next_hypothesis_v1'
COMMIT = '08ecd377a01d536a409d52e9b9a0c38325fff658'
MATH_RUNTIME = Path('/mnt/project/VLA-Drive-st-consequence-diagnostics-runtime-v2-20261018')
RUN = Path('/mnt/project/st-world-policy-20261016/formal/consequence/runs/C1_fulltrain_seed42_official_4096_v1')
KEY = 'a53a189d6603e316f5eaf2254fe85d3903e82ec27dce9167b143e6eddfff33c0'
LEDGER = RUN / 'official_feedback' / KEY
STATE = '7adaf1eb25fa0f180fd85f4d9358c02e1d6e914fe7043e8cea78e7fe3ddb756e'
NUMERIC = 'f8c10d8a1b0c0aae8a67adab92573a0222530f8dfbf2112921d5dd70b84ab0bf'
PROTOCOL = 'db383e17b887e08beb358b282fbeab223c80c558164bead81d52a45b1b798143'
RANK_HASHES = (
    '117920445b50b18d0018273aaf459982dd1cce8ac84de6155a85436bfd61f5cb',
    'dd26a76eb5d3bb7baa7c32253e51dcdff6865c58ab3c8d6b1c3b118d19994d67',
    'f7c7d4b1c9e242eabde0caee4330ee7beef802e03017772d46d6d8d6b0048528',
    '94ff1cb1381e32e4b1570a94ce3e45cb9ff1c3ff41317ef7472147b4f3b1acd6',
)
reads = []
checks = []


def check(value, name):
    if not value:
        raise AssertionError(name)
    checks.append(name)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def read(path, expected=None):
    path = Path(path)
    size = path.stat().st_size
    check(sum(r['bytes'] for r in reads) + size <= 16 << 20, 'bounded logical input bytes')
    raw = path.read_bytes()
    check(len(raw) == size, 'stable file size')
    h = sha(raw)
    if expected:
        check(h == expected, 'pinned checksum: ' + path.name)
    reads.append({'path': str(path), 'bytes': len(raw), 'sha256': h})
    return raw


def read_json(path, expected=None):
    return json.loads(read(path, expected))


def load_defs(source, names, namespace):
    tree = ast.parse(source)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    check({node.name for node in selected} == set(names), 'exact AST function allowlist')
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<reviewed AST functions>', 'exec'), namespace)
    return namespace


def array_hash(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(json.dumps({'dtype': a.dtype.str, 'shape': a.shape}, sort_keys=True).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def main():
    author = {}
    for name in ('analyze_existing.py', 'PLAN.md', 'RESULTS.json', 'REPORT.md', 'RUNBOOK.md'):
        relative = AUTHOR_ROOT + '/' + name
        raw = read(LEAD / relative)
        pinned = subprocess.run(['git', '-C', str(LEAD), 'show', COMMIT + ':' + relative],
                                check=True, capture_output=True).stdout
        check(raw == pinned, 'current author artifact equals exact integrated commit: ' + name)
        author[name] = raw
    report = json.loads(author['RESULTS.json'])
    check(sha(author['analyze_existing.py']) == report['analyzer_sha256'], 'author analyzer self-binding')
    check(sha(author['PLAN.md']) == report['plan_sha256'], 'author preregistered plan binding')
    original_sources = {}
    for relative in (
        'starVLA/model/modules/flow_posttraining/fpo_objective.py',
        'tools/st_world_policy/train_flow_policy.py',
        'navsim_v1.1/navsim/navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py',
    ):
        original_sources[relative] = read(MATH_RUNTIME / relative)
    fpo = load_defs(original_sources['starVLA/model/modules/flow_posttraining/fpo_objective.py'],
                    ('_float_math', 'group_centered_advantages'), {'torch': torch})
    original_center = fpo['group_centered_advantages']
    # Frozen author controls are supplemental; all headline statistics below
    # are independently recomputed from real records, not these fixtures.
    namespace = load_defs(author['analyze_existing.py'],
        ('require', 'bin_of', 'process_profile', 'advantages', 'decompose', 'tie_counts', 'synthetic_checks'),
        {'np': np, 'torch': torch, 'itertools': itertools})

    run_identity = read_json(RUN / 'identity.json', '15535618f98c13c9b5a5929607a67012fbc5bc88f394ab353f70ebec37655930')
    request_doc = read_json(LEDGER / 'REQUESTS.json', 'e6a0400b7d22ed38f8a26cd11f1ced8f68d8dec97c06fca85a78439d380ba9b6')
    complete = read_json(LEDGER / 'attempt_00/COMPLETE.json', 'e1d40b12cd35e022be2ecd309b7f7195c5b53133bf66c724f13a362e850e863f')
    admitted = read_json(LEDGER / 'attempt_00/ADMITTED.json', complete['admission_sha256'])
    ranks = [read_json(RUN / 'rollouts/boundary_003584' / f'rank_{r}_COMPLETE.json', h)
             for r, h in enumerate(RANK_HASHES)]
    request = request_doc['requests']
    check(request_doc['policy_version'] == '3584' and identity(request) == KEY, 'boundary and request digest')
    check(admitted['calls'] == 128 and admitted['indices'] == list(range(128)) and
          admitted['request_hash'] == KEY and admitted['policy_version'] == '3584', 'all 128 original admitted indices')
    result_path = LEDGER / 'attempt_00/RESULTS.pt'
    result_hash = '8019dd29b17efa79c1fd1a7211bdf09a0b5be93931db75cd4db95ad9a3fb122f'
    check(complete['result'] == {'path': str(result_path), 'sha256': result_hash}, 'complete receipt result binding')
    raw = read(result_path, result_hash)  # This is the only real ledger read.
    check(len(raw) == 3441888, 'exact real ledger byte size')
    records = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
    check(len(records) == len(request) == 128, 'complete real candidate population')
    check(len({x['scene_id'] for x in request}) == len({x['source_log'] for x in request}) == 32, '32 real scenes/logs')
    check(run_identity['data'] == report['identities']['training_manifest_identity_from_run_receipt'], 'training identity receipt')
    profiles = []
    metrics = []
    for i, (req, rec) in enumerate(zip(request, records)):
        rank, slot, cand = i // 32, (i % 32) // 4, i % 4
        saved = ranks[rank]['identity']
        check(rec['identity'] == req and rec['status'] == 'VALID_SCORED' and rec['valid'] is True,
              f'real item {i}: identity and score validity')
        check(req['candidate_id'] == f"{saved['rollout_id']}:{slot}:{cand}" and
              req['scene_id'] == request[4*(i//4)]['scene_id'], f'real item {i}: rank scene candidate order')
        check(saved['old_policy']['state_hash'] == STATE and
              saved['old_policy']['version'] == saved['old_policy']['optimizer_step'] == 3584 and
              saved['old_policy']['base_binding_hash'] == report['identities']['base_binding'], f'real item {i}: frozen policy')
        check(saved['numeric_execution_hash'] == req['inference_runtime_hash'] == NUMERIC and
              req['policy_version'] == '3584' and req['generator_identity']['state_sha256'] == STATE and
              req['score_protocol_hash'] == PROTOCOL, f'real item {i}: numeric and scoring protocol')
        check(saved['feedback']['statuses'][4*slot+cand] == 'valid_scored' and
              float(np.float32(rec['metrics']['score'])) == saved['feedback']['rewards'][slot][cand], f'real item {i}: saved training reward')
        check(saved['feedback']['contexts'][slot] == req['metric_cache_sha256'] and
              rec['score_units'] == '0..1' and rec['context_unchanged'] is True, f'real item {i}: units context')
        check(all(rec['metrics_valid'][k] and np.isfinite(v) and 0 <= v <= 1
                  for k, v in rec['metrics'].items()), f'real item {i}: finite valid components')
        check(array_hash(rec['requested_trajectory']) == req['trajectory_sha256'], f'real item {i}: trajectory checksum')
        reference = rec['reference_contract']
        inp, out = reference['input'], reference['execution']
        check(reference['schema'] == 'st_world_reference_input_and_execution_v2' and
              identity(inp) == reference['input_sha256'] and identity(out) == reference['execution_sha256'], f'real item {i}: reference hashes')
        check(inp == records[4*(i//4)]['reference_contract']['input'] and
              inp['full_cache_sha256'] == req['metric_cache_sha256'] and
              inp['score_protocol_hash'] == PROTOCOL and inp['context_hash'] == rec['context_hash'], f'real item {i}: fixed reference context')
        check(array_hash(rec['reference_interpolated_states']) == inp['reference_interpolated_states_sha256'] and
              array_hash(rec['executed_states']) == out['candidate_executed_states_sha256'], f'real item {i}: execution checksums')
        # Independently inspect the actual native process traces. The approved
        # population happens to be empty of events; do not infer rates on a
        # non-existent positive-event population.
        c = rec['consequences']
        flags = np.asarray(c['road']['native_flag'])
        check(flags.dtype == np.bool_ and flags.shape == (41,) and not flags.any(), f'real item {i}: 41 empty road flags')
        check(not c['road']['any'] and not c['road']['event'] and c['road']['first_time_seconds'] is None and
              c['road']['segment_any'] == [False]*8, f'real item {i}: road summary')
        check(not [e for e in c['collision_events'] if e['at_fault']] and
              not [e for e in c['ttc_events'] if e['violation']], f'real item {i}: no physical safety event')
        check(c['collision_summary']['segment_nc_min'] == [1.]*8 and
              c['collision_summary']['nc_min'] == 1. and not c['collision_summary']['event'] and
              c['collision_summary']['first_at_fault_time_seconds'] is None, f'real item {i}: collision summary')
        check(c['ttc_summary']['segment_any'] == [False]*8 and not c['ttc_summary']['any'] and
              not c['ttc_summary']['event'] and c['ttc_summary']['first_state_time_seconds'] is None, f'real item {i}: TTC summary')
        profiles.append(namespace['process_profile'](rec))
        metrics.append(rec['metrics'])

    m = {k: np.array([r[k] for r in metrics], dtype=np.float64).reshape(32, 4) for k in metrics[0]}
    nc, dac, ep, ttc, comfort = (m[k] for k in ('no_at_fault_collisions', 'drivable_area_compliance',
                                             'ego_progress', 'time_to_collision_within_bound', 'comfort'))
    product = nc*dac
    rebuilt = product*(5*ep + 5*ttc + 2*comfort)/12
    reconstruction_error = float(np.max(np.abs(rebuilt-m['score'])))
    check(reconstruction_error <= 1e-12, 'original official multiplicative-weighted score reconstruction')
    check(all(np.array_equal(x, np.ones((32, 4))) for x in (nc, dac, ttc, comfort)), 'all 128 original safety/comfort terms equal one')
    reward = torch.tensor(m['score'], dtype=torch.float32)
    A = original_center(reward, torch.ones_like(reward, dtype=torch.bool)).numpy().astype(np.float64)
    check(np.array_equal(A, namespace['advantages'](m['score'])), 'author center equals actual FPO sum/count implementation on CPU')
    active = np.any(A != 0, axis=1)
    ep_only = active & (np.ptp(ep, axis=1) > 0)
    check(int(active.sum()) == 16 and int(ep_only.sum()) == 16 and np.count_nonzero(A) == 64, 'real EP-only active counts')
    original_other = {'C2': dac-1., 'C3': nc-1., 'C4': ttc-1.}
    check(all(not original_center(torch.tensor(x, dtype=torch.float32), torch.ones((32, 4), dtype=torch.bool)).any()
              for x in original_other.values()), 'same-population C2/C3/C4 zero centered advantage')
    # Compute covariance terms independently in scalar scene loops, including
    # residual against actual float32-center arithmetic. Products are gated
    # contributions, not independent causal factors.
    terms = np.stack((product*ep*5/12, product*ttc*5/12, product*comfort*2/12), axis=-1)
    centered = np.empty_like(terms)
    for scene in range(32):
        for component in range(3):
            centered[scene, :, component] = terms[scene, :, component] - sum(terms[scene, :, component])/4.
    rounding = A - centered.sum(-1)
    parts = np.concatenate((centered, rounding[..., None]), -1)
    V = float(np.sum(A**2))
    cov = np.array([sum(float(A[s, c]*parts[s, c, k]) for s in range(32) for c in range(4)) for k in range(4)])
    expected = report['advantage_value_decomposition']
    check(V == expected['advantage_squared_sum'], 'actual advantage second moment')
    check(np.allclose(cov, list(expected['signed_covariance_numerators'].values()), atol=1e-18, rtol=1e-12), 'independent signed covariance numerators')
    check(np.allclose(cov/V, list(expected['signed_covariance_fractions'].values()), atol=1e-12, rtol=1e-12), 'independent signed covariance fractions')
    check(float(np.max(np.abs(rounding))) == expected['max_absolute_rounding_residual'] and
              np.array_equal(A, parts.sum(-1)), 'float32 rounding retained exact additive reconstruction')
    check(cov[1] == cov[2] == 0 and cov[3] < 0 and cov[0]/V > 1, 'signed fractions not clipped or mislabeled causal')

    pairs = [(s, a, b) for s in range(32) for a, b in itertools.combinations(range(4), 2)]
    check(len(pairs) == 32*6 == 192 and len(set(pairs)) == 192, 'all original unordered within-scene pairs')
    float32_ties = [(s,a,b) for s,a,b in pairs if reward[s,a] == reward[s,b]]
    float64_ties = [(s,a,b) for s,a,b in pairs if m['score'][s,a] == m['score'][s,b]]
    check(float32_ties == float64_ties and len(float32_ties) == 96 and
          len({s for s,_,_ in float32_ties}) == 16, 'real exact ties independent enumeration')
    actual_ties = namespace['tie_counts'](records, profiles)
    check(actual_ties == report['terminal_ties'], 'all real pair summaries exactly match published report')
    check(all(v['nonempty_event_pairs'] == 0 and v['conditional_time_difference_fraction_given_nonempty'] is None
              for v in actual_ties.values()), 'zero event denominator remains undefined')
    check(all(v['eligible_pairs'] == 192 for k,v in actual_ties.items() if k not in ('C1_float32_tie','C1_float64_tie')), 'safety tie denominator 192')
    for s in range(32):
        row = report['per_scene'][s]
        check(row['scene_id'] == request[4*s]['scene_id'] and row['source_log'] == request[4*s]['source_log'] and
              row['original_rank'] == s//8 and row['rank_scene_slot'] == s%8 and
              row['C1_nonzero_advantage'] == bool(active[s]) and row['EP_only'] == bool(ep_only[s]) and
              row['C1_advantages_float32'] == A[s].tolist(), f'all real per-scene fields preserved: {s}')

    supplemental = namespace['synthetic_checks']()
    check(supplemental == report['synthetic_controls'], 'nine supplemental control results reproduced')
    # Independent exact matrix proof; no finite-difference coincidence needed.
    # Both physical trajectory points depend on every one of ten flow updates.
    J = np.vstack((np.full(10, .1), np.full(10, .2)))
    physical_derivative = np.array([1., 2.])
    all_steps = physical_derivative @ J
    check(np.all(all_steps == .5) and np.count_nonzero(all_steps) == 10, 'analytic physical-time versus flow-index counterexample')
    check(not torch.cuda.is_initialized(), 'CUDA not initialized')
    usage = resource.getrusage(resource.RUSAGE_SELF)
    check(usage.ru_maxrss*1024 <= 2 << 30 and usage.ru_utime+usage.ru_stime <= 120, 'independent review CPU/RSS limits')
    check(Counter(r['path'] for r in reads)[str(result_path)] == 1, 'real feedback ledger read exactly once')
    out = {
        'schema': 'independent_consequence_signal_support_replay_v1',
        'status': 'PASS_BOUNDED_EXISTING_REAL_LEDGER_REVIEW',
        'reviewer': '/root/road_proxy_correctness_review',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_author_commit': 'cafc25a6096703b78a83b1b0d803a480545111af',
        'source_integrated_commit': COMMIT,
        'source_files_sha256': {name: sha(raw) for name, raw in author.items()},
        'original_formula_sources_sha256': {name: sha(raw) for name,raw in original_sources.items()},
        'real_result_sha256': result_hash, 'real_result_bytes': len(raw),
        'population': report['population'],
        'recomputed': {'C1_nonzero_scenes': int(active.sum()), 'C1_nonzero_candidates': int(np.count_nonzero(A)),
                       'EP_only_scenes': int(ep_only.sum()), 'C2_C3_C4_nonzero_scenes': [0,0,0],
                       'D_C_T_event_candidates': [0,0,0], 'pair_denominator': len(pairs),
                       'float32_and_float64_tie_pairs': len(float32_ties),
                       'safety_tie_pairs': 192, 'conditional_process_difference': None,
                       'advantage_squared_sum': V, 'signed_covariance_numerators': cov.tolist(),
                       'signed_covariance_fractions': (cov/V).tolist(),
                       'rounding_residual_max': float(np.abs(rounding).max()),
                       'official_reconstruction_max_error': reconstruction_error},
        'checks_passed': len(checks),
        'supplemental_synthetic_controls': supplemental,
        'formal_training_approved': False,
        'scope_limitations': [
            'Only this fixed C1 TRAIN boundary3584 ledger; no full-TRAIN or heldout generalization claim',
            'TRAIN provenance bound to small original receipts; full training manifest deliberately not opened',
            'CPU replay of the original float32 center arithmetic; no new CUDA bitwise-center or parameter-gradient claim',
            'C2/C3/C4 here are counterfactual reward definitions on C1 candidates, not results of their trained policies',
            'No event-positive pair exists; conditional temporal information comparison is undefined, not evidence of temporal uselessness',
            'Signed reward-value covariance is not causal contribution, gradient norm, or optimizer update',
            'Toy analytic counterexample disproves a universal one-to-one time mapping, not an actual NAVSIM Jacobian',
        ],
        'read_ledger': reads,
        'execution': {'logical_read_bytes': sum(r['bytes'] for r in reads), 'real_ledger_reads': 1,
                      'wall_seconds': time.monotonic()-T0, 'cpu_seconds': usage.ru_utime+usage.ru_stime,
                      'peak_rss_bytes': usage.ru_maxrss*1024, 'torch_threads': torch.get_num_threads(),
                      'cuda_initialized': torch.cuda.is_initialized(), 'new_score_calls': 0,
                      'model_loads': 0, 'optimizer_updates': 0, 'full_manifest_reads': 0,
                      'DEV_Navtest_reads': 0, 'metric_cache_reads': 0},
    }
    target = HERE/'REPLAY_RESULT.json'
    check(not target.exists(), 'independent output immutable')
    payload = json.dumps(out, indent=2, ensure_ascii=False, allow_nan=False).encode()+b'\n'
    check(len(payload) <= 2 << 20, 'independent report size budget')
    target.write_bytes(payload)
    print(json.dumps({'status': out['status'], 'checks_passed': len(checks), 'recomputed': out['recomputed'],
                      'execution': out['execution'], 'output_sha256': sha(payload)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
