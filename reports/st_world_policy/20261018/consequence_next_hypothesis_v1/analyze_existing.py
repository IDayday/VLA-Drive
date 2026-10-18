"""Bounded CPU analysis of one immutable existing TRAIN feedback batch.

No repository model import, optimizer, autograd, sampler, scorer, environment
cache or data manifest. PLAN.md must exist before this program is executed.
"""
from __future__ import annotations

import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import resource
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
START = time.monotonic()

import numpy as np
import torch
torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
RUN = Path('/mnt/project/st-world-policy-20261016/formal/consequence/runs/C1_fulltrain_seed42_official_4096_v1')
KEY = 'a53a189d6603e316f5eaf2254fe85d3903e82ec27dce9167b143e6eddfff33c0'
LEDGER = RUN / 'official_feedback' / KEY
RANK_HASHES = (
    '117920445b50b18d0018273aaf459982dd1cce8ac84de6155a85436bfd61f5cb',
    'dd26a76eb5d3bb7baa7c32253e51dcdff6865c58ab3c8d6b1c3b118d19994d67',
    'f7c7d4b1c9e242eabde0caee4330ee7beef802e03017772d46d6d8d6b0048528',
    '94ff1cb1381e32e4b1570a94ce3e45cb9ff1c3ff41317ef7472147b4f3b1acd6',
)
STATE_HASH = '7adaf1eb25fa0f180fd85f4d9358c02e1d6e914fe7043e8cea78e7fe3ddb756e'
NUMERIC_HASH = 'f8c10d8a1b0c0aae8a67adab92573a0222530f8dfbf2112921d5dd70b84ab0bf'
PROTOCOL_HASH = 'db383e17b887e08beb358b282fbeab223c80c558164bead81d52a45b1b798143'
METRICS = ('score', 'no_at_fault_collisions', 'drivable_area_compliance',
           'ego_progress', 'time_to_collision_within_bound', 'comfort', 'driving_direction_compliance')


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def identity(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def array_hash(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(json.dumps({'dtype': a.dtype.str, 'shape': a.shape}, sort_keys=True).encode())
    h.update(a.tobytes())
    return h.hexdigest()


class ReadBudget:
    def __init__(self):
        self.total = 0
        self.entries = []

    def read(self, path, expected=None):
        path = Path(path)
        size = path.stat().st_size
        require(self.total + size <= 32 << 20, 'Task asset read budget exceeded')
        raw = path.read_bytes()
        self.total += len(raw)
        sha = digest(raw)
        self.entries.append({'path': str(path), 'bytes': len(raw), 'sha256': sha})
        require(len(raw) == size and (expected is None or sha == expected), 'Artifact size/hash mismatch')
        return raw

    def json(self, path, expected=None):
        return json.loads(self.read(path, expected))


def bin_of(index):
    require(type(index) is int and 0 <= index <= 40, 'Invalid physical state index')
    return max(0, (index - 1) // 5)


def process_profile(record):
    """Time signatures ignore object identity and repeated contacts at one time."""
    c = record['consequences']
    road = c['road']
    require(road['valid'] is True and len(road['native_flag']) == 41, 'Invalid road trace')
    require(np.asarray(road['native_flag']).dtype == np.bool_, 'Road trace lost bool semantics')
    d_times = tuple(int(x) for x in np.flatnonzero(road['native_flag']))
    d_bins = [False] * 8
    for t in d_times:
        d_bins[bin_of(t)] = True
    require(d_bins == road['segment_any'] and bool(d_times) == road['any'] == road['event'], 'Road aggregation mismatch')
    require(road['first_time_seconds'] == (d_times[0] * .1 if d_times else None), 'Road first time mismatch')

    c_events = [e for e in c['collision_events'] if e['at_fault']]
    c_signature = tuple(sorted({(int(e['state_index']), float(e['nc_contribution'])) for e in c_events}))
    c_bins = [1.] * 8
    for t, severity in c_signature:
        require(severity in (0., .5), 'At-fault severity changed')
        c_bins[bin_of(t)] = min(c_bins[bin_of(t)], severity)
    for e in c_events:
        require(e['state_time_seconds'] == int(e['state_index']) * .1, 'Collision physical time mismatch')
    cs = c['collision_summary']
    require(cs['valid'] is True and cs['segment_nc_min'] == c_bins and cs['nc_min'] == min(c_bins), 'NC aggregation mismatch')
    require(cs['event'] == bool(c_signature), 'NC event flag mismatch')
    require(cs['first_at_fault_time_seconds'] == (min(t for t, _ in c_signature) * .1 if c_signature else None), 'NC first time mismatch')

    t_events = [e for e in c['ttc_events'] if e['violation']]
    t_times = tuple(sorted({int(e['state_index']) for e in t_events}))
    t_projection = tuple(sorted({(int(e['state_index']), int(e['projection_offset_steps']), int(e['observation_index'])) for e in t_events}))
    t_bins = [False] * 8
    for t in t_times:
        t_bins[bin_of(t)] = True
    for e in t_events:
        require(e['projection_offset_steps'] in (0, 3, 6, 9), 'TTC projection contract changed')
        require(e['observation_index'] == e['state_index'] + e['projection_offset_steps'], 'TTC observation identity mismatch')
        require(e['state_time_seconds'] == e['state_index'] * .1 and
                e['observation_time_seconds'] == e['observation_index'] * .1, 'TTC time semantics changed')
    ts = c['ttc_summary']
    require(ts['valid'] is True and ts['segment_any'] == t_bins and ts['any'] == bool(t_times) == ts['event'], 'TTC aggregation mismatch')
    require(ts['first_state_time_seconds'] == (min(t_times) * .1 if t_times else None), 'TTC first time mismatch')

    m = record['metrics']
    terminal = (1. - float(bool(d_times)), min(c_bins), 1. - float(bool(t_times)))
    require(terminal == (m['drivable_area_compliance'], m['no_at_fault_collisions'], m['time_to_collision_within_bound']), 'Event-to-metric mismatch')
    return {'terminal': terminal, 'time': (d_times, c_signature, t_times),
            'bins': (tuple(d_bins), tuple(c_bins), tuple(t_bins)), 'ttc_projection': t_projection}


def advantages(rewards):
    x = torch.tensor(np.asarray(rewards), dtype=torch.float32).reshape(-1, 4)
    return (x - x.mean(dim=1, keepdim=True)).numpy()


def decompose(metrics):
    # Names here describe algebraic products, not independent causal factors.
    nc, dac, ep, ttc, comfort = (np.asarray(metrics[k], np.float64).reshape(-1, 4) for k in
        ('no_at_fault_collisions', 'drivable_area_compliance', 'ego_progress', 'time_to_collision_within_bound', 'comfort'))
    gate = nc * dac
    terms = np.stack((5 * gate * ep / 12, 5 * gate * ttc / 12, 2 * gate * comfort / 12), axis=-1)
    A = advantages(metrics['score']).astype(np.float64)
    parts = terms - terms.mean(axis=1, keepdims=True)
    residual = A - parts.sum(axis=-1)
    all_parts = np.concatenate((parts, residual[..., None]), axis=-1)
    V = float((A * A).sum())
    sums = np.einsum('bg,bgk->k', A, all_parts)
    return A, terms, {'advantage_squared_sum': V, 'advantage_squared_mean': V / A.size,
        'signed_covariance_numerators': dict(zip(('gated_EP', 'gated_TTC', 'gated_Comfort', 'float32_rounding'), sums.tolist())),
        'signed_covariance_fractions': dict(zip(('gated_EP', 'gated_TTC', 'gated_Comfort', 'float32_rounding'), (sums / V).tolist())) if V else None,
        'max_absolute_rounding_residual': float(np.max(np.abs(residual))),
        'exact_additive_residual': float(np.max(np.abs(A - all_parts.sum(axis=-1)))),
        'interpretation': 'Descriptive centered-value algebra; not parameter-gradient share or causal attribution'}


def tie_counts(records, profiles):
    totals = {}
    conditions = ('C1_float32_tie', 'C1_float64_tie', 'safety_vector_tie', 'D_terminal_tie', 'C_terminal_tie', 'T_terminal_tie')
    for condition in conditions:
        totals[condition] = {'eligible_pairs': 0, 'eligible_scenes': set(), 'nonempty_event_pairs': 0,
            'physical_time_or_severity_diff_pairs': 0, 'physical_time_or_severity_diff_scenes': set(),
            'segment_diff_pairs': 0, 'ttc_projection_diff_pairs': 0, 'per_component_nonempty_pairs': [0] * 3,
            'per_component_time_diff_pairs': [0] * 3}
    for s in range(32):
        for a, b in itertools.combinations(range(4), 2):
            i, j = 4 * s + a, 4 * s + b
            r1, r2, p1, p2 = records[i], records[j], profiles[i], profiles[j]
            score1, score2 = r1['metrics']['score'], r2['metrics']['score']
            masks = (np.float32(score1) == np.float32(score2), score1 == score2,
                     p1['terminal'] == p2['terminal'], *(p1['terminal'][k] == p2['terminal'][k] for k in range(3)))
            for index, condition in enumerate(conditions):
                if not masks[index]:
                    continue
                relevant = (index - 3,) if index >= 3 else (0, 1, 2)
                out = totals[condition]
                out['eligible_pairs'] += 1
                out['eligible_scenes'].add(s)
                nonempty = [bool(p1['time'][k] or p2['time'][k]) for k in range(3)]
                different = [p1['time'][k] != p2['time'][k] for k in range(3)]
                out['nonempty_event_pairs'] += int(any(nonempty[k] for k in relevant))
                if any(different[k] for k in relevant):
                    out['physical_time_or_severity_diff_pairs'] += 1
                    out['physical_time_or_severity_diff_scenes'].add(s)
                out['segment_diff_pairs'] += int(any(p1['bins'][k] != p2['bins'][k] for k in relevant))
                if 2 in relevant:
                    out['ttc_projection_diff_pairs'] += int(p1['ttc_projection'] != p2['ttc_projection'])
                for k in relevant:
                    out['per_component_nonempty_pairs'][k] += int(nonempty[k])
                    out['per_component_time_diff_pairs'][k] += int(different[k])
    for out in totals.values():
        for key in ('eligible_scenes', 'physical_time_or_severity_diff_scenes'):
            out[key] = len(out[key])
        denominator = out['nonempty_event_pairs']
        out['conditional_time_difference_fraction_given_nonempty'] = out['physical_time_or_severity_diff_pairs'] / denominator if denominator else None
    return totals


def synthetic_checks():
    def fake(d=(), c=(), t=()):
        flags = np.zeros(41, dtype=bool)
        flags[list(d)] = True
        db, cb, tb = [False] * 8, [1.] * 8, [False] * 8
        for x in d: db[bin_of(x)] = True
        for x, v in c: cb[bin_of(x)] = min(cb[bin_of(x)], v)
        for x, _ in t: tb[bin_of(x)] = True
        cs = {'valid': True, 'segment_nc_min': cb, 'nc_min': min(cb), 'event': bool(c), 'first_at_fault_time_seconds': min(x for x, _ in c) * .1 if c else None}
        ts = {'valid': True, 'segment_any': tb, 'any': bool(t), 'event': bool(t), 'first_state_time_seconds': min(x for x, _ in t) * .1 if t else None}
        return {'metrics': {'drivable_area_compliance': 1.-bool(d), 'no_at_fault_collisions': min(cb), 'time_to_collision_within_bound': 1.-bool(t)},
            'consequences': {'road': {'valid': True, 'native_flag': flags, 'segment_any': db, 'any': bool(d), 'event': bool(d), 'first_time_seconds': min(d) * .1 if d else None},
            'collision_summary': cs, 'ttc_summary': ts,
            'collision_events': [{'at_fault': True, 'state_index': x, 'state_time_seconds': x*.1, 'nc_contribution': v} for x,v in c],
            'ttc_events': [{'violation': True, 'state_index': x, 'state_time_seconds': x*.1, 'projection_offset_steps': y, 'observation_index': x+y, 'observation_time_seconds': (x+y)*.1} for x,y in t]}}
    p1, p2 = process_profile(fake((2,))), process_profile(fake((12,)))
    require(p1['terminal'] == p2['terminal'] and p1['time'] != p2['time'], 'Failed same-terminal/different-time control')
    empty = process_profile(fake())
    require(not any(empty['time']) and empty['terminal'] == (1.,1.,1.), 'Empty event control failed')
    severe = process_profile(fake(c=((3,.5),(17,0.))))
    require(severe['terminal'][1] == 0. and severe['time'][1] == ((3,.5),(17,0.)), 'Later severity control failed')
    a,b = process_profile(fake(t=((5,0),))), process_profile(fake(t=((5,9),)))
    require(a['time'] == b['time'] and a['ttc_projection'] != b['ttc_projection'], 'TTC offset confused with physical time')
    require([bin_of(x) for x in (0,5,6,10,40)] == [0,0,1,1,7], 'Right-closed bins failed')
    require(not np.any(advantages([[.9]*4])), 'Constant reward caused advantages')
    require(float(.9) != float(np.nextafter(.9, 1.)) and np.float32(.9) == np.float32(np.nextafter(.9,1.)), 'Float32 tie control failed')
    # Signed covariance attribution can exceed one or become negative.
    avec = np.array([-1.,1.]); pvec = 2 * avec; nvec = -avec
    require(float(avec @ pvec / (avec @ avec)) == 2. and float(avec @ nvec / (avec @ avec)) == -1., 'Cancellation control failed')
    # Toy joint generation: each flow step writes both physical trajectory points.
    # q=(sum(theta)/10, 2*sum(theta)/10); a t=4s event depends on q1+2*q2.
    def event(theta):
        q = np.array([theta.sum()/10., 2*theta.sum()/10.])
        return float(q[0] + 2*q[1])
    eps = 1e-6; theta = np.zeros(10)
    fd = [(event(theta+eps*np.eye(10)[k])-event(theta-eps*np.eye(10)[k]))/(2*eps) for k in range(10)]
    require(np.allclose(fd, [.5]*10, atol=1e-12, rtol=0), 'Flow/time counterexample arithmetic failed')
    return {'status': 'PASS_SYNTHETIC_CONTROLS_NOT_SCIENTIFIC_APPROVAL', 'control_count': 9,
        'physical_time_to_flow_step_counterexample': {'flow_steps': 10, 'physical_event_time_seconds': 4,
            'joint_command_points': 'q=(sum(theta)/10,2*sum(theta)/10)', 'event': 'q[0]+2*q[1]',
            'analytic_event_derivative_all_flow_parameters': [.5]*10, 'central_difference': fd,
            'physical_command_point_derivative': [1.,2.], 'nonzero_flow_parameters': 10,
            'nonzero_flow_parameters_discarded_by_one_step_mapping': 9,
            'boundary': 'Mathematical counterexample to a universal one-to-one mapping; not an actual NAVSIM sensitivity claim'}}


def main():
    budget = ReadBudget()
    plan_sha = digest(budget.read(HERE/'PLAN.md'))
    code_sha = digest(budget.read(__file__))
    controls = synthetic_checks()
    run_identity = budget.json(RUN/'identity.json', '15535618f98c13c9b5a5929607a67012fbc5bc88f394ab353f70ebec37655930')
    request_doc = budget.json(LEDGER/'REQUESTS.json', 'e6a0400b7d22ed38f8a26cd11f1ced8f68d8dec97c06fca85a78439d380ba9b6')
    require(request_doc['policy_version'] == '3584' and identity(request_doc['requests']) == KEY, 'Wrong request population')
    requests = request_doc['requests']
    complete = budget.json(LEDGER/'attempt_00/COMPLETE.json', 'e1d40b12cd35e022be2ecd309b7f7195c5b53133bf66c724f13a362e850e863f')
    admission = budget.json(LEDGER/'attempt_00/ADMITTED.json', complete['admission_sha256'])
    require(admission['calls'] == 128 and admission['indices'] == list(range(128)) and admission['request_hash'] == KEY and admission['policy_version'] == '3584', 'Incomplete admission')
    result_path = LEDGER/'attempt_00/RESULTS.pt'
    require(complete['result'] == {'path': str(result_path), 'sha256': '8019dd29b17efa79c1fd1a7211bdf09a0b5be93931db75cd4db95ad9a3fb122f'}, 'Wrong existing payload')
    raw = budget.read(result_path, complete['result']['sha256'])
    require(len(raw) == 3441888, 'Payload size changed')
    records = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
    require(len(records) == len(requests) == 128, 'Lost candidates')
    require(len({r['scene_id'] for r in requests}) == 32 and len({r['source_log'] for r in requests}) == 32, 'Scene/log population changed')
    ranks = [budget.json(RUN/'rollouts/boundary_003584'/f'rank_{i}_COMPLETE.json', sha) for i,sha in enumerate(RANK_HASHES)]
    profile = []
    for i,(request,r) in enumerate(zip(requests,records)):
        rank, slot, cand = i//32, (i%32)//4, i%4
        saved = ranks[rank]['identity']
        require(r['identity'] == request and r['status'] == 'VALID_SCORED' and r['valid'] is True, 'Invalid/foreign feedback')
        require(request['candidate_id'] == f"{saved['rollout_id']}:{slot}:{cand}", 'Rank/candidate identity mismatch')
        require(saved['old_policy']['state_hash'] == STATE_HASH and saved['old_policy']['version'] == saved['old_policy']['optimizer_step'] == 3584, 'Frozen policy changed')
        require(saved['numeric_execution_hash'] == NUMERIC_HASH == request['inference_runtime_hash'], 'Numeric runtime changed')
        require(request['policy_version'] == '3584' and request['generator_identity']['state_sha256'] == STATE_HASH and request['score_protocol_hash'] == PROTOCOL_HASH, 'Request policy/protocol changed')
        require(request['scene_id'] == requests[4*(i//4)]['scene_id'], 'Scene candidate order changed')
        require(saved['feedback']['statuses'][slot*4+cand] == 'valid_scored', 'Saved feedback invalid')
        require(float(np.float32(r['metrics']['score'])) == saved['feedback']['rewards'][slot][cand], 'Training reward cast mismatch')
        require(saved['feedback']['contexts'][slot] == request['metric_cache_sha256'], 'Saved context changed')
        require(r['score_units'] == '0..1' and r['context_unchanged'] and set(r['metrics']) == set(METRICS), 'Metric schema/units changed')
        require(all(r['metrics_valid'][k] and np.isfinite(r['metrics'][k]) and 0 <= r['metrics'][k] <= 1 for k in METRICS), 'Invalid metric values')
        require(array_hash(r['requested_trajectory']) == request['trajectory_sha256'], 'Trajectory content changed')
        ref = r['reference_contract']; ref_input = ref['input']; ref_output = ref['execution']
        require(ref['schema'] == 'st_world_reference_input_and_execution_v2' and identity(ref_input) == ref['input_sha256'] and identity(ref_output) == ref['execution_sha256'], 'Reference identity corrupt')
        require(ref_input == records[4*(i//4)]['reference_contract']['input'], 'Reference input differs within scene')
        require(ref_input['full_cache_sha256'] == request['metric_cache_sha256'] and ref_input['score_protocol_hash'] == PROTOCOL_HASH and ref_input['context_hash'] == r['context_hash'], 'Context join failed')
        require(array_hash(r['reference_interpolated_states']) == ref_input['reference_interpolated_states_sha256'] and array_hash(r['executed_states']) == ref_output['candidate_executed_states_sha256'], 'Saved execution checksum mismatch')
        profile.append(process_profile(r))

    metrics = {k: np.array([r['metrics'][k] for r in records]).reshape(32,4) for k in METRICS}
    A,terms,algebra = decompose(metrics)
    reconstruction_error = float(np.max(np.abs(terms.sum(axis=-1) - metrics['score'])))
    require(reconstruction_error <= 1e-12, 'Canonical score decomposition failed')
    changes = {k: np.ptp(metrics[k], axis=1) != 0 for k in METRICS if k != 'score'}
    active = np.any(A != 0, axis=1)
    safe_changes = changes['no_at_fault_collisions'] | changes['drivable_area_compliance'] | changes['time_to_collision_within_bound']
    ep_only = active & changes['ego_progress'] & ~safe_changes & ~changes['comfort'] & np.all(metrics['no_at_fault_collisions'] * metrics['drivable_area_compliance'] > 0, axis=1)
    component_adv = {'C1': int(active.sum())}
    for name,key,sign in (('C2','drivable_area_compliance',1),('C3','no_at_fault_collisions',1),('C4','time_to_collision_within_bound',1)):
        component_adv[name] = int(np.any(advantages(metrics[key]-1.) != 0,axis=1).sum())
    rows=[]
    for s in range(32):
        rows.append({'scene_id': requests[4*s]['scene_id'], 'source_log': requests[4*s]['source_log'],
            'original_rank': s//8, 'rank_scene_slot': s%8, 'C1_nonzero_advantage': bool(active[s]),
            'EP_only': bool(ep_only[s]), 'variable_metrics': [k for k,v in changes.items() if v[s]],
            'C1_rewards_float64': metrics['score'][s].tolist(), 'C1_advantages_float32': A[s].tolist(),
            'event_candidates_D_C_T': [sum(bool(profile[4*s+c]['time'][k]) for c in range(4)) for k in range(3)]})
    out = {'schema': 'st_consequence_signal_support_v1', 'status': 'READ_ONLY_ANALYSIS_COMPLETE_NEEDS_INDEPENDENT_REVIEW',
        'plan_sha256': plan_sha, 'analyzer_sha256': code_sha,
        'source_base_sha': '336e41cda741b58d1a82f798ccd0902d17e88d94',
        'population': {'cell':'C1','boundary':3584,'scenes':32,'logs':32,'candidates':128,'candidate_pairs':192,'valid_candidates':128,'invalid_candidates':0,'rank_count':4,'group_size':4,'selection':'all original rank0..3 slots0..7 candidates0..3; no replacement'},
        'identities': {'policy_state_sha256':STATE_HASH,'numeric_execution_hash':NUMERIC_HASH,'score_protocol_hash':PROTOCOL_HASH,'training_manifest_identity_from_run_receipt':run_identity['data'],'official_result_sha256':complete['result']['sha256'],'ledger_key':KEY,'base_binding':ranks[0]['identity']['old_policy']['base_binding_hash']},
        'canonical_score_reconstruction_max_absolute_error': reconstruction_error,
        'signal': {'nonzero_advantage_scenes':component_adv,'C1_zero_advantage_scenes':int((~active).sum()),'C1_nonzero_advantage_candidates':int(np.count_nonzero(A)),
            'EP_only_nonzero_advantage_scenes':int(ep_only.sum()),'safety_metric_varies_scenes':int(safe_changes.sum()),
            'variable_metric_scene_counts':{k:int(v.sum()) for k,v in changes.items()},
            'event_candidate_counts_D_C_T':[sum(bool(p['time'][k]) for p in profile) for k in range(3)],
            'all_safe_scene_count':int(np.sum(np.all((metrics['no_at_fault_collisions']==1)&(metrics['drivable_area_compliance']==1)&(metrics['time_to_collision_within_bound']==1),axis=1))),
            'zero_score_candidates':int(np.count_nonzero(metrics['score']==0)),
            'all_zero_multiplier_scene_count':int(np.sum(np.all(metrics['no_at_fault_collisions']*metrics['drivable_area_compliance']==0,axis=1))),
            'component_metric_values':{k:dict(zip(*[[str(float(v)) for v in np.unique(metrics[k])], [int(np.sum(metrics[k]==v)) for v in np.unique(metrics[k])]])) for k in ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','comfort')}},
        'advantage_value_decomposition':algebra, 'terminal_ties':tie_counts(records,profile), 'per_scene':rows,
        'synthetic_controls':controls,
        'limits_and_execution':{'CPU_seconds_limit':120,'task_input_logical_read_limit_bytes':32<<20,'output_limit_bytes':4<<20,
            'task_input_logical_read_bytes':budget.total,'read_ledger':budget.entries,
            'wall_seconds':time.monotonic()-START,'cpu_seconds':resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'cuda_initialized':torch.cuda.is_initialized(),'torch_threads':torch.get_num_threads(),
            'new_score_calls':0,'new_rollouts':0,'model_loads':0,'optimizer_updates':0,'real_gradient_computations':0,'full_manifest_reads':0,'metric_cache_reads':0,'DEV_or_Navtest_reads':0},
        'gradient_balance_result':'PENDING_LEAD_HANDOFF_NOT_INFERRED',
        'formal_training_approval':False}
    payload=json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False).encode()+b'\n'
    require(len(payload)<=4<<20,'Output budget exceeded')
    target=HERE/'RESULTS.json'
    require(not target.exists(),'Immutable output already exists')
    target.write_bytes(payload)
    print(json.dumps({'status':out['status'],'population':out['population'],'signal':out['signal'],'tie_counts':out['terminal_ties'],'execution':{k:v for k,v in out['limits_and_execution'].items() if k!='read_ledger'},'output_bytes':len(payload)},ensure_ascii=False))


if __name__=='__main__':
    main()
