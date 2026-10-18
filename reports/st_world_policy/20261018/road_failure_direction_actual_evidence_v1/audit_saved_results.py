"""Independent stdlib-only audit of saved road diagnostic JSON; no replay."""
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import struct
import time

OUT = Path(__file__).resolve().parent
ROOT = Path('/mnt/project/st-world-policy-20261016/audits/consequence/road_failure_direction_v1')
RUN = ROOT / 'diagnostic'
SOURCE = Path('/mnt/project/VLA-Drive-st-road-failure-direction-20261018')
LIMIT = 24 * 2**20  # leave8MiB for earlier inspection and final report reads
reads, checks = [], []
started = time.monotonic()
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
resource.setrlimit(resource.RLIMIT_CPU, (110, 110))
signal.alarm(550)
peak_rss = 0


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def require(value, label):
    if not value:
        raise AssertionError(label)
    checks.append(label)


def read(path, expected=None, parse=True):
    global peak_rss
    path = Path(path)
    require(path.suffix in ('.json', '.py'), 'only saved JSON/source ' + path.name)
    require(sum(x['bytes'] for x in reads) + path.stat().st_size <= LIMIT, 'read cap')
    memory = {line.split(':')[0]: int(line.split()[1]) * 1024 for line in Path('/proc/self/status').read_text().splitlines() if line.startswith(('VmRSS:', 'VmHWM:'))}
    peak_rss = max(peak_rss, memory['VmHWM'], memory['VmRSS'])
    require(peak_rss < 2 * 2**30, 'actual RSS cap')
    raw = path.read_bytes(); digest = sha(raw)
    reads.append({'path': str(path), 'bytes': len(raw), 'sha256': digest})
    if expected:
        require(digest == expected, 'hash ' + str(path))
    return json.loads(raw) if parse else raw


def array_hash(values):
    shape = []
    def flat(v, depth=0):
        if isinstance(v, list):
            if len(shape) <= depth:
                shape.append(len(v))
            require(len(v) == shape[depth], 'rectangular saved array')
            return [y for x in v for y in flat(x, depth + 1)]
        require(type(v) in (int, float) and math.isfinite(v), 'finite saved numeric')
        return [v]
    values = flat(values)
    head = json.dumps({'dtype': '<f8', 'shape': shape}, sort_keys=True).encode()
    return sha(head + struct.pack('<' + 'd' * len(values), *values))


def raw_hash(values):
    flat = [v for row in values for v in row]
    return sha(struct.pack('<' + 'f' * len(flat), *flat))


def norm(values):
    return math.sqrt(sum(v * v for row in values for v in row))


def main():
    reg = read(ROOT / 'REGISTRATION.json', 'c9c76ffd513fcf3cf59e1753358bd552b0635ae6394399e509d15c09dd9cda0a')
    admission = read(ROOT / 'ADMISSION_initial_v1.json')
    binding = read(RUN / 'EXECUTION_BINDING.json')
    require(binding['registration_sha256'] == admission['registration_sha256'] == 'c9c76ffd513fcf3cf59e1753358bd552b0635ae6394399e509d15c09dd9cda0a', 'registration/admission/execution binding')
    require(binding['source_hashes'] == admission['source_hashes'] == reg['source_hashes'], 'source identity shared')
    for path, expected in reg['source_hashes'].items():
        read(SOURCE / path, expected, parse=False)
    reviews = {}
    for role, ref in admission['reviews'].items():
        reviews[role] = read(ref['path'], ref['sha256'])
    selection = read(reg['evidence']['path'], reg['evidence']['sha256'])
    frame = read(selection['frame']['path'], selection['frame']['sha256'])
    binder = read(ROOT / 'BOUND_SELECTION_BINDING_REPORT.json')
    require(binder['selection']['sha256'] == reg['evidence']['sha256'], 'binder binds selection')
    require(binder['population']['replacement_scenes'] == 0, 'no replacement scenes')
    require(sha(encoded(selection['selected'])) == selection['selection_sha256_before_proxy'] == reg['selection_hash'], 'final selected content hash')
    require([r['scene_id'] for r in selection['selected']] == [r['scene_id'] for r in frame['selected']], 'exact original frame order')
    require(len(selection['selected']) == len(set(r['scene_id'] for r in selection['selected'])) == 32, '32 unique bound scenes')
    summary = read(RUN / 'RESULT_failure32_initial_v1.json')
    worker = read(RUN / 'WORKER_RESULT_failure32_initial_v1.json')
    require(worker['status'] == 'WORKER_FINISHED' and worker['error'] is None and worker['cuda_initialized'] is False, 'worker completed without GPU')
    require(summary['supervisor']['owned_worker_exit_confirmed'] and summary['supervisor']['worker_exit_code'] == 0, 'owned worker exited')
    intents = [read(p) for p in sorted((RUN / 'calls').glob('*.json'))]
    require(len(intents) == summary['charged_official_calls'] == 6, 'six charged calls')
    intent_ids = collections.Counter(sha(encoded(x['identity'])) for x in intents)
    require(all(x['charged_before_call'] is True for x in intents), 'durable admission before each call')
    states, requested_safe, complete_rows, status_rows = collections.Counter(), collections.Counter(), [], []
    parity_max = collections.Counter()
    scored_ids = []
    for item, original_frame in zip(selection['selected'], frame['selected']):
        token = item['scene_id']; folder = RUN / 'scenes' / token
        failed, sibling = item['failed'], item['safe_sibling']
        require(item['role'] == 'train' and item['boundary'] == original_frame['policy_version'] and item['rank'] == original_frame['rank'] and item['slot'] == original_frame['local_slot'], 'TRAIN frame slot ' + token)
        for witness in (failed, sibling):
            require(witness['valid'] is True and witness['score_units'] == '0..1' and all(witness['metrics_valid'].values()), 'valid original witness ' + token)
            require(witness['native_t0'] is False, 'original t0 false ' + token)
        require(failed['metrics']['drivable_area_compliance'] == 0 and sibling['metrics']['drivable_area_compliance'] == 1, 'DAC0 and DAC1 sibling ' + token)
        require(failed['context_hash'] == sibling['context_hash'] and failed['reference_input_sha256'] == sibling['reference_input_sha256'], 'same context/reference witness ' + token)
        for k in ('scene_id', 'source_log', 'metric_cache_path', 'metric_cache_sha256', 'score_protocol_hash', 'inference_runtime_hash', 'policy_version', 'generator_identity'):
            require(failed['request'][k] == sibling['request'][k], 'same witness identity ' + token + ' ' + k)
        for k in ('comfort', 'no_at_fault_collisions', 'time_to_collision_within_bound', 'drivable_area_compliance'):
            requested_safe[k] += failed['metrics'][k] == 1
        state = read(folder / 'SCENE_RESULT.json'); begun = read(folder / 'STARTED.json'); parity = read(folder / 'NUMERICAL_PARITY.json')
        require(state == next(x for x in summary['scenes'] if x['scene_id'] == token), 'scene result summary matches ' + token)
        require(state['original_metrics'] == begun['original_metrics'] == failed['metrics'], 'original labels match binding ' + token)
        require(begun['request'] == failed['request'] and raw_hash(begun['original_raw']) == failed['raw_sha256'], 'stored start raw/request binding ' + token)
        for k in ('command_xy_error_m', 'command_yaw_error_rad', 'corner_max_error_m'):
            require(parity[k] <= 1e-9, 'unchanged numerical parity ' + token + ' ' + k)
            parity_max[k] = max(parity_max[k], parity[k])
        states[state['status']] += 1
        status_rows.append({'scene_id': token, 'boundary': item['boundary'], 'source_log': item['source_log'], 'status': state['status'], 'error': state.get('error'), 'original_DAC': failed['metrics']['drivable_area_compliance'], 'sibling_DAC': sibling['metrics']['drivable_area_compliance'], 'original_t0': failed['native_t0']})
        if state['status'] != 'COMPLETE':
            require(state['status'] == 'UNDEFINED_DIRECTION' and state['error'] == 'MOVING_CORNER_BOUNDARY_OR_NEAREST_TIE', 'exact undefined reason ' + token)
            require(not (folder / 'DIRECTIONS.json').exists() and set(p.name for p in folder.glob('*_RESULT.json')) == {'SCENE_RESULT.json'}, 'no variant scores for undefined ' + token)
            continue
        directions = read(folder / 'DIRECTIONS.json')
        expected_seed = int.from_bytes(hashlib.sha256((token + '\0' + reg['selection_hash'] + '\0road_direction_v1').encode()).digest()[:8], 'big')
        require(directions['random_seed'] == expected_seed, 'unchanged random seed ' + token)
        require(raw_hash(directions['variants']['original']['raw']) == failed['raw_sha256'], 'original direction raw binds ' + token)
        require(abs(norm(directions['gradient']) - directions['gradient_norm']) < 1e-12, 'stored gradient norm ' + token)
        require(all(x['passed'] and x['stable_discrete_regime'] for x in directions['finite_difference']), 'both stored finite differences pass ' + token)
        case = {'scene_id': token, 'source_log': item['source_log'], 'boundary': item['boundary'], 'failed_candidate': failed['candidate'], 'safe_sibling_candidate': sibling['candidate'], 'safe_sibling_metrics': sibling['metrics'], 'gradient_norm': directions['gradient_norm'], 'finite_difference': directions['finite_difference'], 'variants': {}}
        originals = None
        for name in ('original', 'negative', 'positive', 'random'):
            value = read(folder / (name + '_RESULT.json')); record = value['record']; direction = directions['variants'][name]
            require(record['valid'] is True and record['status'] == 'VALID_SCORED' and record['identity'] == value['request'], 'valid scored identity ' + token + ' ' + name)
            require(all(record['metrics_valid'].values()), 'valid metrics ' + token + ' ' + name)
            require(value['original_reused'] == (name == 'original'), 'original reused only ' + token + ' ' + name)
            require(record['context_hash'] == failed['context_hash'] and record['context_unchanged'], 'unchanged context ' + token + ' ' + name)
            ref = record['reference_contract']; inp, execution = ref['input'], ref['execution']
            require(sha(encoded(inp)) == ref['input_sha256'] == failed['reference_input_sha256'], 'reference input checksum ' + token + ' ' + name)
            require(sha(encoded(execution)) == ref['execution_sha256'], 'reference execution checksum ' + token + ' ' + name)
            require(array_hash(record['requested_trajectory']) == value['request']['trajectory_sha256'] == execution['candidate_trajectory_sha256'], 'physical request bytes ' + token + ' ' + name)
            require(array_hash(record['executed_states']) == execution['candidate_executed_states_sha256'], 'executed bytes ' + token + ' ' + name)
            require(array_hash(record['reference_interpolated_states']) == inp['reference_interpolated_states_sha256'], 'reference interpolation bytes ' + token + ' ' + name)
            require(array_hash(record['reference_executed_states']) == execution['reference_executed_states_sha256'], 'reference execution bytes ' + token + ' ' + name)
            road = record['consequences']['road']; times = [i for i, flag in enumerate(road['native_flag']) if flag]
            require(len(road['native_flag']) == 41 and all(type(v) is bool for v in road['native_flag']) and not road['native_flag'][0], '41 native bool flags and t0 ' + token + ' ' + name)
            require(record['metrics']['drivable_area_compliance'] == 1 - int(bool(times)), 'native any to DAC ' + token + ' ' + name)
            require(road['first_time_seconds'] == (times[0] * .1 if times else None), 'native first time ' + token + ' ' + name)
            m = record['metrics']; progress = record['progress']
            require(abs(m['score'] - m['no_at_fault_collisions'] * m['drivable_area_compliance'] * (5*m['ego_progress'] + 5*m['time_to_collision_within_bound'] + 2*m['comfort']) / 12) < 1e-12, 'official score algebra ' + token + ' ' + name)
            require(progress['ep_branch'] == 'normalize' and abs(progress['effective_candidate_meters'] / progress['ep_denominator_meters'] - m['ego_progress']) < 1e-12, 'EP gate algebra ' + token + ' ' + name)
            require(state['variants'][name]['metrics'] == m and state['variants'][name]['geometry'] == value['geometry'], 'scene summary agrees with individual score ' + token + ' ' + name)
            if name == 'original':
                originals = record
                require(value['request'] == failed['request'] and m == failed['metrics'], 'reused original exact binding ' + token)
            else:
                scored_ids.append(sha(encoded(value['request'])))
                require(ref['input'] == originals['reference_contract']['input'] and record['reference_executed_states'] == originals['reference_executed_states'], 'same actual reference ' + token + ' ' + name)
                delta = [[x-y for x,y in zip(r,s)] for r,s in zip(direction['raw'], directions['variants']['original']['raw'])]
                require(abs(norm(delta) - .005) < 1e-14, 'same full32 raw L2 ' + token + ' ' + name)
                if name in ('negative', 'positive'):
                    sign = -1 if name == 'negative' else 1
                    require(max(abs(d-sign*.005*g/directions['gradient_norm']) for dr,gr in zip(delta,directions['gradient']) for d,g in zip(dr,gr)) < 1e-15, 'stored signed-gradient direction ' + token + ' ' + name)
            case['variants'][name] = {'metrics': m, 'executed_geometry': value['geometry'], 'command_geometry': {k:v for k,v in direction.items() if k not in ('raw','physical','command_poses')}, 'native_D_time_indices': times, 'native_D_times_seconds': [i*.1 for i in times], 'progress': progress, 'tracking': value['tracking_and_kinematics']}
        origin = case['variants']['original']
        for name, row in case['variants'].items():
            row['raw_progress_delta_vs_original_m'] = row['progress']['raw_meters'] - origin['progress']['raw_meters']
        complete_rows.append(case)
    require(states == {'COMPLETE': 2, 'UNDEFINED_DIRECTION': 30}, 'two complete thirty undefined')
    require(collections.Counter(scored_ids) == intent_ids and len(scored_ids) == 6, 'six unique saved scores match six intents')
    require(dict(requested_safe) == summary['requested_original_compliant_denominators'], 'full32 safety denominators')
    repairs = {v: sum(r['variants'][v]['metrics']['drivable_area_compliance'] == 1 for r in complete_rows) for v in ('original','negative','positive','random')}
    require(repairs == {'original':0,'negative':2,'positive':0,'random':1}, 'observed paired repairs')
    result = {'schema':'independent_road_failure_actual_evidence_v1','status':'SAVED_EVIDENCE_CONSISTENT_LOCAL_ONLY_NOT_NEXT_STAGE_APPROVAL',
        'source_commit':reg['source_commit'],'selection_sha256':reg['selection_hash'],'requested':32,'eligible_original_DAC_failures':32,
        'original_t0_false':32,'same_context_DAC1_siblings':32,'common_valid':2,'undefined':30,'defined_fraction':2/32,
        'undefined_reason':'MOVING_CORNER_BOUNDARY_OR_NEAREST_TIE','boundary_vs_tie_breakdown_available':False,
        'new_official_calls':6,'repairs':repairs,'negative_known_repairs_over_requested_lower_bound':2/32,
        'requested_original_compliant_denominators':dict(requested_safe),'actual_parity_max_errors':dict(parity_max),
        'complete_cases':complete_rows,'fixed_population_statuses':status_rows,'execution_resources':{'supervisor':{k:v for k,v in summary['supervisor'].items() if k not in ('owned_worker_pid','resource_tracker_pid')},'worker':{k:v for k,v in worker['resources'].items() if k!='asset_reads'}},
        'next_stage':'NO_TRAINING_OR_NEW_PROXY_METHOD_APPROVAL','scope':'Two historical TRAIN commands under their fixed contexts; not a trained-policy improvement or population effect.'}
    (OUT/'AUDIT_RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (OUT/'READ_RECEIPT.json').write_text(json.dumps({'schema':'read_only_road_evidence_audit_resources_v1','status':'COMPLETE','logical_content_read_bytes':sum(x['bytes'] for x in reads),
        'prior_inspection_and_final_sealing_reserved_bytes':8*2**20,'total_content_read_upper_bound_bytes':sum(x['bytes'] for x in reads)+8*2**20,
        'wall_seconds':time.monotonic()-started,'cpu_seconds':resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,
        'actual_proc_VmHWM_bytes':peak_rss,'CPU_affinity':sorted(os.sched_getaffinity(0)),'read_pins':reads,'assertion_count':len(checks),
        'proxy_calls':0,'new_scores':0,'models':0,'cache_payloads':0,'GPU':0,'training_updates':0},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'repairs':repairs,'requested':32,'valid':2,'undefined':30,'read_bytes':sum(x['bytes'] for x in reads),'actual_peak_rss_bytes':peak_rss,'assertions':len(checks)}))


if __name__ == '__main__':
    main()
