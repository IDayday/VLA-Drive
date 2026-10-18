"""Read only the five fixed completed runs' bounded scalar logs; no payload IO."""
from pathlib import Path
import hashlib
import json
import math
import statistics
import datetime

ROOT = Path('/mnt/project/st-world-policy-20261016/formal/consequence/runs')
RUNTIME = Path('/mnt/project/VLA-Drive-st-flow-context-runtime-v3-20261017')
OUT = Path(__file__).resolve().parent
ARMS = ['C0', 'C1', 'C2', 'C3', 'C4']
WINDOWS = {'all_1_4096': (1, 4096), '1_1024': (1, 1024), '1025_2048': (1025, 2048),
           '2049_3072': (2049, 3072), '3073_4096': (3073, 4096)}
BUDGET = 128 * 2**20
# The first-line field reconnaissance already occurred. Reserve a conservative
# 1 MiB for it; this scan reads and hashes each scalar file only once.
used = 2**20
pins = []


def read_small(path):
    size = path.stat().st_size
    if size > 1024*1024:
        raise ValueError('Only small control documents are authorized')
    raw = path.read_bytes()
    pins.append({'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    return json.loads(raw)


def stats(values):
    values = list(values)
    assert values and all(math.isfinite(v) for v in values)
    ordered = sorted(values)
    def quantile(q):
        x = q*(len(ordered)-1); lo = int(x); hi = min(lo+1, len(ordered)-1)
        return ordered[lo] + (x-lo)*(ordered[hi]-ordered[lo])
    return {'count': len(values), 'mean': statistics.fmean(values), 'min': min(values),
            'p50': quantile(.5), 'p95': quantile(.95), 'max': max(values)}


def aggregate(rows, arm):
    result = {'updates': len(rows), 'first_step': rows[0]['step'], 'last_step': rows[-1]['step'],
              'counts': {k: sum(r['counts'][k] for r in rows) for k in rows[0]['counts']},
              'loss': {k: stats(r['losses'][k] for r in rows) for k in ['bc','policy','total']},
              'absolute_policy_loss': stats(abs(r['losses']['policy']) for r in rows),
              'gradient_norm_combined_bc_and_policy': stats(r['gradient_norm'] for r in rows),
              'lr': stats(r['lr'] for r in rows),
              'sampled_changed_master_elements': stats(r['sampled_changed_elements'] for r in rows),
              'maximum_sampled_master_abs_change': stats(r['maximum_sampled_change'] for r in rows),
              'fp32_master_verified_updates': sum(r['fp32_master_update_verified'] is True for r in rows),
              'slowest_rank_update_seconds': stats(r['slowest_rank_seconds'] for r in rows),
              'summed_update_seconds_excluding_checkpoint_setup_cleanup': sum(r['slowest_rank_seconds'] for r in rows),
              'official_admitted_calls_cumulative_at_end': rows[-1]['official_admitted_calls'],
              'gradient_norm_above_configured_clip_updates': sum(r['gradient_norm'] > 1 for r in rows)}
    if arm == 'C0':
        assert all(r['old_policy'] is None and all(p['official'] is None for p in r['per_rank']) for r in rows)
        result['feedback'] = {'status': 'NOT_APPLICABLE_C0_NO_FEEDBACK'}
        return result
    ranks = [p for r in rows for p in r['per_rank']]
    scenes = sum(len(p['sample_ids']) for p in ranks)
    valid = sum(p['official']['valid_candidates'] for p in ranks)
    assert valid == 4*scenes == result['counts']['actual_rollouts']
    assert all(p['official']['valid_candidates'] == 4*len(p['sample_ids']) for p in ranks)
    zero = sum(p['official']['zero_advantage_scenes'] for p in ranks)
    metrics = ranks[0]['official']['metrics_mean'].keys()
    metrics_mean = {k: sum(p['official']['metrics_mean'][k]*p['official']['valid_candidates'] for p in ranks)/valid
                    for k in metrics}
    weighted_reward = sum(p['official']['mean_reward']*p['official']['valid_candidates'] for p in ranks)/valid
    maxima = [p['policy_ratio_max_abs_from_one'] for p in ranks]
    assert all(type(v) in (float, int) and math.isfinite(v) for v in maxima)
    active_updates = sum(sum(p['official']['zero_advantage_scenes'] for p in r['per_rank'])
                         < r['counts']['original_GT_scenes'] for r in rows)
    f = {'status':'VALID_SCALAR_SUMMARIES', 'scene_presentations':scenes,
         'valid_candidate_presentations':valid, 'invalid_candidate_presentations_in_completed_updates':0,
         'zero_advantage_scene_presentations':zero, 'zero_advantage_scene_fraction':zero/scenes,
         'nonzero_advantage_scene_presentations':scenes-zero, 'nonzero_advantage_scene_fraction':1-zero/scenes,
         'updates_with_at_least_one_nonzero_advantage_scene':active_updates,
         'updates_with_only_zero_advantage_scenes':len(rows)-active_updates,
         'candidate_weighted_mean_reward':weighted_reward, 'candidate_weighted_metrics_mean':metrics_mean,
         'candidate_DAC_failure_fraction_from_binary_mean':1-metrics_mean['drivable_area_compliance'],
         'candidate_TTC_failure_fraction_from_binary_mean':1-metrics_mean['time_to_collision_within_bound'],
         'candidate_NC_mean_penalty_not_failure_fraction':1-metrics_mean['no_at_fault_collisions'],
         'ratio_max_abs_from_one_per_rank_update':stats(maxima),
         'rank_updates_with_ratio_outside_0p8_1p2':sum(v > .2 for v in maxima),
         'outer_ratio_clip_active_fraction':0.0 if max(maxima)<.2 else None,
         'outer_ratio_clip_inference':'All evaluated ratios inside clip range' if max(maxima)<.2 else 'Exact clipped fraction not logged',
         'internal_per_MC_log_ratio_clip_active_fraction':None,
         'fresh_old_version_matches_previous_optimizer_step':all(r['old_policy']['version']==r['step']-1 and r['old_policy']['optimizer_step']==r['step']-1 for r in rows),
         'distinct_logged_old_state_hashes':len(set(r['old_policy']['state_hash'] for r in rows))}
    if arm in ('C2','C4'):
        f['reward_nonzero_candidate_fraction_from_binary_mean'] = -weighted_reward
    elif arm == 'C3':
        f['reward_nonzero_candidate_fraction'] = None
        f['NC_nonzero_penalty_fraction_bounds_from_0_0p5_1'] = [1-metrics_mean['no_at_fault_collisions'],min(1,2*(1-metrics_mean['no_at_fault_collisions']))]
    result['feedback'] = f
    return result


audit={'schema':'st_consequence4096_scalar_training_audit_v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'scope':'The actual C0-C4 fulltrain seed42 official4096 runs only; no DEV results used to select windows or tune.',
       'analysis_source_base_sha':'cef257107dd802bc1b5ef9c88cb996a4c77fabe9',
       'training_source_sha':'6714b1e9fb2e0c8059b90b8b4d48cd40796fe345','windows_predefined':WINDOWS,
       'scalar_jsonl_read_budget_bytes':BUDGET,'prior_reconnaissance_reserved_bytes':used,
       'payloads_read':False,'official_scoring_performed':False,'arms':{}}
sequence_hashes={}
for arm in ARMS:
    run=ROOT/f'{arm}_fulltrain_seed42_official_4096_v1'
    result=read_small(run/'RESULT.json'); cfg=read_small(run/'resolved_config.json'); identity=read_small(run/'identity.json')
    assert cfg['cell']==arm and cfg['execution']['max_updates']==4096
    assert result['completed']==4096 and result['status']=='COMPLETE' and result['termination_reason']=='MAX_STEPS'
    assert result['source_sha']==audit['training_source_sha']
    path=run/'steps.jsonl'; before=path.stat()
    if used+before.st_size>BUDGET:raise ValueError('Scalar read budget exhausted')
    rows=[];digest=hashlib.sha256();sequence=hashlib.sha256();unique_ids=set()
    with path.open('rb',buffering=1024*1024) as stream:
        for line in stream:
            used+=len(line)
            if used>BUDGET:raise ValueError('Scalar read budget exhausted')
            digest.update(line);row=json.loads(line)
            assert row['step']==len(rows)+1
            assert len(row['per_rank'])==4
            sample_ids=[p['sample_ids'] for p in row['per_rank']]
            flat=[s for part in sample_ids for s in part]
            unique_ids.update(flat)
            assert len(flat)==row['counts']['original_GT_scenes']
            sequence.update(json.dumps({'step':row['step'],'ranked_scene_ids':sample_ids,'lr':row['lr']},sort_keys=True,separators=(',',':')).encode())
            rows.append(row)
    after=path.stat()
    assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
    assert len(rows)==4096
    windows={name:aggregate([r for r in rows if lo<=r['step']<=hi],arm) for name,(lo,hi) in WINDOWS.items()}
    assert windows['all_1_4096']['counts']==result['counts']
    assert windows['all_1_4096']['counts']['original_GT_scenes']==result['exposure']
    sequence_hashes[arm]=sequence.hexdigest()
    audit['arms'][arm]={'run':str(run),'steps':{'path':str(path),'bytes':after.st_size,'sha256':digest.hexdigest()},
      'result_identity':result['identity'],'result_completed':result['completed'],
      'result_new_optimizer_updates_last_invocation_only':result['new_optimizer_updates'],
      'frozen_encoder_unchanged_recorded':result['frozen_encoder_unchanged'],
      'frozen_encoder_hash_recorded':result['frozen_encoder_hash'],
      'unique_scene_ids_in_scalar_log':len(unique_ids),
      'sample_order_and_lr_sha256':sequence.hexdigest(),
      'recipe':{k:cfg[k] for k in ['cell','execution','fpo','loss','optimizer']},
      'feedback_protocol_hash':cfg['feedback']['score_protocol_hash'],
      'training_identity_sha256':next(p['sha256'] for p in pins if p['path']==str(run/'identity.json')),
      'windows':windows}
    del rows
assert len(set(sequence_hashes.values()))==1
source_symbols={'tools/st_world_policy/train_flow_policy.py':['feedback_from_records','FlowTrainingModule.forward','train: old snapshot, official_statistics, per_rank/log writing'],
 'starVLA/model/modules/flow_posttraining/fpo_objective.py':['fpo_ratio','group_centered_advantages','fpo_surrogate'],
 'tools/ddpolicy_vehicle/optimizer_safety.py':['capture_master_samples','master_update_evidence']}
audit['source_semantics']=[{'path':str(RUNTIME/p),'sha256':hashlib.sha256((RUNTIME/p).read_bytes()).hexdigest(),'symbols':s} for p,s in source_symbols.items()]
audit['small_control_pins']=pins
audit['scalar_bytes_read_this_scan']=used-audit['prior_reconnaissance_reserved_bytes']
audit['cumulative_scalar_bytes_upper_bound']=used
audit['matched_sample_order_and_lr_across_all_arms']=True
audit['missing_measurements']={'policy_only_gradient_norm':None,'BC_only_gradient_norm':None,'policy_to_BC_gradient_ratio_or_cosine':None,
 'full_parameter_delta_norm':None,'per_candidate_advantage_distribution':None,'NC_failure_count_exact':None,
 'infrastructure_failed_attempts_before_completed_steps':None,'canonical_zero_score_candidate_count':None,
 'event_first_time_or_severity_distribution':None,'post_update_policy_ratio':None}
audit['limitations']=[
 'Only completed-update scalar summaries are read. Full inference/feedback files and model weights are not read or reverified.',
 'Exposure counts include repeated training scenes. Unique scene count comes from recorded sample IDs, not an independent dataset audit.',
 'Combined gradient and sampled master changes prove total optimization; they do not isolate causal contribution from the policy term.',
 'Fresh old=current and within-scene centered advantage make the pre-update policy scalar approximately zero and the ratio approximately one. This does not imply zero policy gradient.',
 'Binary DAC/TTC means identify candidate failure frequencies. NC has 0/0.5/1 penalties, so its mean identifies weighted penalty but not exact failure frequency.',
 'All-negative equal-reward groups have zero centered advantage despite real failures. Zero-advantage rate measures missing within-scene ranking contrast, not absence of all failures.',
 '4096 update timing is descriptive, excludes checkpoint/setup/cleanup, and is not a dedicated isolated performance profile.',
 'No conclusion about final heldout efficacy is established by training scalars. Final fixed DEV scoring is separate.']
(OUT/'TRAINING_SIGNAL_AUDIT.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
print(json.dumps({'scalar_bytes':audit['scalar_bytes_read_this_scan'],'bounded_total':used,'same_sample_order':True,
 'arms':{a:{'updates':audit['arms'][a]['result_completed'],'scene_exposures':audit['arms'][a]['windows']['all_1_4096']['counts']['original_GT_scenes'],
 'feedback':audit['arms'][a]['windows']['all_1_4096']['feedback'],
 'bc_mean':audit['arms'][a]['windows']['all_1_4096']['loss']['bc']['mean'],
 'policy_abs_max':audit['arms'][a]['windows']['all_1_4096']['absolute_policy_loss']['max'],
 'grad_mean':audit['arms'][a]['windows']['all_1_4096']['gradient_norm_combined_bc_and_policy']['mean']}
 for a in ARMS}},indent=2))
