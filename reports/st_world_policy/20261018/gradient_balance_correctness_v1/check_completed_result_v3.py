"""Small-file evidence review of an already completed gradient diagnostic.

Pure Python: no torch/model/GPU/score/manifest/GT/checkpoint loading.
Does not alter the result or the failed supervisor receipt and never dispatches.
"""
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import time

resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
START = time.monotonic()
HERE = Path(__file__).resolve().parent
ROOT = Path('/mnt/project/st-world-policy-20261016/audits/consequence')
RUNTIME = Path('/mnt/project/VLA-Drive-st-consequence-diagnostics-runtime-v2-20261018')
RUN = ROOT/'gradient_balance_boundary3584_v3'
CONTROL = ROOT/'gradient_balance_boundary3584_v3_controls'
READS = []
CHECKS = 0


def check(ok, message):
    global CHECKS
    CHECKS += 1
    if not ok:
        raise AssertionError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, expected=None):
    path = Path(path)
    check(path.stat().st_size + sum(x['bytes'] for x in READS) <= 2 << 20, 'small-file read budget')
    raw = path.read_bytes()
    h = digest(raw)
    check(expected is None or h == expected, 'pinned input ' + str(path))
    READS.append({'path': str(path), 'bytes': len(raw), 'sha256': h})
    return raw


def document(path, expected=None):
    return json.loads(read(path, expected))


def near(actual, expected, label):
    if expected is None:
        check(actual is None, label)
    else:
        check(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=2e-12, abs_tol=1e-24), label)


def main():
    result_hash = '7ffd159b3dda244b40dec03ad09c3142f47bc50c395d1be70bda9127c8203a2b'
    r = document(RUN/'RESULT.json', result_hash)
    registration_hash = '6a48f942e8b57f568d74be0bd92cab389daa6140d0a0307896acd40da3358770'
    registration = document(CONTROL/'REGISTRATION.json', registration_hash)
    check(read(RUN/'REGISTRATION.json') == json.dumps(registration, sort_keys=True, indent=2).encode()+b'\n', 'atomic output registration content')
    supervisor_hash = '80aedaec45221c99c903ec01442d50f9fb2cecfcfac76cd96f1c79acebaa937c'
    supervisor = document(CONTROL/'RUNTIME_ATTEMPT/SUPERVISOR_RESULT.json', supervisor_hash)
    counters = document(CONTROL/'RUNTIME_ATTEMPT/COUNTERS.json', '232fad015af6728229b64309ee0a0392c1add4651dd135050fde37d732c7e710')
    log = read(CONTROL/'RUNTIME_ATTEMPT/CHILD.log', '8a378ab841a16479fc6dc6e9f4e44dcd0344e131cb977001d5f7727e96888d23').decode()
    admission = document(CONTROL/'ADMISSION.json', '3a4f3b12a5dffe338f4e21d201efbc9be5873b972d25254ae147e3cda8ae467a')
    math_source = read(RUNTIME/'tools/st_world_policy/audit_flow_gradient_balance.py', 'a8c376a5c9cb35becebf9e2ec942cf7df176d0e6f156fd457e423f3f2b54a6e2')
    ast.parse(math_source)  # Parse only; no implementation import or execution.
    prior = document(HERE/'SHARED_V3_EXECUTION_REVIEW.json', '284ae02923107d1cfe43f03c94510643fddce9ae838d787f19e52753af1d672a')
    signal_path = Path('/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261018/consequence_next_hypothesis_v1/RESULTS.json')
    signal = document(signal_path, 'f1e54540107e6f76d2b9ccd3d5eb9c4d53207b1dae8e49e1a3f245557bbbe986')
    check(registration['source_sha'] == prior['source_sha'] == admission['source_sha'] == '83679937fe9702be39f0184154fdf9ebdc4eb98a', 'unchanged reviewed math source')
    check(admission['registration_sha256'] == registration_hash and
          admission['wrapper_sha256'] == prior['wrapper_sha256'] and
          admission['runtime_environment'] == prior['runtime_environment'] and
          admission['gpu_admission'] == prior['gpu_admission'], 'prior wrapper/environment/admission identity')
    check(r['status'] == 'DIAGNOSTIC_COMPLETE' and 'error' not in r, 'complete mathematical result')
    check(r['schema'] == 'st_flow_gradient_balance_v1' and not r['formal_training'] and
          r['optimizer_steps'] == r['new_score_calls'] == 0 and not r['metric_cache_read'], 'no update or new scoring scope')
    check(r['caller_RNG_unchanged'] is True and r['original_weights_unchanged'] is True, 'completed guarded immutability assertions')
    check(r['native_checkpoint']['completed'] == 3584 and
          r['native_checkpoint']['tag'] == registration['checkpoint_tag'] and
          r['native_checkpoint']['native_receipt_sha256'] == registration['checkpoint_receipt']['sha256'], 'native boundary3584 receipt')
    check(r['native_checkpoint']['base_binding'] == signal['identities']['base_binding'] and
          r['native_checkpoint']['training_numeric_execution_hash'] == signal['identities']['numeric_execution_hash'], 'same original Base and numeric identity as signal population')
    check(r['cursor']['epoch'] == 1 and r['cursor']['offset'] == 409 and len(r['cursor']['manifest_indices']) == 32,
          'original next32 cursor')
    receipt_evidence = [row for row in signal['limits_and_execution']['read_ledger']
                        if row['path'].endswith('/attempt_00/COMPLETE.json')]
    check(len(receipt_evidence) == 1 and r['official_receipts'] == [
        {'path': receipt_evidence[0]['path'], 'sha256': receipt_evidence[0]['sha256']}],
        'same existing official feedback receipt')
    check(r['encoder_calls'] == {'visual': 8, 'language': 8}, 'all original encoder microbatches')
    check(len(r['rank_reports']) == 4, 'all four ranks complete')
    ordered_scenes = []
    for rank, item in enumerate(r['rank_reports']):
        check(item['rank'] == rank and len(item['scene_ids']) == 8, 'rank order and count')
        ordered_scenes.extend(item['scene_ids'])
        check(len(item['current_CFM']) == 2 and
              all(x['bit_exact'] is True and x['elements'] == 800 and len(x['sha256']) == 64 for x in item['current_CFM']),
              'current/old CFM50 exact guards')
        check(set(item['rng_before']) == set(item['rng_after_BC']) == {'python', 'numpy', 'torch_cpu', 'torch_cuda'} and
              set(item['bc_streams_after']) == {'bc_noise', 'bc_time'}, 'complete rank RNG evidence')
    check(ordered_scenes == [row['scene_id'] for row in signal['per_scene']], 'same 32 scenes as independent EP-only analysis')
    check(sum(x['nonzero_advantages'] for x in r['rank_reports']) == 64 and
          sum(x['zero_advantage_scenes'] for x in r['rank_reports']) == 16, 'same actual reward-support counts')
    g = r['gradient_balance']; weight = g['policy_weight']
    check(g['microbatch_counts'] == {'bc': 8, 'policy': 8} and weight == 1., 'all16 separate backward contributions')
    check(g['analysis_accumulation'] == 'torch.float32_CPU' and g['dot_reduction'] == 'chunked_float64' and
          g['execution_gradient_dtypes'] == ['torch.bfloat16'] and g['exact_ZeRO2_update_claim'] is False,
          'correct numerical interpretation')
    check(len(g['parameters']) == 359, 'all trainable parameter summaries present')
    sums = {k: {'bc_squared_l2': 0., 'policy_squared_l2': 0., 'dot': 0., 'elements': 0}
            for k in ('all','DiT','qwen_proj','other_action')}
    for name, stat in g['parameters'].items():
        group = 'qwen_proj' if name.startswith('action_head.qwen_proj.') else 'DiT' if name.startswith('action_head.model.') else 'other_action'
        for dest in ('all',group):
            for field in sums[dest]: sums[dest][field] += stat[field]
    for name, stat in list(g['parameters'].items()) + list(g['groups'].items()):
        b, p, d = stat['bc_squared_l2'], stat['policy_squared_l2'], stat['dot']
        check(b >= 0 and p >= 0 and isinstance(stat['elements'],int) and stat['elements'] > 0, 'valid norm counts')
        nb, np_ = math.sqrt(b), math.sqrt(p)
        expected = {'bc_l2':nb,'policy_l2':np_,'weighted_policy_l2':abs(weight)*np_,
                    'cosine':d/(nb*np_) if nb*np_ else None,
                    'weighted_policy_over_bc':abs(weight)*np_/nb if nb else None,
                    'combined_gradient_l2':math.sqrt(max(0.,b+weight*weight*p+2*weight*d)),
                    'weighted_cross_term':2*weight*d,
                    'first_order_BC_change_under_combined_descent_per_unit_lr':-(b+weight*d),
                    'first_order_policy_change_under_combined_descent_per_unit_lr':-(d+weight*p)}
        for field, value in expected.items(): near(stat[field],value,name+':'+field)
        if stat['cosine'] is not None: check(abs(stat['cosine']) <= 1.+1e-12, 'valid cosine range')
    for group, aggregate in sums.items():
        for field, value in aggregate.items(): near(g['groups'][group][field],value,group+':sum parameters '+field)

    check(supervisor['status'] == 'STOPPED_NO_AUTO_RETRY' and supervisor['reason'] == 'MONITOR_OR_CHILD_SETUP_FAILURE' and
          supervisor['error']['message'] == supervisor['final_monitor_error']['message'] == 'Missing live RSS counter', 'supervision failure preserved')
    check(supervisor['exit_code'] == 0 and supervisor['live_owned_remaining'] == 0, 'child exit and cleanup receipt')
    completed_lines = [json.loads(line) for line in log.splitlines() if line.startswith('{"status":')]
    check(completed_lines[-1] == {'status':'DIAGNOSTIC_COMPLETE','output':str(RUN),'error':None}, 'child final complete log after atomic write')
    check(counters['rejected_admission'] is None and supervisor['charged_read_bytes'] ==
          sum(counters[k] for k in ('hash_nominal_bytes','torch_load_nominal_bytes','safetensors_nominal_bytes')), 'nominal charged reads reconciled')
    limits = registration['budget']
    check(supervisor['charged_read_bytes'] < limits['max_read_bytes'] and
          supervisor['peak_observed_RSS_bytes'] < limits['max_rss_bytes'] and
          supervisor['wall_seconds'] < limits['max_seconds'] and
          supervisor['output_bytes_before_receipt'] < limits['max_output_bytes'], 'observed counters within limits, not a successful final monitor')
    all_group = g['groups']['all']
    out = {'schema':'gradient_balance_completed_v3_result_correctness_review_v1',
        'status':'MATHEMATICAL_RESULT_USABLE_WITH_SUPERVISION_FAILURE_RETAINED',
        'reviewer':'/root/road_proxy_correctness_review','review_kind':'completed_result_correctness',
        'created_utc':datetime.now(timezone.utc).isoformat(),'result_path':str(RUN/'RESULT.json'),
        'result_sha256':result_hash,'registration_sha256':registration_hash,
        'source_sha':'83679937fe9702be39f0184154fdf9ebdc4eb98a',
        'wrapper_sha256':admission['wrapper_sha256'],'supervisor_result_sha256':supervisor_hash,
        'mathematical_status':'DIAGNOSTIC_COMPLETE','supervisor_status':supervisor['status'],
        'supervisor_final_error':supervisor['final_monitor_error'],'child_exit_code':supervisor['exit_code'],
        'live_owned_remaining_receipt':supervisor['live_owned_remaining'],
        'all_parameter_summary':all_group,'groups':g['groups'],'parameter_tensor_count':359,
        'same_population_EP_only_signal_report_sha256':'f1e54540107e6f76d2b9ccd3d5eb9c4d53207b1dae8e49e1a3f245557bbbe986',
        'policy_over_BC_percent':100*all_group['weighted_policy_over_bc'],
        'BC_over_policy_norm':1/all_group['weighted_policy_over_bc'],
        'caller_RNG_unchanged_guard_completed':True,'weight_and_buffer_unchanged_guards_completed':True,
        'native_loading_condition_Euler_CFM_guards_completed':True,
        'independent_parameter_or_RNG_reread':False,'new_model_runs':0,'new_score_calls':0,
        'optimizer_updates':0,'formal_training_approved':False,'automatic_retry_approved':False,
        'limits':[
            'Local separate-backward gradient geometry for one fixed C1 batch, not exact original joint ZeRO2, clipped Adam update, convergence or planning gain',
            'A small raw-gradient policy/BC norm ratio and weakly negative cosine do not identify a justified new coefficient',
            'These current candidates only carry EP differences; this is not a direct safety-gradient dominance experiment',
            'Immutable result completion follows all fail-closed numeric/RNG/weight checks in the pinned source; reviewer did not rerun model or reread original weights',
            'Supervisor final RSS read failed despite child exit0; observed counters and child self-peak are available, final whole-process monitor success is not established',
            'No observed overrun and cleanup receipt zero are preserved; do not relabel STOPPED_NO_AUTO_RETRY as a fully PASS runtime',
        ],
        'diagnostic_resource_self_report':r['resources'],'supervisor_observed_resources':supervisor,
        'checks_passed':CHECKS,'read_ledger':READS,
        'review_execution':{'wall_seconds':time.monotonic()-START,
            'cpu_seconds':resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'logical_read_bytes':sum(x['bytes'] for x in READS),'GPU_or_torch_import':False}}
    target = HERE/'COMPLETED_V3_RESULT_REVIEW.json'
    check(not target.exists(),'review immutable')
    target.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':out['status'],'all':all_group,'checks':CHECKS,'review':out['review_execution'],
                      'output_sha256':digest(target.read_bytes())}))


if __name__ == '__main__':
    main()
