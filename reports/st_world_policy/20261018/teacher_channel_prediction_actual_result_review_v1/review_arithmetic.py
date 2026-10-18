"""Verify bounded existing result metadata and scalar arithmetic only."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/mnt/project/st-world-policy-20261016/audits/teacher_channel_prediction_diagnostic_v2')
OUT = Path(__file__).resolve().parent
pins = {}
def read(path, expected=None):
    assert path.stat().st_size < 2 << 20
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if expected:
        assert sha == expected
    pins[str(path)] = sha
    return json.loads(raw)

d = read(ROOT / 'RESULT.json', '6b2b4f6c037b15c3b6f78bca366da57f06ec29f339a55b7e25240bcad97eb0ed')
prefix = read(ROOT / 'TARGET_PREFIX_CHECK.json')
invocation = read(ROOT / 'INVOCATION.json')
control = read(ROOT.with_name(ROOT.name + '_controls') / 'RESULT.json')
assert d['status'] == 'COMPLETE_DIAGNOSTIC_ONLY'
assert d['optimizer_steps'] == 0 and not d['training_loss_claimed'] and not d['gradient_conflict_measured']
assert d['observation_identity']['split'] == 'train'
assert not d['observation_identity']['targets_read'] and not d['observation_identity']['teacher_cache_read']
assert d['observation_identity']['manifest_sha256'] == '77ebc29d9273c309afbbb3e596eb95ac947fbe51d96830f31f7552ca00881947'
assert prefix['exact_equal'] and prefix['max_abs'] == d['native_PCA256_vs_common384_target_max_abs'] == 0
assert prefix['target_sha256'] == d['target_sha256'] and prefix['pca_identity'] == d['pca_basis_identity']
scenes = d['per_scene']
assert len(scenes) == 8 and len({p['scene'] for p in scenes}) == 8
assert len(d['requests']) == 8
for p, views in zip(scenes, d['requests']):
    assert len(views) == 3
    assert p['metrics']['valid_requests'] == 288
    for i, requests in enumerate(views):
        assert len(requests) == len(set(requests)) == 96
        assert all(k.startswith(p['scene'] + '/' + str(i) + '/') for k in requests)
        assert all(0 <= int(k.rsplit('/',1)[1]) < 432 for k in requests)
assert d['metrics']['valid_requests'] == sum(p['metrics']['valid_requests'] for p in scenes) == 2304

def near(a, b):
    assert math.isclose(a, b, rel_tol=2e-12, abs_tol=2e-12), (a, b)

def block(v, requests):
    assert v['elements'] == requests * v['channels']
    near(v['sse'] / v['elements'], v['mse'])
    near(v['sse'] / v['target_energy'], v['relative_sse_to_zero'])
    near(math.fsum(v['per_component_mse']) / v['channels'], v['mse'])
    assert len(v['per_component_mse']) == len(v['per_component_target_mean']) == len(v['per_component_target_variance']) == v['channels']
    near(math.fsum(vr + mean**2 for vr, mean in zip(v['per_component_target_variance'], v['per_component_target_mean'])) / v['channels'], v['target_energy'] / v['elements'])

for metric in [d['metrics']] + [p['metrics'] for p in scenes]:
    n = metric['valid_requests']
    c256, c384 = [metric['common_first256'][key] for key in ('arm256', 'arm384')]
    tail = metric['extra_128']['arm384']
    for b in (c256, c384, tail):
        block(b, n)
    near(c256['target_energy'], c384['target_energy'])
    near(tail['target_energy'] / tail['elements'], metric['extra_128']['zero_prediction_mse'])
    near((256*c384['mse'] + 128*tail['mse']) / 384, metric['arm384_full_mse'])
    near(metric['arm384_weighted_block_mse'], metric['arm384_full_mse'])
    assert metric['extra_128']['arm256'] == 'NOT_PREDICTED_NOT_ZERO_ERROR'
for group, arm in [('common_first256','arm256'),('common_first256','arm384'),('extra_128','arm384')]:
    global_block = d['metrics'][group][arm]
    for k in ('sse', 'target_energy', 'elements'):
        near(math.fsum(p['metrics'][group][arm][k] for p in scenes), global_block[k])
for arm in ('256','384'):
    record = read(ROOT / ('ARM_' + arm + '.json'))
    assert record == d['arms'][arm]
    assert record['actual_encode_calls'] == record['actual_readout_calls'] == 8
    assert record['rng_unchanged'] and record['rng_before'] == record['rng_after']
    assert not record['optimizer_created'] and not record['training_forward_called']
    assert record['checkpoint']['completed'] == 5000
assert control['status'] == 'FAILED_NO_AUTO_RETRY'
res = control['resources']
assert res['returncode'] == 0 and res['failure'] == 'OWNED_DESCENDANTS_AFTER_LEADER_EXIT'
assert res['cleanup'] == {'live_owned_remaining': 0, 'verified_processes': 10}
assert control['cli_result']['result'] == {'path': str(ROOT/'RESULT.json'), 'sha256': pins[str(ROOT/'RESULT.json')]}
m = d['metrics']; a = m['common_first256']['arm256']; b = m['common_first256']['arm384']; t = m['extra_128']['arm384']
report = {
    'status': 'PASS_RESULT_METADATA_AND_ARITHMETIC_WITH_RETAINED_CONTROL_FAILURE',
    'reviewer': '/root/temporal_narrow_review', 'evidence_pins': pins,
    'source_review_inherited': {'commit':'00840b1a','fixed_source':'0388faff949401ceb777176bab2e64311e6f23e9'},
    'scenes':8,'logs':len({p['log'] for p in scenes}),'valid_requests':2304,
    'requests_per_scene':288,'views':3,'positions_per_view':96,
    'prefix':{'exact_equal':True,'max_abs':0,'scope':'Actual native PCA256 target equals first256 of actual384 basis; no new tensor calculation in review.'},
    'common256':{'arm256_mse':a['mse'],'arm384_mse':b['mse'],'delta256_minus384':a['mse']-b['mse'],
        'relative_error_reduction_percent':100*(1-a['mse']/b['mse']),
        'zero_prediction_mse':a['target_energy']/a['elements'],
        'arm256_relative_sse_to_zero':a['relative_sse_to_zero'],'arm384_relative_sse_to_zero':b['relative_sse_to_zero'],
        'scenes_lower_error256':sum(p['metrics']['common_first256']['arm256']['mse']<p['metrics']['common_first256']['arm384']['mse'] for p in scenes)},
    'tail128':{'arm384_mse':t['mse'],'zero_prediction_mse':m['extra_128']['zero_prediction_mse'],
        'relative_sse_to_zero':t['relative_sse_to_zero'],'error_reduction_vs_zero_percent':100*(1-t['relative_sse_to_zero']),
        'arm256':'NOT_PREDICTED_NOT_ZERO_ERROR'},
    'arm384_full_mse':m['arm384_full_mse'],'arm384_decomposition':m['arm384_weighted_block_mse'],
    'arithmetic_checks':['SSE/elements=MSE','SSE/target_energy=relative_SSE','mean per-componentMSE=blockMSE',
        'mean(variance+mean^2)=zero-MSE','per-scene SSE/energy/count sums match aggregate',
        '384MSE=(256commonMSE+128tailMSE)/384','all2304 requests valid unique within scene/view'],
    'controller':{'status':control['status'],'child_returncode':0,'failure':res['failure'],'cleanup':res['cleanup'],
        'wall_seconds':res['invocation_seconds'],'charged_rchar_bytes':control['charged_parent_child_rchar_bytes'],
        'peak_rss_bytes':res['peak_aggregate_rss_bytes'],'automatic_retry':False},
    'actual_runtime':invocation['runtime'],
    'interpretation':['A complete matched8-training-scene eval-FP32 feature diagnostic is available; it is not a clean controller success.',
        'The shared256 error is slightly lower for256 on these scenes, and the384 tail is only about3% better than zero. This supports a limited fit/target-energy observation, not a causal gradient-conflict finding.',
        'Full384MSE cannot be directly compared with256MSE as training superiority: the low-energy tail changes the denominator. Neither model was trained/updated by this diagnostic.',
        'Single seed,8fixed training scenes, no held-out/generalization or planning measurement, no gradients. Existing256-vs384 training loss mean has different common-block coefficient; do not infer pure capacity causality.',
        'No model, teacher, image, tensor payload, simulator or bootstrap was loaded or run by this reviewer. No retry requested solely to turn controller status green.']}
(OUT/'REVIEW.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'review_sha256':hashlib.sha256((OUT/'REVIEW.json').read_bytes()).hexdigest(),'common256':report['common256'],'tail128':report['tail128']}))
