"""Read completed diagnostic JSON only; never loads models, tensors or scores."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/mnt/project/st-world-policy-20261016/audits/teacher_channel_prediction_diagnostic_v2')
CONTROL = ROOT.with_name(ROOT.name + '_controls')
HERE = Path(__file__).resolve().parent
SOURCE = Path('/mnt/project/VLA-Drive-st-teacher-channel-diagnostic-fix-20261018')
SOURCE_SHA = '0388faff949401ceb777176bab2e64311e6f23e9'
READS = []


def read(path, cap=2 << 20):
    stat = path.stat()
    if not path.is_file() or path.is_symlink() or stat.st_size > cap:
        raise ValueError('Expected bounded diagnostic JSON')
    raw = path.read_bytes()
    if (stat.st_ino, stat.st_size, stat.st_mtime_ns) != (
            path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns):
        raise ValueError('Receipt changed while reading')
    READS.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)))
    return json.loads(raw)


def dump(name, value):
    (HERE/name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def main():
    if not (CONTROL/'RESULT.json').exists():
        raise SystemExit('WAITING_TERMINAL: no result claims emitted')
    terminal = read(CONTROL/'RESULT.json')
    invocation = read(ROOT/'INVOCATION.json')
    native_pin = terminal.get('cli_result', {}).get('result')
    native_complete = (terminal.get('cli_result', {}).get('status') == 'COMPLETE_DIAGNOSTIC_ONLY' and
        native_pin is not None and native_pin.get('path') == str(ROOT/'RESULT.json') and
        terminal.get('resources', {}).get('returncode') == 0 and (ROOT/'RESULT.json').exists())
    if not native_complete:
        error = read(ROOT/'ERROR.json') if (ROOT/'ERROR.json').exists() else None
        dump('SUMMARY.json', dict(status='FAILED_INCOMPLETE_DIAGNOSTIC',
            source_sha=SOURCE_SHA, primary_error=error, terminal=terminal, inputs=READS,
            automatic_retry=False, common_channel_comparison_available=False))
        (HERE/'RESULTS.md').write_text('# Teacher-channel diagnostic did not complete\n\n'
            'The actual controller failed. Original failure evidence is preserved; '
            'no arm comparison or successful convergence claim is available. '
            'See SUMMARY.json for the primary error and exact receipt identities. '
            'No retry or model computation was performed by this report.\n')
        return
    result = read(ROOT/'RESULT.json')
    assert READS[-1]['sha256'] == native_pin['sha256']
    prefix = read(ROOT/'TARGET_PREFIX_CHECK.json')
    arms = {d: read(ROOT/f'ARM_{d}.json') for d in ('256','384')}
    selection = read(Path(invocation['binding']['selection']['path']))
    assert READS[-1]['sha256'] == invocation['binding']['selection']['sha256']
    assert result['status'] == 'COMPLETE_DIAGNOSTIC_ONLY'
    assert result['mode'] == 'eval_FP32_TF32_off'
    assert result['optimizer_steps'] == 0 and not result['training_loss_claimed']
    assert prefix['exact_equal'] is True and prefix['max_abs'] == 0
    assert result['native_PCA256_vs_common384_target_max_abs'] == 0
    assert result['requests'] == prefix['requests']
    assert result['target_sha256'] == prefix['target_sha256']
    assert result['pca_basis_identity'] == prefix['pca_identity']
    mask = [v for scene in prefix['valid_mask'] for view in scene for v in view]
    ids = [v for scene in prefix['requests'] for view in scene for v in view]
    assert all(type(v) is bool for v in mask) and len(ids) == len(set(ids))
    metrics = result['metrics']
    assert metrics['valid_requests'] == sum(mask)
    assert len(selection['scenes']) == len(prefix['requests']) == 8
    assert [s['token'] for s in selection['scenes']] == [r['scene'] for r in result['per_scene']]
    for d, arm in arms.items():
        assert arm == result['arms'][d]
        assert arm['checkpoint']['completed'] == 5000
        assert arm['mode'] == result['mode'] and arm['rng_unchanged']
        assert arm['rng_before'] == arm['rng_after']
        assert arm['actual_encode_calls'] == arm['actual_readout_calls'] == 8
        assert not arm['optimizer_created'] and not arm['training_forward_called']
    common = metrics['common_first256'];tail = metrics['extra_128']['arm384']
    a,b = common['arm256'],common['arm384']
    assert metrics['extra_128']['arm256'] == 'NOT_PREDICTED_NOT_ZERO_ERROR'
    assert a['elements'] == b['elements'] == sum(mask)*256
    assert tail['elements'] == sum(mask)*128
    assert a['target_energy'] == b['target_energy']
    assert math.isclose(metrics['arm384_full_mse'],metrics['arm384_weighted_block_mse'],rel_tol=1e-12)
    for block in (a,b,tail):
        assert all(math.isfinite(block[k]) for k in ('mse','sse','target_energy'))
        assert math.isclose(block['mse']*block['elements'],block['sse'],rel_tol=1e-12)
    delta = a['mse']-b['mse']
    rows=[]
    for entry in result['per_scene']:
        m=entry['metrics'];x=m['common_first256']['arm256']['mse'];y=m['common_first256']['arm384']['mse']
        rows.append(dict(scene=entry['scene'],log=entry['log'],valid_requests=m['valid_requests'],
            mse256_common=x,mse384_common=y,delta_256_minus_384=x-y,
            mse384_tail=m['extra_128']['arm384']['mse']))
    assert math.isclose(sum(r['mse256_common']*r['valid_requests'] for r in rows)/sum(mask),a['mse'],rel_tol=1e-12)
    assert math.isclose(sum(r['mse384_common']*r['valid_requests'] for r in rows)/sum(mask),b['mse'],rel_tol=1e-12)
    for block in (a,b,tail):
        for key in ('per_component_mse','per_component_target_mean','per_component_target_variance'):
            assert len(block[key]) == block['channels'] and all(math.isfinite(x) for x in block[key])
    with (HERE/'PER_SCENE.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (HERE/'PER_COMPONENT.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['component_1based','arm256_mse','arm384_mse','target_mean','target_variance'])
        for i in range(384):
            block=b if i<256 else tail;j=i if i<256 else i-256
            writer.writerow([i+1,a['per_component_mse'][i] if i<256 else 'NOT_PREDICTED',
                block['per_component_mse'][j],block['per_component_target_mean'][j],block['per_component_target_variance'][j]])
    evidence = dict(status=('COMPLETE_READ_ONLY_INTERPRETATION' if terminal['status']=='COMPLETE_DIAGNOSTIC_ONLY'
                           else 'COMPLETED_NUMERICAL_ARTIFACTS_SUPERVISOR_FAILED'),source_sha=SOURCE_SHA,
        supervisor_status=terminal['status'],supervisor_failure=terminal['resources']['failure'],
        native_returncode=terminal['resources']['returncode'],cleanup=terminal['resources']['cleanup'],
        native_result_sha256=native_pin['sha256'],independent_artifact_adoption=False,
        invocation_seconds=terminal['resources']['invocation_seconds'],
        charged_rchar_bytes=terminal['charged_parent_child_rchar_bytes'],
        peak_aggregate_rss_bytes=terminal['resources']['peak_aggregate_rss_bytes'],
        mmap_physical_io_measured=False,
        mode=result['mode'],scenes=8,declared_logs=len({x['log'] for x in rows}),
        requested_positions=len(mask),valid_requests=sum(mask),unique_request_ids=len(set(ids)),
        common_elements=a['elements'],tail_elements=tail['elements'],target_prefix_exact=True,
        target_prefix_max_abs=0,common256_mse_arm256=a['mse'],common256_mse_arm384=b['mse'],
        common_delta_256_minus_384=delta,
        common_relative_reduction_by256=(b['mse']-a['mse'])/b['mse'],
        arm256_common_relative_sse_to_zero=a['relative_sse_to_zero'],
        arm384_common_relative_sse_to_zero=b['relative_sse_to_zero'],
        arm384_tail128_mse=tail['mse'],arm384_tail_relative_sse_to_zero=tail['relative_sse_to_zero'],
        tail_zero_prediction_mse=metrics['extra_128']['zero_prediction_mse'],
        arm384_full384_mse=metrics['arm384_full_mse'],
        scenes_lower_common_error_for256=sum(r['delta_256_minus_384']<0 for r in rows),
        scenes_lower_common_error_for384=sum(r['delta_256_minus_384']>0 for r in rows),
        common_components_lower_error_for256=sum(x<y for x,y in zip(a['per_component_mse'],b['per_component_mse'])),
        finite_valid_outputs=True,model_recomputed_by_analysis=False,dev_scores_read=False,
        training_loss_claimed=False,gradient_conflict_measured=False,planning_gain_proven=False,
        pca_basis_identity=result['pca_basis_identity'],inputs=READS,
        interpretation='SINGLE_TRAIN_SUBSET_ENDPOINT_EVIDENCE_ONLY',
        arms={d:{k:arm[k] for k in ('elapsed_seconds','gpu_peak_allocated','gpu_peak_reserved','rng_unchanged','actual_encode_calls','actual_readout_calls')} for d,arm in arms.items()})
    dump('SUMMARY.json',evidence)
    direction='lower' if delta<0 else ('higher' if delta>0 else 'equal')
    text=f'''# Same-request teacher-channel prediction diagnostic

The native numerical result is complete and hash-bound by the child accounting
envelope (return code 0). **The supervisor status remains {terminal['status']}**:
`{terminal['resources']['failure']}`; cleanup reports
{terminal['resources']['cleanup']['live_owned_remaining']} live owned processes
after verifying {terminal['resources']['cleanup']['verified_processes']} identities.
The following is descriptive analysis of completed artifacts, not a rewritten
successful execution receipt or independent artifact-adoption approval.

The 256-channel endpoint has **{direction} common-coordinate error** on this fixed
8-scene training subset: {a['mse']:.8f} versus {b['mse']:.8f} MSE on the exact same
first 256 PCA coordinates ({(b['mse']-a['mse'])/b['mse']*100:+.3f}% relative reduction).
This is one read-only endpoint comparison, not a convergence or planning result.

| Diagnostic block | PCA256 model | PCA384 model |
|---|---:|---:|
| Same first 256 coordinates: MSE | {a['mse']:.8f} | {b['mse']:.8f} |
| Same first 256: SSE / zero-prediction SSE | {a['relative_sse_to_zero']:.8f} | {b['relative_sse_to_zero']:.8f} |
| Extra coordinates 257–384: MSE | Not predicted | {tail['mse']:.8f} |
| Extra coordinates: SSE / zero-prediction SSE | Not predicted | {tail['relative_sse_to_zero']:.8f} |

The PCA384 full-384 MSE is {metrics['arm384_full_mse']:.8f}. It is recorded for
bookkeeping only: comparing that number directly with PCA256's 256-dimensional
MSE changes the coordinate set and is not the aligned comparison. The tail's
zero-prediction MSE is {metrics['extra_128']['zero_prediction_mse']:.8f}; zero is a
diagnostic reference, not a new model. PCA256's missing 128 coordinates are not
filled with zeros and are not reported as zero error.

There are {sum(r['delta_256_minus_384']<0 for r in rows)}/8 scenes with lower common
error for PCA256 and {sum(r['delta_256_minus_384']>0 for r in rows)}/8 for PCA384.
Per-scene differences and all 384 reported component summaries are preserved in
the CSVs. These request positions are correlated within scenes; they are not
2,304 independent scientific trials.

## Checked execution boundary

- Both real endpoints completed 5,000 updates with 159,992 scene presentations,
  matching the registered spatial recipe except teacher output channel dimension.
- Selection was score-free and fixed before running: eight train scenes from the
  previously fixed 64-scene pilot, with eight named log segments. This is not a
  full-population, dev, navtest or independent generalization sample.
- Same three views, 18×24 native teacher grid, 96 positions per view: 2,304 unique
  requests, {sum(mask)} valid; {a['elements']:,} common and {tail['elements']:,} tail
  scalar elements. Same request IDs and masks were reused across both endpoints.
- Native PCA256 targets are bitwise equal to the PCA384 target prefix; max
  absolute difference is zero. The common train-only PCA basis identity is
  `{result['pca_basis_identity']}`. Original-channel mean/std normalization occurs
  before PCA; projected components are not whitened. Different component energy
  remains, so tail MSE alone does not measure greater or lesser usefulness.
- Both runs used deterministic FP32 eval with TF32 off and unchanged RNG. Each
  ran eight `encode_inputs` and eight sparse readouts. No training forward, Flow
  Matching, action generation, optimizer step or backward occurred. Finite-output
  checks passed at all valid requests. The report reads JSON only, not tensors.
- Runtime/source and input hashes are in SUMMARY.json. The native result is
  COMPLETE; the original supervisor failure is preserved as stated above.
  The failed v1 attempt remains separate and was not relabeled successful or
  included as completed comparison evidence. No rerun or retry is performed.

## What this says about the hypothesis

The aligned common-coordinate comparison directly tests whether removing the
extra 128 supervised output coordinates coincides with better reconstruction of
the retained coordinates after this fixed training budget. Its observed direction
is evidence for this small subset and endpoint only. It does not identify whether
the cause is gradient interference, optimization difficulty, regularization,
finite-budget dynamics or sample variation. No gradient conflict was measured.

The observed common-coordinate difference is only about 0.4%, so this endpoint
provides at most a weak directional signal for easier alignment of the retained
coordinates. The extra 128 coordinates reduce their zero-reference SSE by only
{(1-tail['relative_sse_to_zero'])*100:.3f}% on these requests, compared with
{(1-b['relative_sse_to_zero'])*100:.3f}% for the PCA384 model's common coordinates.
That supports checking whether the low-energy tail is weakly predicted; it does
not establish that its training gradients harm the useful channels. No new
whitening, loss reweighting or smaller-channel training run is implied.

The teacher input resolution stayed 384×288 in both arms; student input, world
queries (48 per view), sparse sample count, loss weights and the 20k learning-rate
time axis were held fixed. This diagnostic cannot show that higher teacher
resolution helps or that one channel dimension improves driving. Such claims
need the separately registered held-out planning evaluation and, for the proposed
mechanism, direct gradient/optimization evidence. Do not substitute these FP32
eval MSEs for the original BF16 training loss, or infer asymptotic convergence
from a single 5k endpoint.

The per-arm elapsed times include strict checkpoint verification/reconstruction
and model loading. They are not per-update training throughput or isolated
readout latency. No new teacher cache or model optimization was performed.
'''
    (HERE/'RESULTS.md').write_text(text)


if __name__ == '__main__':
    main()
