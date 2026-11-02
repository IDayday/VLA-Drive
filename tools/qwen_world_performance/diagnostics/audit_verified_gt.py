"""Finite TRAIN-only logged-GT audit. No model, training or NAVTEST labels."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np

P = Path('/mnt/project/qwen-world-performance-rl-20261103')
SOURCE = P / 'runtime/formal_v1'
BASE = P / 'diagnostics/IL70_TRAIN2048'
DATA = Path('/mnt/project/qwen-world-rl-optimization-20261030/data/NAVTRAIN')
RAW = Path('/mnt/project/DriveDreamer-Policy/navsim_raw/navsim_logs/trainval')
OUT = P / 'analysis/VERIFIED_TRAIN_GT_TEACHER_AUDIT'
sys.path.insert(0, str(SOURCE))
from starVLA.rl.qwen_world_research_data.manifest import (
    atomic_json, file_sha, fingerprint, physical, read_sealed,
)
from starVLA.rl.qwen_world_research_data.data import ResearchSceneData
from starVLA.rl.safe_flow_improvement import v1_reward
from starVLA.rl.simwam_recipe.reward_readouts import RewardService

SCHEMA = 'world_TRAIN_verified_logged_GT_teacher_audit_v1'
NC, DAC, TTC, C, EP, SCORE = ('no_at_fault_collisions', 'drivable_area_compliance',
    'time_to_collision_within_bound', 'comfort', 'ego_progress', 'score')


def now():
    return datetime.now(timezone.utc).isoformat()


def native(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {str(k): native(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [native(x) for x in v]
    return v


def seal(v):
    body = native(v)
    return dict(body, sha256=fingerprint(body))


def typed_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((str(a.dtype), a.shape)).encode() + a.tobytes()).hexdigest()


def strict4(m):
    return all(float(m[k]) == 1.0 for k in (NC, DAC, TTC, C))


def strict_high(m):
    return strict4(m) and float(m[SCORE]) >= .9


def baseline_class(m):
    if float(m[SCORE]) == 0.0:
        return 'zero'
    return 'strict_high' if strict_high(m) else 'other_positive'


def quantiles(xs):
    xs = np.asarray(xs, dtype=np.float64)
    return None if not len(xs) else dict(zip(
        ['min', 'q05', 'q25', 'median', 'q75', 'q95', 'max'],
        np.quantile(xs, [0, .05, .25, .5, .75, .95, 1]).tolist()))


def status(stage, **kwargs):
    atomic_json(OUT / 'STATUS.json', dict(schema=SCHEMA, stage=stage, at=now(), **native(kwargs)))
    print(json.dumps(dict(stage=stage, **native(kwargs)), sort_keys=True), flush=True)


def validate_cache(job):
    """No PDM query: exact source cache plus rear-axle/time origin check."""
    token, log, expected_cache, origin, stamp = job
    cache, ci = v1_reward.load_cache(token, log)
    actual = np.asarray(cache.ego_state.rear_axle.serialize(), dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    error = actual - origin
    error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))
    if ci['sha256'] != expected_cache['sha256']:
        raise ValueError('B1/GT reference cache mismatch: ' + token)
    if cache.ego_state.time_point.time_us != stamp or np.max(np.abs(error)) > 1e-9:
        raise ValueError('Raw rear-axle origin/time differs from scorer cache: ' + token)
    return dict(token=token, cache=ci, timestamp_us=int(stamp),
                max_abs_origin_error=float(np.max(np.abs(error))))


def prepare():
    import torch
    from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec

    identity = read_sealed(BASE / 'identity.json')
    summary = read_sealed(BASE / 'SUMMARY.json')
    if (summary['status'] != 'COMPLETE' or summary['invalid_scenes'] != 0
            or summary['scenes'] != 2048 or summary['expected_scenes'] != 2048
            or identity['cohort'] != 'TRAIN_DIAGNOSTIC' or identity['count'] != 2048):
        raise ValueError('Fixed real TRAIN2048 B1 is not complete')
    protocol = v1_reward.protocol_identity(SOURCE)
    if protocol['sha256'] != identity['protocol']['sha256']:
        raise ValueError('Baseline scorer source/dependencies differ; no scores issued')
    data = ResearchSceneData(DATA, role='NAVTRAIN')
    if data.sha != identity['data_sha256']:
        raise ValueError('Baseline NAVTRAIN data manifest differs')
    pop = json.loads(Path(data.manifest['source_population_path']).read_text())
    tokens = identity['tokens']
    if len(tokens) != 2048 or len(set(tokens)) != 2048 or not set(tokens) <= set(data.tokens):
        raise ValueError('Fixed token census changed')
    forbidden_tokens = set(pop['navtest_tokens'])
    forbidden_logs = {physical(pop['token_logs'][t]) for t in forbidden_tokens}
    if set(tokens) & forbidden_tokens or any(physical(data.logs[t]) in forbidden_logs for t in tokens):
        raise ValueError('TRAIN/NAVTEST physical join failed')
    baseline = {}
    groups = defaultdict(list)
    for token in tokens:
        s = read_sealed(BASE / 'scenes' / (token + '.json'))
        if (s['identity'] != identity['sha256'] or s['status'] != 'OK'
                or s['token'] != token or s['log'] != data.logs[token]
                or s['feedback']['token'] != token or s['feedback']['log'] != s['log']
                or s['feedback']['protocol_sha256'] != protocol['sha256']):
            raise ValueError('B1 scene identity/protocol/token/log mismatch: ' + token)
        if not np.array_equal(np.asarray(s['requested_trajectory'], dtype=np.float32),
                              np.asarray(s['feedback']['requested_trajectory'], dtype=np.float32)):
            raise ValueError('B1 actual requested action differs: ' + token)
        baseline[token] = s
        if data.records[token]['log'] != s['log'] or data.records[token]['token'] != token:
            raise ValueError('GT record token/log mismatch: ' + token)
        groups[s['log']].append(token)

    # Original input manifests pin the exact raw files that generated IL labels.
    raw_hashes, source_inputs = {}, []
    for src in data.manifest['source_records']:
        mp = Path(src['root']) / 'manifest.json'
        if file_sha(mp) != src['manifest_sha256']:
            raise ValueError('IL raw-source manifest changed: ' + str(mp))
        m = json.loads(mp.read_text())
        for log, sha in m['raw_logs_sha256'].items():
            if log in raw_hashes and raw_hashes[log] != sha:
                raise ValueError('Conflicting raw log digests')
            raw_hashes[log] = sha
        source_inputs.append(dict(path=str(mp), file_sha256=file_sha(mp),
            action_contract=m['action_contract'], source_dataclasses_sha256=m['source_dataclasses_sha256']))

    spec = dict(repo=str(SOURCE), maps_root='/mnt/navsim/maps',
                cache_roots=data.manifest['canonical_metric_cache_roots'], protocol_sha256=protocol['sha256'])
    v1_reward.initialize(spec)
    from navsim.common.dataclasses import Scene, SceneMetadata
    rows, raw_inputs = {}, []
    for ordinal, (log, wanted) in enumerate(sorted(groups.items())):
        path = RAW / (log + '.pkl')
        actual_sha = file_sha(path)
        if actual_sha != raw_hashes.get(log):
            raise ValueError('Original TRAIN raw log changed: ' + log)
        with path.open('rb') as stream:
            frames = pickle.load(stream)
        indices = {f['token']: i for i, f in enumerate(frames)}
        raw_inputs.append(dict(log=log, path=str(path), sha256=actual_sha))
        for token in wanted:
            rec = data.records[token]
            i = indices[token]
            future = frames[i:i + 9]
            if len(future) != 9:
                raise ValueError('GT lacks eight genuine future poses: ' + token)
            cur = frames[i]
            scene = Scene(SceneMetadata(log, cur['scene_token'], cur['map_location'], token, 1, 8),
                None, [SimpleNamespace(ego_status=Scene._build_ego_status(f)) for f in future])
            official = scene.get_future_trajectory(8).poses.astype(np.float32)
            requested = data.target(token).numpy()
            if (requested.shape != (8, 3) or not np.isfinite(requested).all()
                    or not np.array_equal(official, requested)):
                raise ValueError('Stored IL GT differs from official raw reconstruction: ' + token)
            stamp = int(cur['timestamp'])
            if stamp != int(rec['timestamp']):
                raise ValueError('Current timestamp differs: ' + token)
            dt = np.diff([float(f['timestamp']) for f in future]) / 1e6
            if not np.array_equal(dt, np.asarray(rec['raw_future_dt_seconds'])):
                raise ValueError('IL raw timestamps differ: ' + token)
            if not np.isfinite(dt).all() or not (dt > 0).all():
                raise ValueError('Nonmonotone raw timestamps: ' + token)
            time_error = np.cumsum(dt) - np.arange(1, 9) * .5
            # The production IL batching mask uses FP32 cumulative times.
            il_time = torch.tensor(rec['raw_future_dt_seconds']).cumsum(0)
            il_mask = ((il_time - torch.arange(1, 9) * .5).abs() < .06).numpy()
            legal = bool(il_mask.all())
            norm = ActionCodec.norm_odo(torch.from_numpy(requested)).numpy()
            roundtrip = ActionCodec.denorm_odo(torch.from_numpy(norm)).numpy()
            baseline_scene = baseline[token]
            rows[token] = dict(token=token, log=log, physical_log=physical(log),
                status='VALID_UNSCORED' if legal else 'INVALID_FIXED_TIME_NO_SCORE',
                eligible_for_scoring=legal, coordinate_reconstruction_bitexact=True,
                raw_log_sha256=actual_sha, raw_frame_index=i, timestamp_us=stamp,
                rear_axle_origin=scene.frames[0].ego_status.ego_pose.tolist(),
                raw_future_dt_seconds=dt.tolist(), fixed_time_error_seconds=time_error.tolist(),
                IL_action_valid_mask=il_mask.tolist(),
                requested_GT=requested.tolist(), requested_GT_typed_sha256=typed_sha(requested),
                clean_normalized=norm.tolist(), normalized_typed_sha256=typed_sha(norm),
                codec_roundtrip=dict(bitexact=bool(np.array_equal(requested, roundtrip)),
                    max_abs_error_by_dimension=np.max(np.abs(roundtrip - requested), axis=0).tolist(),
                    score_applies_to='original_requested_GT_FP32_only',
                    normalized_outside_minus1_plus1=bool((np.abs(norm) > 1).any())),
                baseline=dict(scene_path=str(BASE / 'scenes' / (token + '.json')),
                    scene_sha256=baseline_scene['sha256'], metrics=baseline_scene['feedback']['metrics'],
                    cache=baseline_scene['feedback']['cache'],
                    strict4=strict4(baseline_scene['feedback']['metrics']),
                    strict_high=strict_high(baseline_scene['feedback']['metrics']),
                    group=baseline_class(baseline_scene['feedback']['metrics'])))
        del frames
        if (ordinal + 1) % 64 == 0:
            status('RAW_GT_RECONSTRUCTION', logs_done=ordinal + 1, logs_total=len(groups), tokens_done=len(rows))

    sources = {p: file_sha(SOURCE / p) for p in [
        'scripts/analysis/qwen_sourceflow/prepare_data.py',
        'starVLA/rl/qwen_world_research_data/data.py',
        'starVLA/rl/qwen_world_postcot/batching.py',
        'starVLA/model/modules/qwen_sourceflow/action_codec.py',
        'starVLA/rl/simwam_recipe/reward_readouts.py',
        'scripts/analysis/constrained_flow_noise/batched_canonical_score.py',
        'navsim_v1.1/navsim/navsim/common/dataclasses.py',
        'navsim_v1.1/navsim/navsim/evaluate/pdm_score.py']}
    protocol_doc = seal(dict(schema=SCHEMA, status='GT_COORDINATES_AND_TIMING_VERIFIED',
        created_at=now(), source=str(SOURCE), source_files_sha256=sources,
        audit_script_sha256=file_sha(__file__),
        baseline=dict(path=str(BASE), identity_sha256=identity['sha256'],
            summary_sha256=summary['sha256'], manifest=identity['manifest'],
            tokens_sha256=identity['token_sha256'], count=len(tokens),
            seed=identity['seed'], inference=identity['inference'], model_sha256=identity['model_sha256']),
        navtrain=dict(root=str(DATA), manifest_sha256=data.sha,
            records_sha256=data.manifest['records_sha256'],
            population_sha256=data.manifest['population_sha256'],
            NAVTEST_tokens_and_physical_logs_disjoint=True,
            NAVTEST_trajectories_or_labels_read=False, old_DEV_is_NAVTRAIN=True),
        original_IL_inputs=source_inputs, raw_inputs=raw_inputs,
        GT_contract=dict(source='official Scene.get_future_trajectory(8), FP32 exactly matching IL target',
            frame='rear-axle local x-forward/y-left; absolute relative poses, not increments',
            yaw='relative radians normalized to [-pi,pi]', nominal_seconds=(np.arange(1, 9) * .5).tolist(),
            whole_teacher_time_gate='all 8 original IL action_valid points true: abs(FP32 cumulative raw_dt - nominal)<.06s',
            interpolation=False, clipping=False,
            codec='original physical FP32 requested_GT is scored; inverse codec rounding is separately recorded'),
        scoring_protocol=protocol,
        score_rules=dict(strict4='NC=DAC=TTC=Comfort=1', strict_high='strict4 AND score>=.9',
            useful_margin_small='GT strict_high AND GT_score-B1_score>.005',
            useful_margin_rescue='GT strict_high AND GT_score-B1_score>=.02'),
        budget=dict(max_new_PDM_candidate_queries=2048, candidates_per_scene=1,
            baseline_reused_queries=2048, baseline_rescored_queries=0, CPU_only=True,
            validation_is_not_PDM_scoring=True, no_training=True, no_new_GPU=True)))
    atomic_json(OUT / 'PROTOCOL.json', protocol_doc)
    atomic_json(OUT / 'PREPARED.json', seal(dict(schema=SCHEMA, tokens=tokens, rows=rows,
        protocol_sha256=protocol_doc['sha256'])))
    np.savez_compressed(OUT / 'TRAJECTORIES.npz', tokens=np.array(tokens),
        requested_GT_FP32=np.array([rows[t]['requested_GT'] for t in tokens], dtype=np.float32),
        clean_normalized=np.array([rows[t]['clean_normalized'] for t in tokens], dtype=np.float32),
        whole_trajectory_valid=np.array([rows[t]['eligible_for_scoring'] for t in tokens]))
    return tokens, rows, spec, protocol_doc


def aggregate(tokens, rows, protocol, *, stage):
    scored = [r for r in rows.values() if r['status'] == 'SCORED']
    result = dict(schema=SCHEMA, stage=stage, at=now(), protocol_sha256=protocol['sha256'],
        input_scenes=len(tokens), scored_scenes=len(scored),
        unscored=Counter(r['status'] for r in rows.values() if r['status'] != 'SCORED'),
        complete_input_census=True, no_training=True, new_GPU_queries=0,
        actual_new_CPU_PDM_queries=len(scored), max_new_CPU_PDM_queries=2048,
        cohort_statistics={})
    groups = {'all': scored, 'B1_zero': [r for r in scored if r['baseline']['group'] == 'zero'],
        'B1_other_positive': [r for r in scored if r['baseline']['group'] == 'other_positive'],
        'B1_strict_high': [r for r in scored if r['baseline']['strict_high']],
        'B1_strict4': [r for r in scored if r['baseline']['strict4']],
        'B1_not_strict4': [r for r in scored if not r['baseline']['strict4']]}
    for name, rr in groups.items():
        result['cohort_statistics'][name] = dict(count=len(rr),
            GT_strict4=sum(r['GT_strict4'] for r in rr), GT_strict_high=sum(r['GT_strict_high'] for r in rr),
            GT_zero=sum(r['GT_metrics'][SCORE] == 0.0 for r in rr),
            verified_strict_high_gain_gt005=sum(r['verified_teacher_gt005'] for r in rr),
            verified_strict_high_gain_ge02=sum(r['verified_teacher_ge02'] for r in rr),
            GT_score_better_gt005=sum(r['gain'] > .005 for r in rr),
            GT_score_worse_lt_minus005=sum(r['gain'] < -.005 for r in rr),
            GT_score_tie_within005=sum(abs(r['gain']) <= .005 for r in rr),
            B1_mean_score=float(np.mean([r['baseline']['metrics'][SCORE] for r in rr])) if rr else None,
            GT_mean_score=float(np.mean([r['GT_metrics'][SCORE] for r in rr])) if rr else None,
            gain_distribution=quantiles([r['gain'] for r in rr]),
            GT_score_distribution=quantiles([r['GT_metrics'][SCORE] for r in rr]),
            physical_logs=len({r['physical_log'] for r in rr}))
    result['per_metric_mean_delta'] = {k: float(np.mean([r['GT_metrics'][k] - r['baseline']['metrics'][k]
        for r in scored])) if scored else None for k in (NC, DAC, TTC, C, EP, SCORE)}
    result['codec_roundtrip'] = dict(bitexact_count=sum(r['codec_roundtrip']['bitexact'] for r in rows.values()),
        total=len(rows), max_abs_error_by_dimension=np.max([
            r['codec_roundtrip']['max_abs_error_by_dimension'] for r in rows.values()], axis=0).tolist(),
        future_normalized_teacher_must_revalidate_if_request_bits_change=True)
    result['zero_cases'] = [dict(token=r['token'], log=r['log'], status=r['status'],
        GT_metrics=r.get('GT_metrics'), gain=r.get('gain'),
        verified_teacher_gt005=r.get('verified_teacher_gt005'), verified_teacher_ge02=r.get('verified_teacher_ge02'))
        for r in rows.values() if r['baseline']['group'] == 'zero']
    result['limitations'] = [
        'TRAIN2048 diagnostic only; no NAVTEST trajectory/label/teacher; not new generalization score.',
        'Logged human GT is privileged offline TRAIN supervision, not an oracle or actor input.',
        'Scores use fixed official reference plus exactly one GT candidate per scene; reference never replaced.',
        'Raw timing-invalid whole trajectories are unscored, not zero rewards; no interpolation.',
        'GT has no on-policy log probability/advantage and is not eligible for PPO credit.',
        'A good GT alongside a bad B1 alone does not prove G16 has no successful candidate.',
        'No GT teacher has been admitted into any bank, S6, registered algorithm or current run.',
        'FP32 norm/denorm can change requested bits; this score certifies original physical FP32 GT only.']
    return seal(result)


def run(workers):
    if not 1 <= workers <= 8:
        raise ValueError('CPU workers must be 1..8')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'scenes').mkdir(exist_ok=True)
    with (OUT / 'audit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUT / 'BUDGET_LEDGER.json').exists():
            raise RuntimeError('Existing ledger retained: no automatic duplicate scoring/retry')
        status('PREPARING', pid=os.getpid(), CPU_workers=workers, GPU=False)
        tokens, rows, spec, protocol = prepare()
        legal = [t for t in tokens if rows[t]['eligible_for_scoring']]
        zero = [t for t in tokens if rows[t]['baseline']['group'] == 'zero']
        controls = sorted([t for t in tokens if rows[t]['baseline']['strict_high']],
                          key=lambda t: fingerprint(['TRAIN_GT_pilot', 42, t]))[:16]
        pilot = list(dict.fromkeys(zero + controls))
        ledger = dict(schema=SCHEMA, created_at=now(), workers=workers, hard_query_cap=2048,
            baseline_PDM_queries_reused=2048, baseline_new_queries=0, issued={}, completed={}, failures={})
        atomic_json(OUT / 'BUDGET_LEDGER.json', ledger)
        service = RewardService(SOURCE, workers=workers, cache_roots=spec['cache_roots'])
        try:
            # Validate origin/cache for the complete legal input before scoring any GT.
            status('CACHE_ORIGIN_PREFLIGHT', legal_scenes=len(legal), invalid_time_scenes=len(tokens) - len(legal))
            futures = {service.pool.submit(validate_cache, (t, rows[t]['log'], rows[t]['baseline']['cache'],
                rows[t]['rear_axle_origin'], rows[t]['timestamp_us'])): t for t in legal}
            for future in as_completed(futures):
                t = futures[future]
                rows[t]['cache_origin_validation'] = future.result()
            atomic_json(OUT / 'ORIGIN_VALIDATION.json', seal(dict(schema=SCHEMA, count=len(legal),
                all_pass=True, details={t: rows[t]['cache_origin_validation'] for t in legal})))
            for stage, selected in [('PILOT', [t for t in pilot if t in legal]),
                                    ('FULL_TRAIN2048', [t for t in legal if t not in pilot])]:
                status(stage + '_SCORING', new_scene_queries=len(selected), prior_completed=len(ledger['completed']))
                futures = {}
                for t in selected:
                    if t in ledger['issued'] or len(ledger['issued']) >= 2048:
                        raise RuntimeError('Duplicate or over-budget query refused')
                    r = rows[t]
                    ledger['issued'][t] = dict(stage=stage, at=now(), candidate_count=1,
                        requested_sha256=r['requested_GT_typed_sha256'])
                    atomic_json(OUT / 'BUDGET_LEDGER.json', ledger)
                    futures[service.submit((t, r['log'], np.asarray(r['requested_GT'], dtype=np.float32)[None]))] = t
                for future in as_completed(futures):
                    t = futures[future]
                    feedback = future.result()
                    if len(feedback) != 1:
                        raise ValueError('Exactly one GT request/record expected')
                    f = feedback[0]
                    r = rows[t]
                    if (f['status'] != 'OK' or f['token'] != t or f['log'] != r['log']
                            or f['protocol_sha256'] != spec['protocol_sha256']
                            or f['cache']['sha256'] != r['baseline']['cache']['sha256']
                            or typed_sha(f['requested_trajectory']) != r['requested_GT_typed_sha256']):
                        raise ValueError('Actual scored GT provenance differs: ' + t)
                    m = f['metrics']
                    r.update(status='SCORED', GT_metrics=m, GT_strict4=strict4(m),
                        GT_strict_high=strict_high(m), gain=float(m[SCORE] - r['baseline']['metrics'][SCORE]))
                    r['verified_teacher_gt005'] = bool(r['GT_strict_high'] and r['gain'] > .005)
                    r['verified_teacher_ge02'] = bool(r['GT_strict_high'] and r['gain'] >= .02)
                    atomic_json(OUT / 'scenes' / (t + '.json'), seal(dict(r, feedback=f)))
                    ledger['completed'][t] = dict(at=now(), score=float(m[SCORE]), gain=r['gain'], stage=stage)
                    atomic_json(OUT / 'BUDGET_LEDGER.json', ledger)
                atomic_json(OUT / (stage + '_SUMMARY.json'), aggregate(tokens, rows, protocol, stage=stage))
                status(stage + '_COMPLETE', completed_queries=len(ledger['completed']))
        except Exception as exc:
            ledger['failure'] = dict(at=now(), type=type(exc).__name__, message=str(exc))
            atomic_json(OUT / 'BUDGET_LEDGER.json', ledger)
            status('FAILED_NO_FAKE_ZERO', exception_type=type(exc).__name__, message=str(exc))
            raise
        finally:
            service.close()
        for t in tokens:
            if rows[t]['status'] != 'SCORED':
                atomic_json(OUT / 'scenes' / (t + '.json'), seal(rows[t]))
        result = aggregate(tokens, rows, protocol, stage='COMPLETE')
        result['status'] = 'COMPLETE_ALL2048_INPUTS_ACCOUNTED'
        result['scored_input_count'] = len(ledger['completed'])
        result.pop('sha256')
        atomic_json(OUT / 'SUMMARY.json', seal(result))
        with (OUT / 'SCENES.jsonl').open('w') as stream:
            for t in tokens:
                stream.write(json.dumps(native(rows[t]), sort_keys=True, allow_nan=False) + '\n')
        atomic_json(OUT / 'COMPLETE.json', seal(dict(schema=SCHEMA, at=now(),
            status='COMPLETE', input_count=2048, scored_count=len(ledger['completed']),
            new_CPU_PDM_queries=len(ledger['issued']), maximum=2048,
            files_sha256={name: file_sha(OUT / name) for name in (
                'SUMMARY.json', 'PROTOCOL.json', 'BUDGET_LEDGER.json', 'ORIGIN_VALIDATION.json',
                'TRAJECTORIES.npz', 'SCENES.jsonl')})))
        status('COMPLETE', actual_new_CPU_PDM_queries=len(ledger['completed']), inputs=2048)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    run(args.workers)
