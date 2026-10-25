"""Build/resume bounded TRAIN/DEV candidate shards with real canonical PDM.

Each external --rank/--world owns disjoint scenes and one CPU scoring process.
The parent scheduler controls parallelism and the finite host lease. No VLM or
student hidden state is loaded, and no scoring runs inside an IL update.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import pickle
import sqlite3
import sys
import time
import traceback

import numpy as np

from starVLA.model.modules.qwen_world_postcot.candidate_bank import (
    SCHEMA, CODEC_VERSION, COORDINATE_VERSION, CandidateBank, CandidateBankWriter,
    CandidateRecord, atomic_json, file_sha, fingerprint, merge_bank_parts,
    observation_fingerprint, trajectory_hash,
)


def motion_check(trajectory, *, max_speed=55., max_acceleration=12., max_jerk=35.) -> dict:
    """Generous numerical/kinematic bounds, not a collision or safety filter."""
    trajectory = np.asarray(trajectory, dtype=np.float32)
    if trajectory.shape != (8, 3) or not np.isfinite(trajectory).all():
        return dict(valid=False, reason='nonfinite_or_wrong_shape')
    points = np.concatenate([np.zeros((1, 2)), trajectory[:, :2]], axis=0)
    velocity = np.diff(points, axis=0)/.5
    acceleration = np.diff(velocity, axis=0)/.5
    jerk = np.diff(acceleration, axis=0)/.5
    speed = float(np.linalg.norm(velocity, axis=1).max())
    accel = float(np.linalg.norm(acceleration, axis=1).max())
    peak_jerk = float(np.linalg.norm(jerk, axis=1).max())
    heading_step = float(np.abs(np.diff(np.unwrap(np.r_[0., trajectory[:, 2]]))).max())
    valid = speed <= max_speed and accel <= max_acceleration and peak_jerk <= max_jerk and heading_step <= 1.5
    return dict(valid=bool(valid), maximum_speed_mps=speed,
                maximum_acceleration_mps2=accel, maximum_jerk_mps3=peak_jerk,
                maximum_heading_step_radians=heading_step,
                safety_not_checked_by_this_filter=True)


def coherent_proposals(gt, count: int = 16, current_velocity=None):
    """One exact GT plus deterministic smooth speed/brake/lateral variants.

    Changes are temporally coherent, finite, low-dimensional interventions.
    Their safety and comfort are not presumed: every candidate is simulated.
    No random independent-waypoint perturbations or artificial metric labels.
    """
    gt = np.asarray(gt, dtype=np.float32)
    trajectory_hash(gt)
    timepoints = np.arange(1, 9, dtype=np.float64)*.5
    u = timepoints/4.
    smooth = u*u*(3.-2.*u)
    path_time = np.r_[0., timepoints]
    path = np.concatenate([np.zeros((1, 3)), gt.astype(np.float64)], axis=0)
    path[:, 2] = np.unwrap(path[:, 2])
    values = [dict(trajectory_fp32=gt.tolist(), source='logged_gt',
                   parameters=dict(kind='unaltered_logged_target'))]
    for delay in [.05, .10, .20, .35, .50, .75, 1.0]:
        # q'(0)=1: braking changes ramp in instead of abruptly changing t0.
        warped = timepoints-delay*smooth
        poses = np.stack([np.interp(warped, path_time, path[:, d]) for d in range(3)], axis=1)
        values.append(dict(trajectory_fp32=poses.astype(np.float32).tolist(), source='coherent_variant',
                           parameters=dict(kind='smooth_brake_time_warp', final_delay_seconds=delay)))
    for amount in [-.8, -.5, -.3, -.15, .15, .3, .5, .8]:
        poses = gt.astype(np.float64).copy()
        lateral = amount*smooth
        poses[:, 0] -= np.sin(poses[:, 2])*lateral
        poses[:, 1] += np.cos(poses[:, 2])*lateral
        # A small path-tangent correction. Near standstill, omit this family
        # rather than invent a right-angle steering maneuver from no motion.
        ds = np.linalg.norm(np.diff(np.r_[np.zeros((1, 2)), poses[:, :2]], axis=0), axis=1)
        if np.mean(ds) < .1:
            continue
        correction = np.arctan2(np.gradient(lateral), np.maximum(ds, .1))
        poses[:, 2] += correction
        poses[:, 2] = (poses[:, 2]+np.pi)%(2*np.pi)-np.pi
        values.append(dict(trajectory_fp32=poses.astype(np.float32).tolist(), source='coherent_variant',
                           parameters=dict(kind='smooth_lateral_shift', final_offset_meters=amount)))
    # A controlled progression family also provides independent executable
    # candidates for stationary scenes, where time warps are exact duplicates.
    for advance in [.05, .10, .15, .2, .3, .4, .6, .8, 1., 1.3, 1.6, 2., 2.5, 3., 4., 5.]:
        poses = gt.astype(np.float64).copy()
        distance = advance*u*u*u
        poses[:, 0] += np.cos(gt[:, 2])*distance
        poses[:, 1] += np.sin(gt[:, 2])*distance
        values.append(dict(trajectory_fp32=poses.astype(np.float32).tolist(), source='coherent_variant',
                           parameters=dict(kind='smooth_longitudinal_progression', final_advance_meters=advance)))
    # When a logged path is numerically rough, do not erase the scene. A
    # bounded family of gentle braking/curvature plans from the observed
    # forward velocity supplies real alternatives for actual simulation.
    # This is not a safety teacher and never creates presumed good labels.
    if current_velocity is not None:
        velocity = np.asarray(current_velocity, dtype=np.float64)
        if velocity.shape != (2,) or not np.isfinite(velocity).all():
            raise ValueError('Current velocity must be the observed ego xy velocity in m/s')
        initial = float(velocity[0])
        dt = .05
        for deceleration in [0., 1., 2., 3., 4.]:
            for curvature in [0., -.005, .005, -.01, .01]:
                x = y = yaw = 0.; history = []
                for step in range(80):
                    speed = np.sign(initial)*max(abs(initial)-deceleration*(step+.5)*dt, 0.)
                    turn = speed*curvature*dt
                    x += speed*np.cos(yaw+.5*turn)*dt
                    y += speed*np.sin(yaw+.5*turn)*dt
                    yaw += turn
                    if (step+1) % 10 == 0:
                        history.append([x,y,yaw])
                values.append(dict(trajectory_fp32=np.asarray(history,dtype=np.float32).tolist(),
                    source='coherent_variant',parameters=dict(kind='observed_forward_velocity_brake_curvature',
                        initial_forward_velocity_mps=initial,deceleration_mps2=deceleration,
                        curvature_per_meter=curvature,lateral_velocity_not_modeled=True,
                        safety_is_only_determined_by_actual_official_simulation=True)))
    valid, seen, rejected = [], set(), []
    for value in values:
        poses = np.asarray(value['trajectory_fp32'], dtype=np.float32)
        check = motion_check(poses)
        key = trajectory_hash(poses)
        if key in seen:
            continue
        seen.add(key)
        value['dynamics_check'] = check
        # Logged/own-policy trajectories are observations of actual behavior,
        # including bad behavior. A generous intervention-construction filter
        # must not erase difficult original scenes or silently remove GT.
        if not check['valid'] and value['source'] == 'coherent_variant':
            rejected.append(value)
            continue
        valid.append(value)
        if len(valid) == count:
            break
    return valid, rejected


def load_raw(data_root: Path, split: str):
    manifest = json.loads((data_root/'manifest.json').read_text())
    seal = json.loads((data_root/'COMPLETE').read_text())
    if manifest['cohort'] != split or seal['manifest_sha256'] != file_sha(data_root/'manifest.json'):
        raise ValueError('Wrong or unsealed raw data role')
    if file_sha(data_root/'records.pkl') != manifest['records_sha256']:
        raise ValueError('Raw current-observation/GT records changed')
    with (data_root/'records.pkl').open('rb') as stream:
        rows = pickle.load(stream)
    if set(rows) != set(manifest['tokens']):
        raise ValueError('Raw manifest and current-scene records disagree')
    return manifest, rows


def logged_candidate_provenance(row):
    """Logged poses are exact 0.5s evaluator requests, not presumed valid GT."""
    times = row.get('raw_future_dt_seconds')
    return dict(raw_future_dt_seconds=None if times is None else list(times),
        timestamp_gap=bool(row.get('timestamp_gap', False)),
        source_semantics='logged_pose_candidate_time_mismatch' if row.get('timestamp_gap')
            else 'logged_pose_candidate_timing_unknown' if times is None
            else 'logged_pose_candidate_no_recorded_time_gap',
        actual_official_request_times_seconds=[.5*index for index in range(1,9)],
        candidate_scoring_does_not_certify_imitation_target_timing=True)


def _identity(args, manifest, tokens, protocol):
    return dict(split=args.split, source_data_manifest_sha256=file_sha(Path(args.data_root)/'manifest.json'),
        source_records_sha256=manifest['records_sha256'], population_sha256=manifest['population_sha256'],
        scene_manifest_sha256=fingerprint(tokens), source_token_order_sha256=fingerprint(manifest['tokens']),
        coordinate_version=COORDINATE_VERSION, normalization_version=CODEC_VERSION,
        score_protocol_sha256=protocol['sha256'], metric_version=protocol['name'],
        generator_version='smooth_temporal_interventions_v3_timing_provenance', target_candidates=args.target_candidates,
        selection_seed=args.seed, extra_candidates_index_sha256=None if not args.extra_candidates_index else file_sha(args.extra_candidates_index),
        reused_bank_manifest_sha256=[file_sha(Path(path)/'MANIFEST.json') for path in args.reuse_bank])


def reusable_feedback(record, *, observation_hash, token, log, protocol, context):
    """Reuse a scored exact request only under the same current scene/function.

    Physical deduplication does not relax data-role, observation, evaluator or
    actual metric-context identity. False means explicitly resimulate instead.
    """
    if (record.scene_id != token or record.source_log != log
            or record.observation_hash != observation_hash
            or record.score_protocol_sha256 != protocol['sha256']
            or record.context_sha256 != context['sha256']
            or not all(record.valid_masks.values())):
        return None
    return dict(token=token, log=log, status='OK', protocol=record.metric_version,
        protocol_sha256=record.score_protocol_sha256, cache=context,
        metrics=dict(record.raw_factor_labels, score=record.raw_total_score),
        requested_trajectory=record.trajectory_fp32)


def _extra_candidates(index, token):
    if index is None:
        return []
    records = []
    for (payload,) in index.execute('SELECT payload FROM candidates WHERE token=? ORDER BY record_id', (token,)):
        records.append(json.loads(payload))
    return records


def build(args):
    from starVLA.rl.qwen_world_postcot.contracts import require_dispatch, boundary_stop_due
    from starVLA.rl.safe_flow_improvement import v1_reward
    from starVLA.rl.simwam_recipe.reward_readouts import RewardService
    require_dispatch()
    data_root, output = Path(args.data_root), Path(args.output)
    manifest, rows = load_raw(data_root, args.split)
    population = json.loads(Path('/mnt/project/flow-safe-improvement-20261013/population/manifest.json').read_text())
    expected_role = population['train_tokens' if args.split == 'TRAIN' else 'dev_tokens']
    if not set(manifest['tokens']) <= set(expected_role) or set(manifest['tokens']) & set(population['navtest_tokens']):
        raise ValueError('Candidate fitting cannot contain NAVTEST or a changed role')
    if args.split == 'TRAIN':
        physical = lambda log: log.rsplit('_', 2)[0]
        train_logs = {physical(population['token_logs'][token]) for token in manifest['tokens']}
        excluded_logs = {physical(population['token_logs'][token])
                         for token in population['dev_tokens']+population['navtest_tokens']}
        if train_logs & excluded_logs:
            raise ValueError('TRAIN must embargo whole physical logs present in fixed DEV/NAVTEST')
    tokens = list(manifest['tokens'])
    if args.scene_manifest:
        chosen = json.loads(Path(args.scene_manifest).read_text())
        selected = chosen['tokens'] if isinstance(chosen, dict) else chosen
        if len(set(selected)) != len(selected) or not set(selected) <= set(tokens):
            raise ValueError('Fixed candidate scene list crosses its legal data role')
        tokens = list(selected)
    if not args.all_scenes:
        if not args.scene_manifest:
            # Freeze the pilot before reading scores and spread it over logs
            # instead of timing only the first contiguous segment directory.
            tokens = sorted(tokens, key=lambda token: fingerprint([args.seed, token]))
        tokens = tokens[:args.max_scenes]
    protocol = v1_reward.protocol_identity(Path.cwd())
    identity = _identity(args, manifest, tokens, protocol)
    assigned = tokens[args.rank::args.world]
    part = output/'parts'/f'rank{args.rank:05d}'
    if not assigned:
        raise ValueError('Empty shard; use fewer producers than requested scenes')
    if (part/'COMPLETE.json').exists():
        existing = CandidateBank(part, allowed_splits=(args.split,), expected_identity=identity)
        try:
            if existing.manifest['scene_ids'] != assigned:
                raise ValueError('Complete part has a different sampler population')
            print(json.dumps(dict(status='REUSED_COMPLETE_PART', rank=args.rank,
                                  bank_hash=existing.bank_hash, scenes=len(assigned))))
            return 0
        finally:
            existing.close()
    part.mkdir(parents=True, exist_ok=True)
    atomic_json(part/'RESOLVED_BUILD.json', dict(arguments=vars(args), identity=identity,
        assigned_scenes=assigned, scoring_workers_per_producer=1,
        concurrency='external scene shards; one official CPU scorer per shard'))
    revision = dict(name='NAVSIM_v1.1', git_commit=None,
        source_tree_sha256=fingerprint(protocol['sources']),
        adapter_source_sha256=file_sha(Path('scripts/analysis/constrained_flow_noise/batched_canonical_score.py')),
        dependencies=protocol['dependencies'])
    index = None if not args.extra_candidates_index else sqlite3.connect(f'file:{Path(args.extra_candidates_index).resolve()}?mode=ro&immutable=1', uri=True)
    if index is not None:
        index_receipt = json.loads(Path(args.extra_candidates_index).with_suffix('.COMPLETE.json').read_text())
        if index_receipt['status'] != 'COMPLETE' or index_receipt['index_sha256'] != file_sha(args.extra_candidates_index):
            raise ValueError('Native candidate index is incomplete or its bytes changed')
        stored = json.loads(index.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()[0])
        if (stored['score_protocol_sha256'] != protocol['sha256'] or stored['split'] != args.split
                or stored['source_data_manifest_sha256'] != identity['source_data_manifest_sha256']):
            raise ValueError('Imported own-policy labels have incompatible metric/data identity')
    reuse_banks = [CandidateBank(path, allowed_splits=(args.split,)) for path in args.reuse_bank]
    reusable_scene_sets = [set(bank.manifest['scene_ids']) for bank in reuse_banks]
    required_identity = ('split', 'source_data_manifest_sha256', 'source_records_sha256',
        'population_sha256', 'coordinate_version', 'normalization_version', 'score_protocol_sha256')
    for bank in reuse_banks:
        if any(bank.manifest['identity'].get(key) != identity[key] for key in required_identity):
            raise ValueError('Reuse bank has different data/observation/metric identity')
    started = time.monotonic(); scored = reused = bank_reused = completed = submitted = 0
    source_counts, scene_candidate_counts = Counter(), Counter()
    for attempt_file in (part/'scoring_attempts').glob('*/*.json'):
        attempt = json.loads(attempt_file.read_text())
        submitted += len(attempt.get('requested_hashes', []))
        if attempt.get('status') == 'VALID_SCORED':
            scored += len(attempt.get('requested_hashes', []))
    service = RewardService(Path.cwd(), workers=1)
    try:
        with CandidateBankWriter(part, identity, assigned, minimum_candidates=2,
                maximum_candidates=args.target_candidates, shard_bytes=args.shard_mib*1024**2,
                resume=args.resume) as writer:
            if args.resume:
                # Recover accounting from committed records, not from a
                # possibly interrupted progress JSON or prefetched requests.
                for shard, offset, length in writer.db.execute('SELECT shard,offset,length FROM records ORDER BY shard,offset'):
                    with (part/shard).open('rb') as stream:
                        stream.seek(offset); value = json.loads(stream.read(length))
                    source_counts[value['candidate_source']] += 1
                    scene_candidate_counts[value['scene_id']] += 1
            for token in assigned:
                if boundary_stop_due():
                    atomic_json(part/'PAUSED.json', dict(status='OPERATOR_PAUSED', reason='task host deadline/withdrawal',
                        completed_scenes=completed, actual_new_scores=scored, same_producer_resume_required=True))
                    return 75
                require_dispatch()
                if writer.scene_count(token) >= args.target_candidates:
                    completed += 1
                    continue
                row = rows[token]
                images = {name: None if path is None else file_sha(path) for name, path in row['images'].items()}
                observation = observation_fingerprint(row, identity['source_data_manifest_sha256'], images)
                proposals_path = part/'proposals'/f'{token}.json'
                existing = _extra_candidates(index, token)
                extras = [dict(trajectory_fp32=r['requested_trajectory'], source=r.get('candidate_source', 'native_qwen_rl'),
                               actor_checkpoint_sha256=r['actor_checkpoint_sha256'], sampling=r['sampling'],
                               parameters=r['source_parameters'], cached_feedback=r['feedback']) for r in existing[:8]]
                generated, rejected = coherent_proposals(row['requested_GT'], args.target_candidates+len(extras),
                                                        current_velocity=row['proprio'][:2])
                generated[0]['parameters'].update(logged_candidate_provenance(row))
                # Always retain logged GT; existing trained-policy samples take
                # remaining priority, then fill with coherent scored variants.
                ordered = generated[:1]+extras+generated[1:]
                proposals, seen = [], set()
                for value in ordered:
                    poses = np.asarray(value['trajectory_fp32'], dtype=np.float32)
                    key = trajectory_hash(poses)
                    if key in seen or (value['source'] == 'coherent_variant' and not motion_check(poses)['valid']):
                        continue
                    seen.add(key); proposals.append(value)
                    if len(proposals) == args.target_candidates:
                        break
                if len(proposals) < 2:
                    atomic_json(part/'insufficient'/f'{token}.json', dict(status='INSUFFICIENT_VALID_CANDIDATES',
                        token=token, generated=len(proposals), rejected=rejected))
                    raise ValueError('Fewer than two real valid candidates; cannot fabricate coverage')
                proposal_record = dict(scene_id=token, observation_hash=observation, proposals=proposals,
                                       rejected_variants=rejected, schema='world_postcot_fixed_requests_v1')
                if proposals_path.exists():
                    if json.loads(proposals_path.read_text()) != proposal_record:
                        raise ValueError('Scene prediction proposals changed during scoring retry/resume')
                else:
                    atomic_json(proposals_path, proposal_record)
                feedback, pending, positions = [None]*len(proposals), [], []
                old_bank_records = {}
                for old_bank, scenes in zip(reuse_banks, reusable_scene_sets):
                    if token in scenes:
                        for old_record in old_bank.records(token, observation_hash=observation):
                            previous = old_bank_records.get(old_record.trajectory_sha256)
                            if previous is not None and (previous.raw_factor_labels != old_record.raw_factor_labels
                                    or previous.context_sha256 != old_record.context_sha256):
                                raise ValueError('Reusable banks disagree on identical request labels')
                            old_bank_records[old_record.trajectory_sha256] = old_record
                current_context = None
                if old_bank_records:
                    from starVLA.rl.simwam_recipe.contracts import CACHE_ROOTS
                    for root in CACHE_ROOTS:
                        path = Path(root)/row['log']/'unknown'/token/'metric_cache.pkl'
                        if path.is_file():
                            current_context = dict(path=str(path), sha256=file_sha(path)); break
                    if current_context is None:
                        raise FileNotFoundError('Cannot validate reused labels without their real current metric context')
                for i, value in enumerate(proposals):
                    old_record = old_bank_records.get(trajectory_hash(value['trajectory_fp32']))
                    if old_record is not None:
                        old = reusable_feedback(old_record, observation_hash=observation, token=token,
                            log=row['log'], protocol=protocol, context=current_context)
                        if old is not None:
                            feedback[i] = old; reused += 1; bank_reused += 1; continue
                    old = value.get('cached_feedback')
                    if old is not None:
                        cache = Path(old['cache']['path'])
                        if (old['protocol_sha256'] == protocol['sha256'] and cache.is_file()
                                and file_sha(cache) == old['cache']['sha256']):
                            feedback[i] = old; reused += 1; continue
                    positions.append(i); pending.append(value['trajectory_fp32'])
                if submitted + len(pending) > args.max_simulation_budget:
                    atomic_json(part/'PAUSED.json', dict(status='OPERATOR_PAUSED', reason='explicit simulation budget',
                        actual_new_scores=scored, scoring_requested_candidates=submitted,
                        next_scene=token, fixed_requests_preserved=True))
                    return 75
                if pending:
                    job = token, row['log'], np.asarray(pending, dtype=np.float32)
                    records = None
                    for attempt in range(args.scoring_retries+1):
                        if submitted + len(pending) > args.max_simulation_budget:
                            raise RuntimeError('Explicit simulation-attempt budget exhausted; preserve retryable scene')
                        attempt_path = part/'scoring_attempts'/token/f'attempt_{time.time_ns()}.json'
                        begin = time.monotonic()
                        submitted += len(pending)
                        try:
                            records = service.score(job)
                            if len(records) != len(pending):
                                raise ValueError('Official scorer returned incomplete candidate labels')
                            atomic_json(attempt_path, dict(status='VALID_SCORED', token=token,
                                requested_hashes=[trajectory_hash(v) for v in job[2]],
                                seconds=time.monotonic()-begin))
                            scored += len(pending)
                            break
                        except (OSError, TimeoutError) as error:
                            atomic_json(attempt_path, dict(status='TRANSIENT_SCORING_FAILURE', token=token,
                                error=repr(error), seconds=time.monotonic()-begin,
                                requested_hashes=[trajectory_hash(v) for v in job[2]]))
                            if attempt == args.scoring_retries:
                                raise
                    for i, record in zip(positions, records):
                        if trajectory_hash(record['requested_trajectory']) != trajectory_hash(proposals[i]['trajectory_fp32']):
                            raise ValueError('Official score is not attached to the exact replayed trajectory')
                        feedback[i] = record
                candidates = [CandidateRecord.from_feedback(row=row, observation_hash=observation,
                    feedback=record, evaluator_revision=revision, source=value['source'], split=args.split,
                    actor_checkpoint_sha256=value.get('actor_checkpoint_sha256'),
                    sampling=value.get('sampling', {}), source_parameters=value.get('parameters', {}))
                    for value, record in zip(proposals, feedback)]
                previous_requests = {value[0] for value in writer.db.execute(
                    'SELECT trajectory_hash FROM records WHERE scene_id=?', (token,))}
                writer.append_many(candidates)
                for candidate in candidates:
                    if candidate.trajectory_sha256 not in previous_requests:
                        source_counts[candidate.candidate_source] += 1
                        scene_candidate_counts[candidate.scene_id] += 1
                completed += 1
                atomic_json(part/'PROGRESS.json', dict(status='BUILDING', completed_scenes=completed,
                    assigned_scenes=len(assigned), actual_new_scores=scored, reused_scores=reused,
                    reused_bank_scores=bank_reused,
                    source_counts=dict(source_counts),
                    scoring_requested_candidates=submitted, seconds=time.monotonic()-started,
                    rank=args.rank, world=args.world))
            result = writer.finalize()
        atomic_json(part/'BUILD_COST.json', dict(status='COMPLETE', scenes=len(assigned),
            candidates=result['record_count'], actual_new_scores=scored, reused_scores=reused,
            reused_bank_scores=bank_reused,
            source_counts=dict(source_counts),
            candidates_per_scene_histogram=dict(Counter(scene_candidate_counts.values())),
            scoring_requested_candidates=submitted, seconds=time.monotonic()-started, scoring_workers=1))
        if args.world == 1:
            merge_bank_parts(output, [part], identity=identity, scene_ids=tokens)
        print(json.dumps(dict(status='COMPLETE_PART', rank=args.rank, scenes=len(assigned),
                              candidates=result['record_count'], seconds=time.monotonic()-started)))
        return 0
    finally:
        service.close()
        if index is not None:
            index.close()
        for old_bank in reuse_banks:
            old_bank.close()


def finalize(args):
    output = Path(args.output)
    parts = sorted((output/'parts').glob('rank*'))
    if len(parts) != args.world or any(not (part/'COMPLETE.json').exists() for part in parts):
        raise ValueError('Not all declared producer parts are complete')
    records = [json.loads((part/'BUILD_IDENTITY.json').read_text()) for part in parts]
    identity = records[0]['identity']
    if any(value['identity'] != identity for value in records):
        raise ValueError('Producer data/protocol identities differ')
    # Recover original interleaved sampler order, not rank-concatenation order.
    tokens = []
    longest = max(len(value['scene_ids']) for value in records)
    for offset in range(longest):
        for value in records:
            if offset < len(value['scene_ids']):
                tokens.append(value['scene_ids'][offset])
    if fingerprint(tokens) != identity['scene_manifest_sha256']:
        raise ValueError('Producer scene lists do not reconstruct the frozen population')
    manifest = merge_bank_parts(output, parts, identity=identity, scene_ids=tokens)
    print(json.dumps(dict(status='COMPLETE', scenes=manifest['scene_count'],
                          candidates=manifest['record_count'], bank_hash=manifest['sha256'])))
    return 0


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data-root', default='/mnt/project/qwen-world-postcot-20261025/data/TRAIN')
    parser.add_argument('--split', choices=['TRAIN', 'DEV'], default='TRAIN')
    parser.add_argument('--output', required=True)
    parser.add_argument('--scene-manifest')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world', type=int, default=1)
    parser.add_argument('--max-scenes', type=int, default=256)
    parser.add_argument('--all-scenes', action='store_true')
    parser.add_argument('--target-candidates', type=int, default=16)
    parser.add_argument('--max-simulation-budget', type=int, default=4096)
    parser.add_argument('--scoring-retries', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--shard-mib', type=int, default=64)
    parser.add_argument('--extra-candidates-index', help='Verified own native-policy SQLite index')
    parser.add_argument('--reuse-bank', action='append', default=[], help='Sealed identical observation/protocol bank; reuse exact labels without resimulation')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.rank < args.world or not 2 <= args.target_candidates <= 32:
        parser.error('Invalid rank/world or candidate budget')
    if args.max_scenes < 1 or args.max_simulation_budget < 1 or args.scoring_retries < 0 or args.shard_mib < 1:
        parser.error('All resource bounds must be positive')
    return args


def main():
    args = parse_args()
    try:
        return finalize(args) if args.finalize else build(args)
    except BaseException:
        destination = Path(args.output)/'failures'/f'rank{args.rank:05d}_{time.time_ns()}.json'
        atomic_json(destination, dict(status='FAILED', error=traceback.format_exc(), arguments=vars(args)))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
