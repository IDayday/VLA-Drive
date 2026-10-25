"""Index completed own-policy TRAIN samples, or re-simulate sealed bank labels."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import pickle
import sqlite3
import time

import numpy as np
import torch

from starVLA.model.modules.qwen_world_postcot.candidate_bank import (
    CandidateBank, OFFICIAL_METRIC_NAMES, atomic_json, file_sha, fingerprint,
    trajectory_hash, CandidateRecord,
)


def index_native(args):
    from starVLA.rl.qwen_world_postcot.contracts import require_dispatch, boundary_stop_due
    require_dispatch()
    population = json.loads(Path('/mnt/project/flow-safe-improvement-20261013/population/manifest.json').read_text())
    original_train, navtest = set(population['train_tokens']), set(population['navtest_tokens'])
    train_manifest = Path(args.train_manifest)
    legal = json.loads(train_manifest.read_text())
    seal = json.loads((train_manifest.parent/'COMPLETE').read_text())
    if legal['cohort'] != 'TRAIN' or seal['manifest_sha256'] != file_sha(train_manifest):
        raise ValueError('Native candidate import requires the sealed current TRAIN role')
    train = set(legal['tokens'])
    if not train <= original_train:
        raise ValueError('Current TRAIN includes a foreign population')
    physical = lambda log: log.rsplit('_', 2)[0]
    excluded = {physical(population['token_logs'][token]) for token in population['dev_tokens']+population['navtest_tokens']}
    if {physical(population['token_logs'][token]) for token in train} & excluded:
        raise ValueError('Whole physical DEV/NAVTEST logs must be embargoed from candidate fitting')
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError('Keep the completed native-candidate index immutable')
    destination.parent.mkdir(parents=True, exist_ok=True)
    building = destination.with_name(destination.name+'.building')
    if building.exists() and not args.resume:
        raise FileExistsError('Prior partial native index is retained; explicitly choose a new attempt')
    if args.resume and not building.exists():
        raise FileNotFoundError('No partial native index exists for explicit resume')
    db = sqlite3.connect(building)
    db.execute('CREATE TABLE IF NOT EXISTS candidates(record_id TEXT PRIMARY KEY,token TEXT NOT NULL,payload TEXT NOT NULL)')
    db.execute('CREATE INDEX IF NOT EXISTS tokens ON candidates(token)')
    db.execute('CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS completed_files(path TEXT PRIMARY KEY,sha256 TEXT NOT NULL,eligible INTEGER NOT NULL)')
    plan = dict(runs=[str(Path(path).resolve()) for path in args.run], rank=args.rank, world=args.world,
                max_files=args.max_files, all_files=args.all_files, source_data_manifest_sha256=file_sha(train_manifest))
    existing_plan = db.execute("SELECT value FROM metadata WHERE key='build_plan'").fetchone()
    if existing_plan and json.loads(existing_plan[0]) != plan:
        raise ValueError('Native index resume dependencies changed')
    if not existing_plan:
        db.execute('INSERT INTO metadata VALUES(?,?)', ('build_plan', json.dumps(plan, sort_keys=True)))
    db.commit()
    seen_files, protocols, run_records = [], set(), []
    previous_protocol = db.execute("SELECT value FROM metadata WHERE key='protocol'").fetchone()
    if previous_protocol:
        protocols.add(previous_protocol[0])
    started = time.monotonic()
    try:
        for run_path in args.run:
            run = Path(run_path).resolve()
            identity_path = run/'identity.json'; identity = json.loads(identity_path.read_text())
            body = dict(identity); expected = body.pop('sha256')
            if fingerprint(body) != expected or identity.get('purpose') != 'formal' or identity.get('schema') != 'native_qwen_sourceflow_RL_v1':
                raise ValueError('Only a real, own native Qwen-Flow formal RL lineage can supply these samples')
            for rank in range(8):
                done = json.loads((run/f'result_rank{rank}.json').read_text())
                if done.get('status') != 'COMPLETE' or done.get('qualification') or done['updates'] != 50000:
                    raise ValueError('Index only completed immutable own runs; not a live partially written behavior tree')
            files = sorted((run/'behavior').glob('collection*/rank*.pt'))
            if not args.all_files:
                files = files[:args.max_files]
            files = files[args.rank::args.world]
            run_records.append(dict(run=str(run), identity_sha256=file_sha(identity_path),
                source_SFT_sha256=identity['SFT_sha256'], declared_files=len(files)))
            for path in files:
                if boundary_stop_due():
                    atomic_json(destination.with_suffix('.PAUSED.json'), dict(status='OPERATOR_PAUSED',
                        reason='finite task resource deadline', partial_index=str(building)))
                    return 75
                previous_file = db.execute('SELECT sha256 FROM completed_files WHERE path=?', (str(path),)).fetchone()
                artifact_sha = file_sha(path)
                if previous_file is not None:
                    if previous_file[0] != artifact_sha:
                        raise ValueError('Completed native behavior artifact changed during index resume')
                    seen_files.append(dict(path=str(path), sha256=artifact_sha))
                    continue
                payload = torch.load(path, map_location='cpu', weights_only=False)
                token, log = payload['token'], payload['log']
                if token not in original_train or token in navtest:
                    raise ValueError('A non-TRAIN candidate entered the native behavior importer')
                if population['token_logs'][token] != log:
                    raise ValueError('Native behavior scene/log mismatch')
                seen_files.append(dict(path=str(path), sha256=artifact_sha))
                if token not in train:
                    db.execute('INSERT INTO completed_files VALUES(?,?,?)', (str(path), artifact_sha, 0))
                    db.commit()
                    continue
                version = payload['behavior_version']
                actor_identity = fingerprint(dict(SFT=identity['SFT_sha256'], LoRA=version['adapter_sha256']))
                for index, feedback in enumerate(payload['records']):
                    if feedback['status'] != 'OK' or feedback['token'] != token or feedback['log'] != log:
                        raise ValueError('A scoring failure is not reusable as a zero label')
                    request = np.asarray(feedback['requested_trajectory'])
                    key = trajectory_hash(request)
                    metrics = {name: float(feedback['metrics'][name]) for name in (*OFFICIAL_METRIC_NAMES, 'score')}
                    if not all(np.isfinite(v) and 0 <= v <= 1 for v in metrics.values()):
                        raise ValueError('Invalid raw native PDM labels')
                    protocols.add(feedback['protocol_sha256'])
                    if len(protocols) != 1:
                        raise ValueError('Native candidate index cannot mix scoring protocols')
                    db.execute('INSERT OR IGNORE INTO metadata VALUES(?,?)', ('protocol', feedback['protocol_sha256']))
                    compact = dict(token=token, log=log, status='OK', protocol=feedback['protocol'],
                        protocol_sha256=feedback['protocol_sha256'], cache=feedback['cache'],
                        metrics=metrics, requested_trajectory=request.astype(np.float32).tolist())
                    row = dict(requested_trajectory=compact['requested_trajectory'], candidate_source='native_qwen_rl',
                        actor_checkpoint_sha256=actor_identity, feedback=compact,
                        sampling=dict(seed=identity['recipe']['seed'], scheme=identity['rng_protocol'],
                            source_sampler=identity['recipe']['grpo']['sample'], candidate_index=index,
                            initial_noise_sha256=__import__('hashlib').sha256(
                                payload['chain'][index, 0].contiguous().numpy().tobytes()).hexdigest()),
                        source_parameters=dict(behavior_artifact=str(path), behavior_artifact_sha256=artifact_sha,
                            native_run=str(run), native_run_identity=identity['sha256'],
                            behavior_version=version, SFT_sha256=identity['SFT_sha256'],
                            actor_identity_kind='SFT+LoRA parameter-state hashes; not a serialized intermediate checkpoint'))
                    record_id = fingerprint([token, key, feedback['protocol_sha256'], feedback['cache']['sha256']])
                    previous = db.execute('SELECT payload FROM candidates WHERE record_id=?', (record_id,)).fetchone()
                    if previous is not None:
                        if json.loads(previous[0])['feedback'] != compact:
                            raise ValueError('Identical native requests have conflicting canonical labels')
                        continue
                    db.execute('INSERT INTO candidates VALUES(?,?,?)', (record_id, token,
                        json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False)))
                db.execute('INSERT INTO completed_files VALUES(?,?,?)', (str(path), artifact_sha, 1))
                db.commit()
        if len(protocols) != 1:
            raise ValueError('Native candidate index cannot mix scoring protocol versions')
        info = dict(schema='world_postcot_own_native_candidate_index_v1', split='TRAIN',
            score_protocol_sha256=next(iter(protocols)), runs=run_records,
            source_data_manifest_sha256=file_sha(train_manifest),
            input_files_sha256=fingerprint(seen_files), input_files=len(seen_files),
            rank=args.rank, world=args.world, complete_for_declared_input_files=True,
            all_files_requested=args.all_files, source_file_budget=None if args.all_files else args.max_files,
            candidates=db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0],
            scenes=db.execute('SELECT COUNT(DISTINCT token) FROM candidates').fetchone()[0],
            skipped_embargo_source_files=db.execute('SELECT COUNT(*) FROM completed_files WHERE eligible=0').fetchone()[0],
            seconds=time.monotonic()-started, external_model_weights_loaded=False)
        db.execute('INSERT INTO metadata VALUES(?,?)', ('identity', json.dumps(info, sort_keys=True)))
        db.commit(); db.close(); building.rename(destination)
        atomic_json(destination.with_suffix('.COMPLETE.json'), dict(status='COMPLETE',
            identity=info, index_sha256=file_sha(destination)))
        print(json.dumps(info))
        return 0
    except BaseException:
        db.close()
        raise


def merge_native_indices(args):
    """Atomically join disjoint scene-independent behavior-file index shards."""
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError('Native index versions are immutable')
    parts = [Path(path).resolve() for path in args.part]
    if not parts or len(set(parts)) != len(parts):
        raise ValueError('Distinct completed native index parts are required')
    identities, seals = [], []
    for path in parts:
        seal = json.loads(path.with_suffix('.COMPLETE.json').read_text())
        if seal['status'] != 'COMPLETE' or seal['index_sha256'] != file_sha(path):
            raise ValueError('Native index part is not sealed or changed')
        identities.append(seal['identity']); seals.append(seal)
    first = identities[0]
    if (set(value['rank'] for value in identities) != set(range(first['world']))
            or len(parts) != first['world']):
        raise ValueError('Every declared native indexing rank must finish before merge')
    keys = ('schema', 'split', 'world', 'source_data_manifest_sha256', 'score_protocol_sha256',
            'all_files_requested', 'source_file_budget')
    if any(any(value[key] != first[key] for key in keys) for value in identities):
        raise ValueError('Native indexing parts have different data, budget or scorer identities')
    source_runs = [[(r['run'], r['identity_sha256'], r['source_SFT_sha256']) for r in value['runs']]
                   for value in identities]
    if any(value != source_runs[0] for value in source_runs):
        raise ValueError('Native index parts came from different actor lineages')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name+'.building')
    if temporary.exists():
        raise FileExistsError('Preserve prior merge attempt; use a new output version')
    output = sqlite3.connect(temporary)
    output.execute('CREATE TABLE candidates(record_id TEXT PRIMARY KEY,token TEXT NOT NULL,payload TEXT NOT NULL)')
    output.execute('CREATE INDEX tokens ON candidates(token)')
    output.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
    output.execute('CREATE TABLE completed_files(path TEXT PRIMARY KEY,sha256 TEXT NOT NULL,eligible INTEGER NOT NULL)')
    started = time.monotonic()
    try:
        for path in parts:
            with sqlite3.connect(f'file:{path}?mode=ro&immutable=1', uri=True) as part:
                cursor = part.execute('SELECT record_id,token,payload FROM candidates')
                while values := cursor.fetchmany(1024):
                    for record_id, token, payload in values:
                        previous = output.execute('SELECT payload FROM candidates WHERE record_id=?', (record_id,)).fetchone()
                        if previous is not None:
                            if json.loads(previous[0])['feedback'] != json.loads(payload)['feedback']:
                                raise ValueError('Native index shards disagree on exact canonical candidate labels')
                        else:
                            output.execute('INSERT INTO candidates VALUES(?,?,?)', (record_id, token, payload))
                    output.commit()
                for value in part.execute('SELECT path,sha256,eligible FROM completed_files'):
                    # A duplicate source artifact means producer ownership was
                    # not disjoint; never count it twice as new supervision.
                    output.execute('INSERT INTO completed_files VALUES(?,?,?)', value)
                output.commit()
        files = list(output.execute('SELECT path,sha256 FROM completed_files ORDER BY path'))
        info = {key: first[key] for key in keys if key != 'world'}
        info.update(rank=0, world=1, runs=[dict(run=run, identity_sha256=sha, source_SFT_sha256=sft)
            for run, sha, sft in source_runs[0]], complete_for_declared_input_files=True,
            input_files=len(files), input_files_sha256=fingerprint(files),
            candidates=output.execute('SELECT COUNT(*) FROM candidates').fetchone()[0],
            scenes=output.execute('SELECT COUNT(DISTINCT token) FROM candidates').fetchone()[0],
            skipped_embargo_source_files=output.execute('SELECT COUNT(*) FROM completed_files WHERE eligible=0').fetchone()[0],
            merged_parts=[dict(path=str(path), sha256=seal['index_sha256']) for path,seal in zip(parts,seals)],
            external_model_weights_loaded=False, seconds=time.monotonic()-started)
        output.execute('INSERT INTO metadata VALUES(?,?)', ('identity', json.dumps(info, sort_keys=True)))
        output.commit(); output.close(); temporary.rename(destination)
        atomic_json(destination.with_suffix('.COMPLETE.json'), dict(status='COMPLETE', identity=info,
            index_sha256=file_sha(destination)))
        print(json.dumps(info)); return 0
    except BaseException:
        output.close(); raise


def audit(args):
    from starVLA.rl.qwen_world_postcot.contracts import require_dispatch, boundary_stop_due
    from starVLA.rl.simwam_recipe.reward_readouts import RewardService
    require_dispatch()
    bank = CandidateBank(args.bank, allowed_splits=('TRAIN', 'DEV'), max_cached_scenes=0)
    tokens = sorted(bank.manifest['scene_ids'], key=lambda token: fingerprint([args.seed, token]))[:args.max_scenes]
    service = RewardService(Path.cwd(), workers=1)
    checks, coverage, calls = [], {name: set() for name in OFFICIAL_METRIC_NAMES}, 0
    started = time.monotonic()
    try:
        if service.protocol['sha256'] != bank.manifest['identity']['score_protocol_sha256']:
            raise ValueError('Real scorer version differs from the bank labels')
        for token in tokens:
            if boundary_stop_due():
                raise RuntimeError('Finite resource deadline reached before label audit completed')
            rows = bank.records(token)[:args.max_candidates]
            if calls + 2*len(rows)+1 > args.max_simulation_budget:
                raise ValueError('Audit simulation budget cannot cover its fixed requested checks')
            requests = np.asarray([row.trajectory_fp32 for row in rows], dtype=np.float32)
            batch = service.score((token, rows[0].source_log, requests)); calls += len(rows)
            reverse = service.score((token, rows[0].source_log, requests[::-1].copy()))[::-1]; calls += len(rows)
            single = service.score((token, rows[0].source_log, requests[:1]))[0]; calls += 1
            errors = []
            for index, (saved, fresh, reordered) in enumerate(zip(rows, batch, reverse)):
                for name in OFFICIAL_METRIC_NAMES:
                    target = saved.raw_factor_labels[name]
                    if saved.valid_masks[name]:
                        coverage[name].add(target)
                        errors += [abs(target-fresh['metrics'][name]), abs(target-reordered['metrics'][name])]
                        if index == 0:
                            errors.append(abs(target-single['metrics'][name]))
                errors += [abs(saved.raw_total_score-fresh['metrics']['score']),
                           abs(saved.raw_total_score-reordered['metrics']['score'])]
            maximum = max(errors, default=0.)
            if maximum > args.atol:
                raise ValueError(f'Canonical cached/batch/single/pool-order label mismatch: {token}, {maximum}')
            checks.append(dict(scene_id=token, candidates=len(rows), max_absolute_error=maximum))
        report = dict(status='PASS', actual_backend='existing official tracking/simulation canonical reference+one candidate',
            bank_hash=bank.bank_hash, scene_count=len(checks), fixed_scene_ids=tokens,
            candidate_checks=sum(row['candidates'] for row in checks), actual_scoring_requests=calls,
            absolute_tolerance=args.atol, maximum_absolute_error=max(row['max_absolute_error'] for row in checks),
            cached_vs_recomputed=True, batch_vs_single=True, candidate_reordering_invariant=True,
            valid_label_values={name: sorted(values) for name, values in coverage.items()},
            boundary_coverage={name: 0. in values and 1. in values for name, values in coverage.items()},
            missing_boundary_coverage=[name for name, values in coverage.items() if name in OFFICIAL_METRIC_NAMES[:3] and not (0. in values and 1. in values)],
            coverage_scope='Pass applies to actual observed label values; absent failure types are explicitly untested',
            seconds=time.monotonic()-started, checks=checks)
        if Path(args.output).exists():
            raise FileExistsError('Keep previous real label audit evidence')
        atomic_json(args.output, report)
        print(json.dumps(report))
        return 0
    finally:
        service.close(); bank.close()


def _audit_production_shard(job):
    root, name, expected, timing = job[:4]
    index_rows = job[4] if len(job)>4 else None
    path = Path(root)/name
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Shard escaped the immutable bank')
    digest=hashlib.sha256();sources=Counter();scenes=Counter();boundaries={};corrections=[];soft_nc=0
    offset=seen=0
    with path.open('rb') as stream:
        for line in stream:
            digest.update(line);record=CandidateRecord(**json.loads(line)).validate()
            if index_rows is not None:
                if seen>=len(index_rows):raise ValueError('Shard has an unindexed complete record')
                indexed_offset,length,checksum,record_id=index_rows[seen]
                if (indexed_offset!=offset or length!=len(line) or checksum!=hashlib.sha256(line).hexdigest()
                        or record_id!=record.record_id):
                    raise ValueError('Full production record offset/checksum/identity disagrees with immutable index')
            offset+=len(line);seen+=1
            sources[record.candidate_source]+=1;scenes[record.scene_id]+=1
            values=record.raw_factor_labels
            expected_score=values[OFFICIAL_METRIC_NAMES[0]]*values[OFFICIAL_METRIC_NAMES[1]]*(
                5*values[OFFICIAL_METRIC_NAMES[3]]+5*values[OFFICIAL_METRIC_NAMES[2]]+2*values[OFFICIAL_METRIC_NAMES[4]])/12
            if abs(record.raw_total_score-expected_score)>5e-15:
                raise ValueError('Canonical physical candidate aggregation changed')
            bits=0
            for index,name in enumerate(OFFICIAL_METRIC_NAMES[:2]):
                if record.valid_masks[name] and values[name] in (0.,1.):bits|=1<<(2*index+int(values[name]))
            boundaries[record.scene_id]=boundaries.get(record.scene_id,0)|bits
            soft_nc+=values[OFFICIAL_METRIC_NAMES[0]]==.5
            if record.scene_id in timing and record.candidate_source=='logged_gt':
                target=timing[record.scene_id]
                if record.trajectory_sha256!=target['trajectory_sha256']:
                    raise ValueError('Logged-pose timing correction attached to a different request')
                corrections.append(dict(target,record_id=record.record_id,
                    score_protocol_sha256=record.score_protocol_sha256,context_sha256=record.context_sha256,
                    old_candidate_source='logged_gt',source_semantics='logged_pose_candidate_time_mismatch',
                    valid_official_candidate=True,qualified_fixed_dt_imitation_target=False,
                    actual_official_request_times_seconds=[.5*index for index in range(1,9)]))
    if digest.hexdigest()!=expected['sha256'] or path.stat().st_size!=expected['bytes']:
        raise ValueError('Production shard checksum or complete byte count changed')
    if index_rows is not None and seen!=len(index_rows):raise ValueError('Index contains absent shard records')
    return dict(sources=dict(sources),scenes=dict(scenes),boundaries=boundaries,
        corrections=corrections,soft_NC_half_count=int(soft_nc),bytes=path.stat().st_size,
        every_record_index_offset_checksum_verified=index_rows is not None)


def production_coverage(args):
    """Deep complete-record audit and immutable timing-provenance correction.

    Existing scored coordinates/labels are never modified or resimulated.
    """
    from starVLA.rl.qwen_world_postcot.contracts import require_dispatch
    require_dispatch()
    bank=CandidateBank(args.bank,allowed_splits=('TRAIN',),max_cached_scenes=0)
    manifest=bank.manifest;bank.close()
    if manifest['record_count']>args.max_records:raise ValueError('Explicit complete-record audit budget is too small')
    data=Path(args.data_root);raw=json.loads((data/'manifest.json').read_text())
    if (file_sha(data/'manifest.json')!=manifest['identity']['source_data_manifest_sha256']
            or file_sha(data/'records.pkl')!=raw['records_sha256']):
        raise ValueError('Timing-provenance source changed')
    with (data/'records.pkl').open('rb')as stream:records=pickle.load(stream)
    timing={token:dict(scene_id=token,source_log=row['log'],
        trajectory_sha256=trajectory_hash(np.asarray(row['requested_GT'],dtype=np.float32)),
        raw_future_dt_seconds=row['raw_future_dt_seconds'],timestamp_gap=True)
        for token,row in records.items()if row.get('timestamp_gap')}
    del records
    started=time.monotonic();sources=Counter();scenes=Counter();boundaries={};corrections=[];soft_nc=byte_count=0
    index_records={name:[] for name in manifest['shards']}
    with sqlite3.connect(f'file:{Path(args.bank).resolve()/"index.sqlite3"}?mode=ro&immutable=1',uri=True)as index:
        for name,offset,length,checksum,record_id in index.execute('SELECT shard,offset,length,checksum,record_id FROM records'):
            if name not in index_records:raise ValueError('Index refers to a foreign complete shard')
            index_records[name].append((offset,length,checksum,record_id))
    if sum(map(len,index_records.values()))!=manifest['record_count']:raise ValueError('Full index record denominator changed')
    for values in index_records.values():values.sort(key=lambda value:value[0])
    jobs=[(str(Path(args.bank).resolve()),name,info,timing,index_records[name])for name,info in manifest['shards'].items()]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'))as pool:
        for result in pool.map(_audit_production_shard,jobs):
            sources.update(result['sources']);scenes.update(result['scenes']);corrections+=result['corrections']
            if not result['every_record_index_offset_checksum_verified']:raise ValueError('Incomplete per-record index audit')
            soft_nc+=result['soft_NC_half_count'];byte_count+=result['bytes']
            for token,value in result['boundaries'].items():boundaries[token]=boundaries.get(token,0)|value
    if (set(scenes)!=set(raw['tokens']) or dict(scenes)!=manifest['candidate_counts']
            or sum(scenes.values())!=manifest['record_count'] or min(scenes.values())<2):
        raise ValueError('Actual complete records disagree with full fixed population/index counts')
    if {row['scene_id']for row in corrections}!=set(timing) or len(corrections)!=len(timing):
        raise ValueError('Every timing-gap logged candidate must have exactly one auditable correction')
    if Path(args.output).exists():raise FileExistsError('Keep previous immutable production audit')
    report=dict(status='PASS',bank_hash=manifest['sha256'],scene_count=len(scenes),record_count=sum(scenes.values()),
        full_record_checksums_and_FP32_coordinate_hashes=True,every_record_index_offset_and_identity_verified=True,
        whole_shard_hashes_verified=True,
        source_counts=dict(sources),candidates_per_scene_histogram=dict(Counter(scenes.values())),
        hard_NC_or_DAC_pair_scenes=sum((value&3)==3 or (value&12)==12 for value in boundaries.values()),
        official_NC_half_labels=soft_nc,payload_bytes=byte_count,source_manifest_sha256=file_sha(data/'manifest.json'),
        timing_gap_logged_pose_candidates=len(corrections),timing_provenance_corrections=corrections,
        existing_coordinates_and_labels_modified=False,new_simulations=0,
        correction_reason='Raw logged poses with irregular future timestamps remain legal fixed0.5s scorer requests, but are not certified fixed-dt imitation GT',
        seconds=time.monotonic()-started,CPU_workers=args.workers,GPU=False)
    atomic_json(args.output,report);print(json.dumps({k:v for k,v in report.items()if k!='timing_provenance_corrections'}));return 0


def main():
    parser = argparse.ArgumentParser(__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    index = commands.add_parser('index-native')
    index.add_argument('--run', action='append', required=True)
    index.add_argument('--output', required=True)
    index.add_argument('--rank', type=int, default=0)
    index.add_argument('--world', type=int, default=1)
    index.add_argument('--max-files', type=int, default=1024)
    index.add_argument('--all-files', action='store_true')
    index.add_argument('--resume', action='store_true')
    index.add_argument('--train-manifest', default='/mnt/project/qwen-world-postcot-20261025/data/TRAIN/manifest.json')
    merge = commands.add_parser('merge-native')
    merge.add_argument('--part', action='append', required=True)
    merge.add_argument('--output', required=True)
    check = commands.add_parser('audit')
    check.add_argument('--bank', required=True)
    check.add_argument('--output', required=True)
    check.add_argument('--max-scenes', type=int, default=4)
    check.add_argument('--max-candidates', type=int, default=16)
    check.add_argument('--max-simulation-budget', type=int, default=256)
    check.add_argument('--seed', type=int, default=42)
    check.add_argument('--atol', type=float, default=1e-12)
    coverage=commands.add_parser('coverage')
    coverage.add_argument('--bank',required=True);coverage.add_argument('--output',required=True)
    coverage.add_argument('--data-root',default='/mnt/project/qwen-world-postcot-20261025/data/TRAIN')
    coverage.add_argument('--workers',type=int,default=8);coverage.add_argument('--max-records',type=int,default=1500000)
    args = parser.parse_args()
    if args.command == 'index-native':
        if not 0 <= args.rank < args.world or args.max_files < 1:
            parser.error('Invalid native-index resource bounds')
        return index_native(args)
    if args.command == 'merge-native':
        return merge_native_indices(args)
    if args.command=='coverage':
        if not 1<=args.workers<=16 or args.max_records<1:parser.error('Bounded CPU coverage budget required')
        return production_coverage(args)
    if min(args.max_scenes, args.max_candidates, args.max_simulation_budget) < 1 or args.atol < 0:
        parser.error('Invalid real label-audit bounds')
    return audit(args)


if __name__ == '__main__':
    raise SystemExit(main())
