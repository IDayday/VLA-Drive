"""Read only bounded journals and sealed evaluation metadata, never model data."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

OLD = Path('/mnt/project/qwen-world-postcot-20261025')
FOUR = Path('/mnt/project/qwen-world-four-frameworks-20261029')
REGISTRY = Path('/mnt/project/VLA-Drive-qwen-world-experiment-registry-20261029/reports/qwen_world_experiment_registry_20261029')
PROTOCOL = '2dee107a4e8358bc873530c0eb2ee16d8761e471eab1d80acc76022e789654e1'
TOKEN_HASH = '8c8536f7369f584794ae215fcfffd5027c4f2ca5bc84b2e5f320e001d040009b'
METHODS = {'RAW': '原轨迹 / 原始 L1', 'V12': 'V12 优化轨迹 / 原始 L1',
           'TIME_RESIDUAL': '时间分布加权', 'HUBER': '物理尺度 Huber',
           'PHYSICAL': '物理通道加权', 'DELTA': '坐标 + 相邻增量',
           **{'ARCH_'+a: '架构 '+a for a in 'ABCD'}}
METHODS.update({'SWITCH_'+k:'短续训63k→65k / '+label for k,label in {
    'CONTROL':'原损失对照','PHYSICAL':'物理加权','TIME_RESIDUAL':'时间加权',
    'DELTA':'相邻增量','HUBER':'Huber'}.items()})


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def tail(path, limit=64):
    """Bound every read; incomplete JSONL writes never become valid updates."""
    path = Path(path)
    try:
        with path.open('rb') as stream:
            start = max(0, path.stat().st_size-768*1024)
            stream.seek(start)
            lines = stream.read().splitlines()
    except OSError:
        return []
    if start:
        lines = lines[1:]
    rows = []
    for raw in lines[-limit-1:]:
        try:
            row = json.loads(raw)
        except (ValueError, UnicodeError):
            continue
        if isinstance(row, dict) and isinstance(row.get('updates'), int):
            rows.append(row)
    return rows[-limit:]


class Evidence:
    def __init__(self):
        self.inputs, self.warnings, self.cache = {}, [], {}

    def read(self, path, *, required=False, sealed=False):
        path = Path(path)
        if path in self.cache:
            value = self.cache[path]
        else:
            try:
                raw = path.read_bytes()
                value = json.loads(raw)
                self.inputs[str(path)] = dict(sha256=hashlib.sha256(raw).hexdigest(),
                                              bytes=len(raw), mtime=path.stat().st_mtime)
            except (OSError, ValueError) as error:
                if required:
                    self.warnings.append(f'{path}: {type(error).__name__}')
                value = {}
            self.cache[path] = value
        if sealed:
            body = dict(value)
            if body.pop('sha256', None) != fingerprint(body):
                raise ValueError('元数据 seal 不符: '+str(path))
        return value


def training_job(state, root):
    candidates = []
    for key, job in state.get('jobs', {}).items():
        command = job.get('command', [])
        if '--output' not in command or command[command.index('--output')+1] != str(root):
            continue
        if not any('train_il' in part for part in command):
            continue
        # The four-framework launcher names this timestamp `at`; lexical job
        # names cannot decide whether a recovered invocation supersedes a fail.
        dispatched=max(job.get(field,0)or 0 for field in('dispatch_unix','dispatched_unix','at'))
        candidates.append((dispatched, key, job))
    return max(candidates, default=(0, None, {}))[2]


def journal_snapshot(evidence, key, root, state, *, target=100000, selected=True):
    from .lifecycle import user_termination
    root = Path(root)
    rank_rows = [tail(root/f'metrics_rank{rank}.jsonl') for rank in range(8)]
    lasts = [rows[-1]['updates'] if rows else None for rows in rank_rows]
    complete_ranks = all(value is not None for value in lasts)
    common = min(lasts) if complete_ranks else None
    maps = [{row['updates']: row for row in rows} for rows in rank_rows]
    aligned = sorted(set.intersection(*(set(rows) for rows in maps))) if complete_ranks else []
    aligned = [step for step in aligned if step <= common][-48:]
    losses, seconds, data_seconds, gradient, skipped = {}, [], [], [], []
    for step in aligned:
        for rows in maps:
            row = rows[step]
            for name, value in row.get('losses', {}).items():
                if finite(value) and name != 'sigma_mean':
                    losses.setdefault(name, []).append(value)
            if finite(row.get('grad_norm')):
                gradient.append(row['grad_norm'])
        # Wall/update is a rank maximum, not the sum of GPU worker times.
        durations = [rows[step].get('slowest_rank_seconds', rows[step].get('seconds')) for rows in maps]
        if all(finite(x) for x in durations):
            seconds.append(max(durations))
        waits = [rows[step].get('data_seconds') for rows in maps]
        if all(finite(x) for x in waits):
            data_seconds.append(max(waits))
    for rows in rank_rows:
        if rows:
            skipped.append(rows[-1].get('optimizer_skipped_steps'))
    files = [root/f'metrics_rank{rank}.jsonl' for rank in range(8)]
    ages = [time.time()-p.stat().st_mtime for p in files if p.exists()]
    # A stale or missing rank cannot be hidden by one fresh writer.
    age = max(ages) if len(ages) == 8 else None
    job = training_job(state, root)
    receipt = Path(job['receipt']) if job.get('receipt') else None
    worker = evidence.read(receipt/'RESULT.json') if receipt else {}
    running = evidence.read(receipt/'RUNNING.json') if receipt else {}
    if not worker and receipt:
        worker = evidence.read(receipt/'result.json')
    status = ('已完成' if common is not None and common >= target else
              '筛选后未继续' if not selected else
              '训练故障 / 待恢复' if worker.get('status') in ('FAILED', 'FORCED_EXIT_REVIEW_REQUIRED') else
              '训练更新中（日志新鲜）' if age is not None and age < 1800 else
              '等待 / 状态待核实')
    latest = evidence.read(root/'LATEST.json')
    latest_rows=[rows[-1]for rows in rank_rows if rows]
    optimizer_counts=[row.get('optimizer_global_steps')for row in latest_rows]
    count_valid=complete_ranks and all(row.get('optimizer_global_steps',row['updates'])==row['updates']for row in latest_rows)
    if not count_valid and complete_ranks:
        status='optimizer更新计数待核验'
        common=None
    if user_termination(root):status='用户已终止'
    speed = statistics.mean(seconds) if len(seconds) >= 8 else None
    eta = ((target-common)*speed/3600 if speed and common is not None and common < target
           and status.startswith('训练更新中') else None)
    loss_mean = {k: statistics.mean(v) for k, v in losses.items()}
    curve = []
    for step in aligned:
        row = dict(updates=step)
        for name in loss_mean:
            values = [rows[step].get('losses', {}).get(name) for rows in maps]
            if all(finite(x) for x in values):
                row[name] = statistics.mean(values)
        curve.append(row)
    identity = evidence.read(root/'IDENTITY.json')
    return dict(id=key, label=METHODS.get(key, key), root=str(root), target=target,
        updates=common, observed_rank_updates=lasts, latest_checkpoint=latest,
        fraction=common/target if common is not None else None, status=status,
        journal_age_seconds=age, host=job.get('host'), worker_status=worker.get('status', running.get('status')),
        worker_returncode=worker.get('returncode'), worker_evidence=str(receipt) if receipt else None,
        recent_losses=loss_mean, recent_window_updates=len(aligned),
        recent_seconds_per_update=speed, recent_data_wait_seconds=statistics.mean(data_seconds) if data_seconds else None,
        estimated_remaining_hours=eta, ETA_excludes_future_saves_evaluations_and_contention=True,
        optimizer_skipped_steps=skipped, recent_gradient=statistics.mean(gradient) if gradient else None,
        optimizer_counts=optimizer_counts,optimizer_count_contract_valid=count_valid,
        peak_allocated_GiB=max((row.get('peak_allocated', 0) for rows in rank_rows for row in rows), default=0)/1024**3,
        scene_exposures=common*32 if common is not None else None,
        source_files_sha256=fingerprint(identity.get('source_files', {})),
        evidence_mtime=max((p.stat().st_mtime for p in files if p.exists()), default=None), recent_curve=curve)


def classify(label, path):
    value = (label+' '+str(path)).upper()
    if '_RL_' in value or '/RL_' in value or label.startswith(('ATTENTION_', 'MEMORY_RL', 'TIME_RL')):
        return None
    for a in 'ABCD':
        if f'ARCH_{a}' in value or label.startswith(a+'_IL_'):
            return 'ARCH_'+a
    if 'OPTIMIZED' in value or 'V12' in value:
        return 'V12'
    method = next((name for name in ('TIME_RESIDUAL', 'PHYSICAL', 'HUBER', 'DELTA') if name in value), None)
    if 'LOSS_DESIGNS_20261028' in value or 'COORDINATE_LOSS_20261028' in value:
        return 'SWITCH_'+(method or 'CONTROL')
    return method or 'RAW'


def score_row(e, summary_path, label, *, method=None):
    """Only full, sealed, matching single-trajectory Test becomes a main score."""
    path = Path(summary_path)
    if not path.exists():
        return None
    try:
        summary = e.read(path, sealed=True)
        cohort = summary.get('cohort')
        evaluation = Path(summary.get('evaluation', path.parent))
        identity = e.read(evaluation/'identity.json', sealed=True)
        if cohort is None:
            cohort = identity.get('cohort')
        if cohort != 'NAVTEST':
            return None
        if summary.get('RL_updates') is not None:
            return None
        method = method or classify(label, path)
        if method is None:
            return None
        if (summary.get('status') != 'COMPLETE' or summary.get('scenes') != 12146
                or summary.get('invalid_scenes') != 0 or identity.get('seed') != 2101
                or identity.get('qualification', False)
                or summary.get('evaluation_identity_sha256') != identity.get('sha256')
                or identity.get('token_sha256') != TOKEN_HASH
                or identity.get('protocol', {}).get('sha256') != PROTOCOL):
            raise ValueError('完整性 / seed / 协议 / 人口合同不符')
        inference = identity.get('inference')
        if not (inference == 'single_candidate_cleanx_K10_shift5' or
                isinstance(inference, dict) and inference.get('sampler') == 'ODE'
                and inference.get('steps') == 10 and inference.get('shift') == 5
                and inference.get('candidates') == 1):
            raise ValueError('非主单候选 K10 shift5 ODE')
        metric = summary['means']['score']*100
        if not finite(metric) or abs(metric-summary['PDMS']) > 1e-8:
            raise ValueError('主分和小分重组均值不同')
        registration = e.read(evaluation.parent/'registration.json')
        update = registration.get('updates', summary.get('updates'))
        if update is None:
            export = e.read(evaluation.parent/'export/export.json')
            update = export.get('IL_parent', {}).get('seal', {}).get('updates')
        if update is None:
            import re
            matches = re.findall(r'(?:update|IL_?)(\d{4,6})', label+' '+str(evaluation), re.I)
            update = int(matches[-1]) if matches else None
        if update is None:
            raise ValueError('缺少 checkpoint update 身份')
        return dict(method=method, label=label, updates=update, PDMS=metric,
            means={k:100*v for k,v in summary['means'].items()}, evidence=str(path),
            evaluation=str(evaluation), summary_sha256=summary['sha256'],
            model_sha256=summary.get('model_sha256', identity.get('model_sha256')),
            token_sha256=TOKEN_HASH, protocol_sha256=PROTOCOL, seed=2101,
            scenes=12146, invalid_scenes=0, scope='已多次观察的 Test；不是盲测')
    except (KeyError, TypeError, ValueError) as error:
        e.warnings.append(f'不发布分数 {path}: {error}')
        return None


def discover_scores(e):
    candidates = []
    csv_path = REGISTRY/'all_completed_scores.csv'
    if csv_path.exists():
        with csv_path.open() as stream:
            for row in csv.DictReader(stream):
                if row['group'] == 'RL':
                    continue
                evaluation = Path(row['evaluation'])
                candidates.append((evaluation.parent/'SUMMARY.json', row['label']))
                candidates.append((evaluation/'SUMMARY.json', row['label']))
    # Only shallow control/index directories. Never recurse into scenes,
    # checkpoints, images, model tensors or candidate payloads.
    roots = [p for p in OLD.iterdir() if p.is_dir() and 'navtest' in p.name.lower()]
    roots += [p for p in FOUR.iterdir() if p.is_dir() and 'navtest' in p.name.lower()]
    roots += [OLD/'optimized_labels_IL_20261028/NAVTEST']
    roots += [OLD/'fresh_loss_IL_20261029'/arm/'NAVTEST' for arm in ('TIME_RESIDUAL','HUBER','PHYSICAL','DELTA')]
    # Legacy full-IL owners put one pinned request below each update directory.
    # Inspect these bounded control levels; never recurse through scene data.
    roots += [child for root in list(roots) if root.name=='NAVTEST'
              for child in root.glob('update*') if child.is_dir()]
    watched = set(roots)
    for root in roots:
        if not root.is_dir():
            continue
        pinned = e.read(root/'PINNED.json')
        names = pinned.get('models', {})
        children = [root/name for name in names] if names else [p for p in root.iterdir() if p.is_dir()]
        for child in children:
            if child.name in ('jobs','invocations','scenes','export','checkpoint','previous_owners'):
                continue
            watched.update([child, child/'evaluation'])
            candidates += [(child/'SUMMARY.json', child.name), (child/'evaluation/SUMMARY.json', child.name)]
    seen, scores = set(), []
    for path, label in candidates:
        if str(path) in seen:
            continue
        seen.add(str(path))
        row = score_row(e, path, label)
        if row:
            scores.append(row)
    unique = {}
    for row in scores:
        key = (row['method'], row['model_sha256'] or row['summary_sha256'], row['token_sha256'], row['protocol_sha256'], row['seed'])
        old = unique.get(key)
        if old is None or len(row['label']) < len(old['label']):
            unique[key] = row
    return sorted(unique.values(), key=lambda x:(x['method'], x['updates'])), watched


def source_audit(e):
    formal = FOUR/'runtime/formal_fdaff80'
    relative = ['starVLA/model/modules/qwen_world_compare/'+name+'.py' for name in
                ('model','independent_actor','shared_actor','world_stream','qwen_flow_adapter','heads','contracts')]
    relative += ['configs/qwen_world_compare/common.yaml']
    rows = []
    for rel in relative:
        path = formal/rel
        if path.exists():
            raw = path.read_bytes()
            rows.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()))
    return dict(scientific_source='fdaff809652370328e45c0d996ad706e1af2fc0c',
                diagram_interpretation='按实际执行实现绘图，不以提案代替已实现结构', files=rows)


def auxiliary_audit(e):
    result = []
    for step in (20000,40000,60000):
        path=OLD/f'aux_progress_20261028/measurements/update{step:06d}.json'
        data=e.read(path)
        for role in ('TRAIN','DEV'):
            rows=[r for r in data.get('rows',[]) if r.get('role')==role]
            if not rows:
                continue
            row=dict(updates=step,role=role,scenes=len(rows),evidence=str(path))
            for name in ('dino','future'):
                for metric in ('MSE','cosine','TRAIN_mean_baseline_MSE'):
                    values=[r[name][metric] for r in rows if finite(r.get(name,{}).get(metric))]
                    row[name+'_'+metric]=statistics.mean(values)if values else None
            objects=[r.get('objects',{}) for r in rows]
            count=sum(r.get('visible_objects',0) for r in objects)
            row['object_matched_class_accuracy']=sum(r.get('matched_class_correct',0) for r in objects)/count if count else None
            row['object_xy_error_m']=sum(r.get('matched_xy_distance_m_sum',0) for r in objects)/count if count else None
            row['object_recall_2m']=sum(r.get('visible_positive_recall2m_count',0) for r in objects)/count if count else None
            row['object_precision']=None
            result.append(row)
    return result


def snapshot(report_root):
    e=Evidence()
    campaign=e.read(FOUR/'campaign/STATE.json',required=True)
    plan=e.read(FOUR/'RUNS.json').get('plan',{})
    optimized=e.read(OLD/'optimized_labels_IL_20261028/STATE.json')
    selection=e.read(OLD/'fresh_loss_IL_20261029/SELECTION.json')
    runs=[journal_snapshot(e,'RAW',OLD/'runs/IL_full_seed42',e.read(OLD/'il_resume_user20261027T1507Z/controller_attempt002/STATE.json')),
          journal_snapshot(e,'V12',OLD/'optimized_labels_IL_20261028/runs/IL_full_seed42',optimized)]
    watch_dirs={OLD, FOUR, FOUR/'campaign', OLD/'fresh_loss_IL_20261029', OLD/'optimized_labels_IL_20261028'}
    for arm in ('PHYSICAL','TIME_RESIDUAL','DELTA','HUBER'):
        root=OLD/'fresh_loss_IL_20261029'/arm
        runs.append(journal_snapshot(e,arm,root/'runs/IL_full_seed42',e.read(root/'STATE.json'),selected=arm in selection.get('selected',[])))
        watch_dirs.update([root, root/'jobs', root/'runs/IL_full_seed42', root/'NAVTEST'])
    for arm in 'ABCD':
        entry=plan.get('architectures',{}).get(arm,{})
        run=Path(entry.get('IL',FOUR/f'runs/ARCH_{arm}/IL_seed42'))
        runs.append(journal_snapshot(e,'ARCH_'+arm,run,campaign))
        watch_dirs.add(run)
    scores, score_dirs=discover_scores(e)
    watch_dirs.update(score_dirs)
    for run in runs:
        watch_dirs.add(Path(run['root']))
        if run['worker_evidence']:
            watch_dirs.add(Path(run['worker_evidence']))
    dev=[]
    for path in(FOUR/'evaluations').glob('ARCH_*/DEV_*/SUMMARY.json'):
        try:
            data=e.read(path,sealed=True)
            if data.get('status')=='COMPLETE'and data.get('scenes')==1696 and not data.get('invalid_scenes'):
                dev.append(dict(architecture=path.parts[-3],updates=int(path.parent.name[-6:]),PDMS=data['PDMS'],evidence=str(path)))
        except (KeyError,ValueError):
            e.warnings.append('DEV 辅助记录待核验: '+str(path))
        watch_dirs.update([path.parent.parent,path.parent])
    policy=e.read('/mnt/project/server_dispatch_policy.json')
    allowed=[h for h in policy.get('allowed_hosts',{})if policy.get('global_dispatch_prohibitions',{}).get(h,{}).get('new_task_dispatch_allowed')is not False]
    source=source_audit(e)
    request=e.read(FOUR/'MATCHED_IL_EVAL_LATEST.json')
    pending_evaluations=[]
    for key in ('quad_root','legacy_root'):
        if not request.get(key):continue
        queue=Path(request[key]);watch_dirs.add(queue)
        pins=e.read(queue/'PINNED.json')
        for label,pin in pins.get('models',{}).items():
            directory=queue/label;watch_dirs.update([directory,directory/'evaluation'])
            done=e.read(directory/'COMPLETE.json')
            counts=[]
            for metadata in(directory/'evaluation').glob('rank*.json'):
                data=e.read(metadata);counts.append(data.get('new',0)+data.get('reused',0))
            pending_evaluations.append(dict(label=label,updates=pin['updates'],scenes_completed=sum(counts),
                total_scenes=12146,status='完整完成'if done.get('status')=='COMPLETE'else'评测 / 等待资源或native校验',
                root=str(directory)))
    original=e.read('/mnt/project/VLA-Drive-qwen-world-optimized-label-fit-20261030/reports/qwen_world_optimized_label_fit_20261030/original_loss_analysis.json')
    v12=e.read('/mnt/project/VLA-Drive-qwen-world-optimized-label-fit-20261030/reports/qwen_world_optimized_label_fit_20261030/optimized_loss_analysis.json')
    return dict(schema='qwen_world_IL_live_report_v1',generated_unix=time.time(),
        generated_utc=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),runs=runs,scores=scores,
        dev_diagnostics=dev,selection=selection,source_audit=source,auxiliary_audit=auxiliary_audit(e),
        loss_analyses={'RAW':original,'V12':v12},policy_authorized_hosts=allowed,
        protocol=dict(name='canonical NAVSIM v1.1',count=12146,seed=2101,inference='single clean-x K10 shift5 ODE',
                      hash=PROTOCOL,token_hash=TOKEN_HASH),
        four_framework_plan=dict(IL_updates=plan.get('IL_updates'),RL_updates=plan.get('RL_updates'),
            selection='完成各自100k IL后按已登记完整DEV规则选RL父；Test性能单列，不改选择器',
            no_short_training_elimination=plan.get('no_short_training_elimination')),
        matched_architecture_analysis=e.read(Path(report_root)/'reporting/matched_5k_analysis.json'),
        checkpoint_evaluation_request=request,pending_evaluations=pending_evaluations,
        policy_host_windows={h:row for h,row in policy.get('allowed_hosts',{}).items()if row.get('temporary')},
        warnings=e.warnings,inputs=e.inputs,watch_directories=sorted(str(p)for p in watch_dirs if p.is_dir()),
        read_only=True,new_GPU_forwards=0,new_scores=0,training_mutations=0)
