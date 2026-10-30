#!/usr/bin/env python3
"""Render the living RL report from receipts and bounded journal tails.

This program only reads experiment evidence and writes reporting artifacts. It
never launches a model, scores a scene, probes a GPU, or changes a run.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import tempfile
import signal
import subprocess
import sys
import threading
import time
import traceback
from typing import Any

DEFAULT_ROOT = Path('/mnt/project/qwen-world-rl-optimization-20261030')
REVIEW_ROOT = Path('/mnt/project/rl-grpo-review-20261030')
HISTORICAL_CURVES = Path('/mnt/project/VLA-Drive-qwen-world-rl-midpoints-20261028/reports/qwen_world_RL_midpoints_20261028/RL_learning_curves.json')
COLORS = ['#1769aa', '#d97919', '#15957d', '#8655bd', '#dc5266', '#687789', '#4ba1a5', '#b28a14']
ARTIFACT_ROOT = None


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def utc(value: Any = None) -> str:
    if value is None:
        value = datetime.now(timezone.utc).timestamp()
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        return str(value)
    except (ValueError, OverflowError, OSError):
        return str(value)


def finite(value: Any) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def fmt(value: Any, digits: int = 3, *, percent: bool = False) -> str:
    if not finite(value):
        return '待记录'
    if percent:
        value *= 100
    return f'{value:,.{digits}f}'


def number(value: Any) -> str:
    return f'{int(value):,}' if finite(value) else '待记录'


def gradient_fmt(value: Any) -> str:
    return f'{value:.3e}' if finite(value) else '待记录'


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path, warnings: list, inputs: list, *, optional: bool = True) -> Any:
    if not path.is_file():
        if not optional:
            warnings.append(f'缺少证据文件：{path}')
        return None
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
        inputs.append({'path': str(path), 'mtime': path.stat().st_mtime,
                       'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)})
        return value
    except (OSError, json.JSONDecodeError) as error:
        warnings.append(f'读取失败，保留为待核验：{path.name}（{type(error).__name__}）')
        return None


def read_tail(path: Path, limit: int = 64) -> list[dict]:
    """Ignore a partially appended last line and never scan an entire journal."""
    if not path.is_file():
        return []
    with path.open('rb') as handle:
        size = handle.seek(0, os.SEEK_END)
        start = max(0, size-512*1024)
        handle.seek(start)
        lines = handle.read().splitlines()
    if start and lines:
        lines = lines[1:]
    rows = []
    for raw in lines[-limit-1:]:
        try:
            row = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(row, dict) and finite(row.get('updates')):
            rows.append(row)
    return rows[-limit:]


def object_rows(value: Any, key: str) -> list[dict]:
    if isinstance(value, dict) and key in value:
        value = value[key]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    if isinstance(value, dict):
        return [dict(v, id=v.get('id', k)) for k, v in value.items() if isinstance(v, dict)]
    return []


def numeric_mean(rows: list[dict], path: tuple[str, ...]) -> float | None:
    values = []
    for row in rows:
        value = row
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if finite(value):
            values.append(float(value))
    return sum(values)/len(values) if values else None


def journal_snapshot(run: dict, warnings: list, inputs: list) -> dict:
    path = Path(run.get('root') or run.get('run_root') or '')
    snapshot = {'id': run.get('id', run.get('name', '未命名实验')), **run}
    if not path.is_dir() or not run.get('root', run.get('run_root')):
        snapshot['journal_state'] = '尚未写入训练日志'
        return snapshot
    progress = read_json(path/'progress.json', warnings, inputs) or {}
    snapshot['progress_receipt'] = progress
    rank_rows = {}
    for journal in sorted(path.glob('metrics_rank*.jsonl')):
        rows = read_tail(journal)
        if rows:
            rank_rows[journal.stem.removeprefix('metrics_rank')] = rows
    if not rank_rows:
        snapshot['journal_state'] = '尚无有效更新日志；不据登记声明已开训'
        snapshot['updates'] = progress.get('updates')
        return snapshot
    last = [rows[-1] for rows in rank_rows.values()]
    common = min(int(row['updates']) for row in last)
    expected_ranks = int(run.get('world_size', progress.get('world_size', 8)))
    rows = [row for rank in rank_rows.values() for row in rank if row['updates'] <= common]
    snapshot.update(updates=common, rank_updates={k: v[-1]['updates'] for k, v in rank_rows.items()},
                    journal_ranks=len(rank_rows), expected_ranks=expected_ranks,
                    journal_state=(f'已观察到 {len(rank_rows)}/{expected_ranks} rank 原生更新日志'
                                   +('；尚不能确认全局共同更新数' if len(rank_rows)<expected_ranks else '')),
                    evidence_mtime=max((path/f'metrics_rank{k}.jsonl').stat().st_mtime for k in rank_rows),
                    metric_window_rows=len(rows),
                    recent_reward=numeric_mean(rows, ('exploration', 'canonical_random_mean')),
                    recent_clipfrac=numeric_mean(rows, ('metrics', 'clipfrac')),
                    recent_ppo_blocked_fraction=numeric_mean(rows, ('metrics', 'ppo_blocked_fraction')),
                    recent_positive_active_fraction=numeric_mean(rows, ('metrics', 'positive_active_fraction')),
                    recent_negative_active_fraction=numeric_mean(rows, ('metrics', 'negative_active_fraction')),
                    recent_cispo_upper_hit_fraction=numeric_mean(rows, ('metrics', 'cispo_upper_hit_fraction')),
                    recent_nft_positive_loss=numeric_mean(rows, ('metrics', 'nft_positive_loss')),
                    recent_nft_negative_loss=numeric_mean(rows, ('metrics', 'nft_negative_loss')),
                    recent_nft_current_old_velocity_mse=numeric_mean(rows, ('metrics', 'nft_current_old_velocity_mse')),
                    recent_gradient=numeric_mean(rows, ('gradient_norm',)),
                    recent_loss=numeric_mean(rows, ('loss',)),
                    recent_seconds_update=numeric_mean(rows, ('seconds_update',)))
    for key in ['collections', 'new_candidates', 'raw_scene_exposures']:
        field = 'collection' if key == 'collections' else key
        values = [row.get(field) for row in last]
        if all(finite(v) for v in values):
            snapshot[key] = min(values)
        elif key in progress:
            snapshot[key] = progress[key]
    if snapshot.get('collections') is None and finite(snapshot.get('raw_scene_exposures')):
        exposures=snapshot['raw_scene_exposures']
        if expected_ranks>0 and exposures%expected_ranks==0:
            snapshot['collections']=exposures/expected_ranks
            snapshot['collections_source']='derived from global raw scene exposures / declared world size'
    snapshot['checkpoint'] = snapshot.get('checkpoint') or progress.get('checkpoint')
    snapshot['host'] = snapshot.get('host') or progress.get('host')
    if len(rank_rows) < expected_ranks:
        snapshot['observed_rank_min_update'] = common
        snapshot['updates'] = None
    if progress.get('status') == 'COMPLETE' and common >= progress.get('updates', math.inf):
        snapshot['training_complete'] = True
    return snapshot


def run_snapshot(run: dict, warnings: list, inputs: list, *, registry_mtime: float = 0) -> dict:
    """Read operator receipts separately from scientific update evidence."""
    record=dict(run,registry_status=run.get('status'))
    owner_path=Path(run['owner_state']) if run.get('owner_state') else None
    owner=read_json(owner_path,warnings,inputs) if owner_path else None
    if isinstance(owner,dict):
        record['owner_receipt']=owner
        record['owner_evidence_mtime']=owner_path.stat().st_mtime
        newer=record['owner_evidence_mtime']>=registry_mtime
        record['owner_receipt_newer_than_registry']=newer
        if newer:
            for key in ['status','phase','stage_budget_updates','checkpoint','target_updates']:
                if key in owner:record[key]=owner[key]
            if 'stage_target_updates' in owner:record['stage_budget_updates']=owner['stage_target_updates']
            job=owner.get('current_job',owner.get('job'))
            if isinstance(job,str):record['job']=job
            elif isinstance(job,dict):
                record['job']=job.get('root',job.get('path',record.get('job')))
    snapshot=journal_snapshot(record,warnings,inputs)
    job_root=Path(record['job']) if record.get('job') else None
    if job_root:
        if job_root.is_file():
            payload=read_json(job_root,warnings,inputs)
            snapshot['job_receipt']=payload
        else:
            job_receipts={}
            for filename in ['LAUNCH.json','RUNNING.json','RESULT.json','process.json']:
                value=read_json(job_root/filename,warnings,inputs)
                if isinstance(value,dict):job_receipts[filename]=value
            snapshot['job_receipts']=job_receipts
            result=job_receipts.get('RESULT.json')
            running=job_receipts.get('RUNNING.json',job_receipts.get('process.json'))
            snapshot['worker_terminal_receipt']=result is not None
            snapshot['worker_receipt_status']=(result.get('status','RESULT_RECORDED') if result else
                                               running.get('status','STARTED') if running else
                                               'LAUNCH_ONLY' if job_receipts else 'NO_WORKER_RECEIPT')
            snapshot['worker_returncode']=result.get('returncode',result.get('exit_code')) if result else None
            snapshot['worker_receipt_identity']={k:(running or result or {}).get(k) for k in
                ['host','worker_pid','worker_start_ticks','pid','start_ticks','gpu_uuids','gpus','start_unix']}
    return snapshot


def evaluation_entries(document: Any, *, default_arm: str | None = None) -> list[dict]:
    """Expand controller results/stage nodes without treating progress as scores."""
    entries=[]
    def walk(value):
        if isinstance(value,list):
            for node in value:walk(node)
            return
        if not isinstance(value,dict):return
        summary=value.get('summary')
        spec=value.get('spec') if isinstance(value.get('spec'),dict) else {}
        if isinstance(summary,dict):
            entry={k:value[k] for k in ('label','arm','cohort','updates') if k in value}
            entry.setdefault('label',spec.get('label'))
            entry.setdefault('cohort',spec.get('cohort'))
            entry['summary']=summary
            if default_arm:entry.setdefault('arm',default_arm)
            entries.append(entry)
        if (value.get('cohort') in ('TRAIN_DIAGNOSTIC','NAVTEST','DEV','RL_DEV')
                and isinstance(value.get('label'),str) and isinstance(value.get('output'),str)):
            entry=dict(label=value['label'],cohort=value['cohort'],evaluation=value['output'],
                       evidence=str(Path(value['output'])/'SUMMARY.json'),status='PENDING',
                       manifest=value.get('manifest'),evaluation_spec=value)
            if default_arm:entry['arm']=default_arm
            entries.append(entry)
        for key,node in value.items():
            if key not in ('summary','command','error','traceback'):walk(node)
    walk(document)
    return entries


def normalize_result(row: dict, warnings: list, inputs: list) -> dict:
    result = dict(row)
    nested=result.pop('summary',None)
    if isinstance(nested,dict):
        result={**nested,**{k:v for k,v in result.items() if v is not None}}
        if isinstance(nested.get('evidence'),str):result['evidence']=nested['evidence']
    elif isinstance(nested,str):result.setdefault('evidence',nested)
    if not result.get('evidence') and isinstance(result.get('evaluation'),str):
        result['evidence']=str(Path(result['evaluation'])/'SUMMARY.json')
    evidence = result.get('evidence')
    result['evidence_integrity_valid']=True
    if isinstance(evidence, str) and Path(evidence).is_file():
        source = read_json(Path(evidence), warnings, inputs)
        if isinstance(source, dict):
            if isinstance(source.get('sha256'),str):
                body=dict(source);saved=body.pop('sha256')
                actual=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                if saved!=actual:
                    warnings.append('评测 SUMMARY seal 不符，不发布分数：'+evidence)
                    result['evidence_integrity_valid']=False
            # Explicit registry fields define the comparison identity, while
            # denominator and completion are always taken from the evidence.
            for field in ['status', 'scenes', 'expected_scenes', 'invalid_scenes', 'PDMS',
                          'updates', 'seed', 'cohort', 'protocol_sha256', 'sha256',
                          'paired', 'means', 'train_feedback_allowed', 'old_DEV_is_train_diagnostic',
                          'RL_updates', 'IL_updates','token_sha256','data_manifest_sha256',
                          'model_sha256','evaluation','inference']:
                if field in source:
                    result[field] = source[field]
            if 'PDMS' not in result and isinstance(source.get('mean_percent'), dict):
                result['PDMS'] = source['mean_percent'].get('score')
            result['evidence_mtime'] = Path(evidence).stat().st_mtime
        else:
            result['evidence_integrity_valid']=False
    elif isinstance(evidence,str):
        # A controller may advertise a completed object before its real source
        # is visible. Do not promote the advertised scalar without that file.
        if str(result.get('status','')).upper()=='COMPLETE':
            result['evidence_integrity_valid']=False
        output=Path(result.get('evaluation') or Path(evidence).parent)
        if output.is_dir():
            result['scenes']=len(list((output/'scenes').glob('*.json')))
            result['status']='RUNNING' if result['scenes'] else 'PENDING'
            identity=read_json(output/'identity.json',warnings,inputs)
            if isinstance(identity,dict):
                result.setdefault('expected_scenes',identity.get('count'))
                result.setdefault('token_sha256',identity.get('token_sha256'))
                result.setdefault('RL_updates',identity.get('RL_updates'))
                result.setdefault('IL_updates',identity.get('IL_updates'))
        manifest=result.get('manifest')
        if result.get('expected_scenes') is None and isinstance(manifest,str):
            value=read_json(Path(manifest),warnings,inputs)
            if isinstance(value,dict):result['expected_scenes']=value.get('count',len(value.get('tokens',[])))
    label=result.get('label')
    if not result.get('arm'):
        if isinstance(label,str) and '_u' in label:
            result['arm']=label.split('_u',1)[0]
        elif isinstance(label,str) and label.startswith('IL70'):
            result['arm']='IL70_BASELINE'
        elif label:result['arm']=label
    is_baseline=(result.get('RL_updates') is None and finite(result.get('IL_updates')))
    result['evaluation_kind']='IL_BASELINE' if is_baseline else 'RL'
    result['rl_updates_for_plot']=0 if is_baseline else result.get('RL_updates',result.get('updates'))
    if result.get('updates') is None:
        result['updates']=result.get('RL_updates') if result.get('RL_updates') is not None else result.get('IL_updates')
    cohort = str(result.get('cohort') or '未指定')
    result['cohort'] = cohort
    scenes = result.get('scenes', result.get('completed_scenes'))
    expected = result.get('expected_scenes')
    if expected is None and cohort.upper() in ['NAVTEST', 'TEST', 'NAVTEST_V1']:
        expected = 12146
    result['scenes'], result['expected_scenes'] = scenes, expected
    valid = (str(result.get('status', '')).upper() == 'COMPLETE'
             and finite(scenes) and finite(expected) and int(scenes) == int(expected)
             and int(scenes)==scenes and int(expected)==expected
             and int(expected) > 0 and result.get('invalid_scenes', 0) == 0
             and finite(result.get('PDMS')) and result['evidence_integrity_valid'])
    result['publishable_complete'] = bool(valid)
    result['display_PDMS'] = result.get('PDMS') if valid else None
    result['completion_label'] = '完整结果' if valid else '未完整／待核验，不发布均分'
    return result


def relative_link(root: Path, path: Any, label: str = '原始证据') -> str:
    if not isinstance(path, str) or not path:
        return '<span class="muted">尚无收据</span>'
    if path.startswith('https://'):
        target = path
    else:
        candidate=Path(path) if Path(path).is_absolute() else root/path
        if not candidate.exists():
            return f'<span class="muted" title="{esc(candidate)}">{esc(label)}（待生成）</span>'
        target = os.path.relpath(candidate, root)
    return f'<a href="{esc(target)}">{esc(label)}</a>'


def table(headers: list[str], rows: list[list[str]], *, empty: str = '尚无登记项') -> str:
    body = ''.join('<tr>'+''.join('<td>'+cell+'</td>' for cell in row)+'</tr>' for row in rows)
    if not rows:
        body = f'<tr><td colspan="{len(headers)}" class="muted">{esc(empty)}</td></tr>'
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+body+'</tbody></table></div>'


def chart(series: list[dict], *, title: str, xlabel: str, ylabel: str) -> str:
    """Standard scientific plotting, embedded SVG and standalone PNG/PDF."""
    points = [(x, y) for row in series for x, y in row['points'] if finite(x) and finite(y)]
    if not points:
        return f'<div class="chart-empty"><strong>{esc(title)}</strong><p>等待符合完整性要求的结果；不绘制估计分数。</p></div>'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'svg.fonttype':'path','svg.hashsalt':'qwen-world-rl-report',
                                'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    fig, ax = plt.subplots(figsize=(11.5,3.8),layout='constrained')
    for i, row in enumerate(series):
        color = COLORS[i % len(COLORS)]
        ps = sorted((x,y) for x,y in row['points'] if finite(x) and finite(y))
        if ps:
            ax.plot([p[0] for p in ps],[p[1] for p in ps],'-o',color=color,lw=2,ms=4,label=row['label'])
    ax.set(xlabel=xlabel,ylabel=ylabel)
    ax.grid(axis='y',alpha=.2)
    ax.legend(frameon=False,ncols=min(4,len(series)),fontsize=9)
    buffer=io.StringIO();fig.savefig(buffer,format='svg',metadata={'Date':None})
    document=buffer.getvalue();document=document[document.index('<svg'):]
    downloads=''
    if ARTIFACT_ROOT is not None:
        slug='figure_'+hashlib.sha256(title.encode()).hexdigest()[:12]
        directory=ARTIFACT_ROOT/'reporting'/'figures';directory.mkdir(parents=True,exist_ok=True)
        atomic_text(directory/(slug+'.svg'),document)
        for extension in ['png','pdf']:
            binary=io.BytesIO();fig.savefig(binary,format=extension,dpi=180,
                metadata={'CreationDate':None,'ModDate':None} if extension=='pdf' else None)
            target=directory/(slug+'.'+extension)
            fd,temporary=tempfile.mkstemp(prefix='.'+target.name+'.',dir=directory)
            with os.fdopen(fd,'wb') as handle:
                handle.write(binary.getvalue());handle.flush();os.fsync(handle.fileno())
            os.replace(temporary,target)
        downloads='<p class="muted">独立图件：'+relative_link(ARTIFACT_ROOT,str(directory/(slug+'.png')),'PNG')+' · '+relative_link(ARTIFACT_ROOT,str(directory/(slug+'.pdf')),'PDF')+'</p>'
    plt.close(fig)
    return '<figure><figcaption><b>'+esc(title)+'</b></figcaption>'+document+downloads+'</figure>'


def flatten_resources(data: Any) -> list[dict]:
    if not isinstance(data, dict):
        return []
    data = data.get('second') or data.get('latest') or data.get('inventory') or data
    if isinstance(data, dict) and isinstance(data.get('hosts'), dict):
        data = data['hosts']
    if not isinstance(data, dict):
        return []
    result = []
    for host, record in data.items():
        if not isinstance(record, dict) or not isinstance(record.get('gpus'), list):
            continue
        owners = record.get('owners', [])
        for gpu in record['gpus']:
            if not isinstance(gpu, dict):
                continue
            uuid = gpu.get('uuid') or gpu.get('gpu_uuid')
            result.append({**gpu, 'host':host, 'uuid':uuid,
                           'reported_owners':sum(o.get('uuid', o.get('gpu_uuid')) == uuid for o in owners),
                           'inventory_at':record.get('at', record.get('utc'))})
    return result


def load_snapshot(root: Path) -> dict:
    warnings, inputs = [], []
    registration = read_json(root/'REGISTRATION.json', warnings, inputs) or {}
    runs_input = read_json(root/'RUNS.json', warnings, inputs) or {}
    state = read_json(root/'STATE.json', warnings, inputs) or {}
    campaign = read_json(root/'owners'/'CAMPAIGN'/'STATE.json',warnings,inputs) or {}
    run_map = {}
    # RUNS is the live dispatch registry. A historical STATE.runs must never
    # restore old output paths/hosts/status over its newer dispatch records.
    for record in object_rows(registration.get('runs', []), 'runs')+object_rows(state.get('runs', []), 'runs')+object_rows(runs_input, 'runs'):
        key = record.get('id') or record.get('name')
        if key:
            run_map[key] = {**run_map.get(key, {}), **record}
    # The campaign's current owner state is dynamic; registration identity
    # remains sourced from RUNS, while newer stage fields may advance here.
    for record in object_rows(campaign.get('runs',[]),'runs'):
        key=record.get('id') or record.get('name')
        if key in run_map:
            run_map[key]=dict(run_map[key],campaign_stage_status=record.get('status'))
    registry_mtime=(root/'RUNS.json').stat().st_mtime if (root/'RUNS.json').exists() else 0
    runs = [run_snapshot(v, warnings, inputs,registry_mtime=registry_mtime) for v in run_map.values()]
    results_input = read_json(root/'RESULTS.json', warnings, inputs) or {}
    result_rows = object_rows(registration.get('results', []), 'results')+object_rows(state.get('results', []), 'results')+object_rows(results_input, 'results')
    result_rows += evaluation_entries(campaign)
    baseline_spec=read_json(root/'IL_BASELINE_EVALUATION.json',warnings,inputs)
    if baseline_spec:result_rows+=evaluation_entries(baseline_spec)
    for run in runs:
        owner=run.get('owner_receipt',{})
        if isinstance(owner,dict):
            for row in object_rows(owner.get('results',[]),'results'):
                result_rows.append(dict(arm=run['id'],**row) if 'arm' not in row else row)
            result_rows+=evaluation_entries(owner,default_arm=run['id'])
    results = [normalize_result(v, warnings, inputs) for v in result_rows]
    result_map = {}
    for row in results:
        key = row.get('evidence') or (row.get('arm'), row.get('updates'), row.get('cohort'),row.get('token_sha256'))
        previous=result_map.get(key)
        if previous and previous['publishable_complete'] and not row['publishable_complete']:continue
        result_map[key] = row
    results = list(result_map.values())
    history = read_json(HISTORICAL_CURVES, warnings, inputs) or {}
    journal_audit = read_json(REVIEW_ROOT/'world_rl_journal_audit.json', warnings, inputs) or {}
    chain_audit = read_json(REVIEW_ROOT/'world_rl_archive_audit.json', warnings, inputs) or {}
    reward_signal=read_json(root/'analysis'/'first1000_reward_signal.json',warnings,inputs) or {}
    nft_paired_analysis=read_json(root/'analysis'/'NFT_u001000_vs_IL70.json',warnings,inputs) or {}
    ppo_paired_analysis=read_json(root/'analysis'/'PPO_FULL_u001000_vs_IL70.json',warnings,inputs) or {}
    evaluation_throughput=read_json(root/'analysis'/'evaluation_throughput_20261030.json',warnings,inputs) or {}
    advantage_input=read_json(root/'analysis'/'advantage_normalization_u001000.json',warnings,inputs) or {}
    advantage_normalization=dict(status=advantage_input.get('status'),scope=advantage_input.get('scope'),
        arms={arm:{key:value.get(key) for key in ('unique_optimizer_updates','collections','normalization_evidence',
            'raw_batch_std','centered_batch_std','raw_to_centered_scale')} for arm,value in advantage_input.get('arms',{}).items()})
    stage_policy=read_json(root/'STAGE_POLICY_AMENDMENT_20261030.json',warnings,inputs) or {}
    continuation_files=list(root.glob('CONTINUATION_POLICY_AMENDMENT*.json'))
    continuation_path=max(continuation_files,key=lambda p:p.stat().st_mtime) if continuation_files else None
    continuation_policy=read_json(continuation_path,warnings,inputs) if continuation_path else {}
    early_continuation=read_json(root/'EARLY_5K_CONTINUATION_20261030.json',warnings,inputs) or {}
    batchstd_registration=read_json(root/'EXPERIMENT_PPO_BATCHSTD_REGISTRATION.json',warnings,inputs) or {}
    comfort_registration=read_json(root/'EXPERIMENT_PPO_SAFE_COMFORT_REGISTRATION.json',warnings,inputs) or {}
    comfort_counterfactual=read_json(root/'analysis'/'safe_comfort_counterfactual_u001000.json',warnings,inputs) or {}
    pipeline_startup=read_json(root/'STARTUP_SEVEN_PIPELINES_20261030.json',warnings,inputs) or {}
    evaluation_scheduling=read_json(root/'EVALUATION_SCHEDULING_POLICY.json',warnings,inputs) or {}
    first_collection_match=read_json(root/'analysis'/'ALGORITHM_ARMS_FIRST_COLLECTION_MATCH.json',warnings,inputs) or {}
    migration_path=root/'owners'/'CAMPAIGN'/'async_adoption_and_cispo_expansion_20261030'/'RECEIPT.json'
    cispo_expansion=read_json(migration_path,warnings,inputs) or {}
    cispo_refusal=read_json(root/'jobs'/'CISPO_FULL'/'formal_to005000_attempt001'/'RESULT.json',warnings,inputs) or {}
    adaptive_input=read_json(root/'adaptive_stages'/'stage000001.json',warnings,inputs) or {}
    adaptive_stage={key:value for key,value in adaptive_input.items() if key!='ordered_tokens'}
    report_config=read_json(root/'reporting'/'REPORT_CONFIG.json',warnings,inputs) or {}
    data_audit = None
    for path in [root/'data'/'DATA_AUDIT.json', root/'DATA_AUDIT.json', root/'data'/'population_audit.json',
                 root/'data'/'NAVTRAIN'/'POPULATION_AUDIT.json']:
        data_audit = read_json(path, warnings, inputs) or data_audit
    resource_input = (read_json(root/'RESOURCE_INVENTORY.json', warnings, inputs)
                      or read_json(root/'resources'/'inventory_latest.json', warnings, inputs)
                      or read_json(root/'resources'/'inventory_initial.json', warnings, inputs)
                      or read_json(root/'RESOURCE_PREFLIGHT.json', warnings, inputs)
                      or state.get('resources', {}))
    withdrawal=read_json(root/'resources'/'TSINGHUA_ZT2_WITHDRAWAL_20261030.json',warnings,inputs) or {}
    resources=flatten_resources(resource_input)
    remaining=withdrawal.get('remaining_hosts')
    active_resources=([row for row in resources if row['host'] in remaining] if isinstance(remaining,list) else
                      [row for row in resources if row['host']!=withdrawal.get('withdrawn_host')])
    return dict(generated_utc=utc(), generated_unix=datetime.now(timezone.utc).timestamp(),
                registration=registration, state=state,campaign_state=campaign, runs=runs, results=results,
                resources=resources,active_resources=active_resources,resource_raw=resource_input,
                resource_withdrawal=withdrawal,
                state_runs_older_than_registry=bool((root/'STATE.json').exists()
                    and (root/'STATE.json').stat().st_mtime<registry_mtime),
                historical=history, journal_audit=journal_audit, chain_audit=chain_audit,
                matched_reward_signal=reward_signal,
                nft_first_stage_paired_analysis=nft_paired_analysis,
                ppo_full_first_stage_paired_analysis=ppo_paired_analysis,
                evaluation_throughput_snapshot=evaluation_throughput,
                advantage_normalization=advantage_normalization,
                stage_policy_amendment=stage_policy,report_config=report_config,
                continuation_policy_amendment=continuation_policy or {},
                continuation_policy_path=str(continuation_path) if continuation_path else None,
                early_5k_continuation=early_continuation,
                batchstd_registration=batchstd_registration,cispo_expansion=cispo_expansion,
                comfort_registration=comfort_registration,comfort_counterfactual=comfort_counterfactual,
                pipeline_startup=pipeline_startup,evaluation_scheduling=evaluation_scheduling,
                first_collection_match=first_collection_match,
                cispo_prechild_refusal=cispo_refusal,adaptive_stage_one=adaptive_stage,
                data_audit=data_audit, warnings=warnings, inputs=inputs,
                events=state.get('events', registration.get('events', []))+campaign.get('events',[]))


def update_history(root: Path, snapshot: dict) -> list[dict]:
    path = root/'reporting'/'training_history.json'
    try:
        history = json.loads(path.read_text()) if path.is_file() else []
    except (OSError, ValueError):
        history = []
    for run in snapshot['runs']:
        if not finite(run.get('updates')):
            continue
        row = {k: run.get(k) for k in ['id', 'updates', 'new_candidates', 'recent_reward', 'recent_clipfrac', 'recent_gradient']}
        row['observed_unix'] = run.get('evidence_mtime', snapshot['generated_unix'])
        if not any(v.get('id') == row['id'] and v.get('updates') == row['updates'] for v in history):
            history.append(row)
    atomic_json(path, history)
    return history


def render(root: Path, snapshot: dict, history: list[dict]) -> str:
    registration, state = snapshot['registration'], snapshot['state']
    runs, results = snapshot['runs'], snapshot['results']
    complete = [r for r in results if r['publishable_complete']]
    observed = [r for r in runs if finite(r.get('updates')) and r['updates']>0]
    dispatched = [r for r in runs if r.get('job')]
    newest = max([v['mtime'] for v in snapshot['inputs']]
                 +[r['evidence_mtime'] for r in runs if r.get('evidence_mtime')], default=None)
    phase = state.get('phase', state.get('status', registration.get('status', '准备与资源核验')))
    if snapshot.get('state_runs_older_than_registry') and dispatched:
        phase=f'{len(dispatched)} 条实验已登记派发；{len(observed)} 条已观察全 rank 原生更新。逐实验 owner / worker 收据见下表'
    campaign=snapshot.get('campaign_state',{})
    if campaign:
        independent=(snapshot.get('continuation_policy_amendment',{}).get('implementation')=='APPLIED_INDEPENDENT_PIPELINES'
                     or campaign.get('independent_arm_pipelines') or 'independent' in str(campaign.get('scheduling','')).lower())
        schedule_label=('各实验臂独立 pipeline，周期评测异步' if independent else
                        '研究 controller 阶段门槛 '+str(campaign['current_gate']) if campaign.get('current_gate') is not None else
                        str(campaign.get('scheduling') or campaign.get('status','研究运行中')))
        phase=f'{schedule_label}；{len(dispatched)} 条已登记派发，{len(observed)} 条已观察全 rank 更新。逐实验状态与完整评测见下方'
    cards = [('本轮登记派发', str(len(dispatched))), ('全 rank 已有更新', str(len(observed))), ('本轮完整评测', str(len(complete))),
             ('本轮完整 Navtest 最佳', fmt(max((r['PDMS'] for r in complete if r['cohort'].upper() in ['NAVTEST','TEST','NAVTEST_V1']), default=None),4))]
    sections = []
    ppo_first=next((r for r in complete if r.get('arm')=='PPO_FULL' and r.get('cohort')=='NAVTEST' and r.get('RL_updates',r.get('updates'))==1000),None)
    pinned_analysis=snapshot.get('ppo_full_first_stage_paired_analysis') or snapshot.get('nft_first_stage_paired_analysis',{})
    pinned_reference=pinned_analysis.get('reference_PDMS') if pinned_analysis.get('status')=='COMPLETE' else None
    if ppo_first and finite(pinned_reference):
        delta=ppo_first['PDMS']-pinned_reference
        sections.append('<section><h2>已确认的完整成绩：PPO_FULL · 1,000 更新</h2><p class="note">完整 NAVTEST <b>'+fmt(ppo_first['PDMS'],6)+'</b>，固定 IL70 父模型 <b>'+fmt(pinned_reference,6)+'</b>，实测提高 <b>+'+fmt(delta,6)+' PDMS 点</b>。人口 12,146 场景、invalid=0，单候选 ODE K10 / shift5 / 推理 seed2101。'+relative_link(root,ppo_first.get('evidence'),'完整 SUMMARY')+'</p>')
        metric_labels={'no_at_fault_collisions':'无责任碰撞 NC','drivable_area_compliance':'可行驶区域 DAC','time_to_collision_within_bound':'碰撞时间 TTC','ego_progress':'行驶进度 EP','comfort':'舒适性','driving_direction_compliance':'方向合规'}
        sections.append(table(['完整 NAVTEST 指标','均值（0–100）'],[[label,fmt(100*ppo_first['means'][key],6)] for key,label in metric_labels.items() if finite(ppo_first.get('means',{}).get(key))]))
        train=next((r for r in complete if r.get('arm')=='PPO_FULL' and r.get('cohort')=='TRAIN_DIAGNOSTIC' and r.get('RL_updates',r.get('updates'))==1000 and r.get('scenes')==1024),None)
        train_base=next((r for r in complete if r.get('evaluation_kind')=='IL_BASELINE' and r.get('cohort')=='TRAIN_DIAGNOSTIC' and r.get('scenes')==1024),None)
        if train and train_base:
            sections.append('<p>固定 TRAIN1024 panel 为 '+fmt(train['PDMS'],6)+'，对其自身 IL 基线 '+fmt(train_base['PDMS'],6)+' 变化 '+f'{train["PDMS"]-train_base["PDMS"]:+.6f}'+' 点。<b>TRAIN panel 与完整 NAVTEST 的变化方向不同</b>；两个人口分别报告，不能用 panel 的下降替代 Test 结论。</p>')
        old_points=[r for r in snapshot.get('historical',{}).get('points',[]) if r.get('scenes')==12146 and finite(r.get('PDMS'))]
        if old_points:
            old=max(old_points,key=lambda r:r['PDMS'])
            sections.append('<p>历史已测最高仍为 '+esc(old.get('route'))+' / '+number(old.get('updates'))+' 更新的 <b>'+fmt(old['PDMS'],6)+'</b>；其父模型与训练预算不同。本轮这个 1k 点低于该旧纪录，不称为全系列新纪录，也不能直接按分数差归因算法优劣。</p>')
        has_resume=any(v.get('arm')=='PPO_FULL' for v in snapshot.get('early_5k_continuation',{}).get('dispatches',[]))
        if has_resume:sections.append('<p>该臂已实际登记原生 checkpoint 1,000 → 5,000 续训，恢复原 optimizer；新增训练更新以全 rank 日志为准。</p>')
        ppo_pair=snapshot.get('ppo_full_first_stage_paired_analysis',{})
        paired=ppo_pair.get('paired',{})
        interval=ppo_pair.get('CI95_PDMS',[])
        if ppo_pair.get('status')=='COMPLETE' and paired.get('scene_denominator')==12146 and len(interval)==2 and all(finite(v) for v in interval):
            pair_rows=[['同父模型配对差值',fmt(ppo_pair.get('delta_PDMS'),6)],
                ['物理 log 配对 95% 区间','['+', '.join(f'{v:+.6f}' for v in interval)+']'],
                ['G+ / G−（对全人口均分的贡献）',fmt(ppo_pair.get('G_plus_PDMS'),6)+' / '+fmt(ppo_pair.get('G_minus_PDMS'),6)],
                ['修复零分 / 原零分',number(paired.get('repaired_zero'))+' / '+number(paired.get('baseline_zero_denominator'))],
                ['新增零分 / 原非零分',number(paired.get('new_zero'))+' / '+number(paired.get('baseline_nonzero_denominator'))],
                ['高分（≥80）降零 / 原高分',number(paired.get('high_to_zero'))+' / '+number(paired.get('baseline_high_denominator'))]]
            sections.append('<h3>一次性 CPU 配对分析</h3><p>同一 12,146 场景；按 '+number(paired.get('physical_groups'))+' 个物理 log、'+number(paired.get('bootstrap'))+' 次 bootstrap、统计 seed '+number(paired.get('bootstrap_seed'))+'。</p>'+table(['配对统计','PDMS 点 / 场景数'],pair_rows))
            transitions=paired.get('safety_transitions',{})
            sections.append(table(['安全转移','新增失败 / IL原通过','修复 / IL原失败'],[[metric_labels.get(key,key),number(v.get('new_failure'))+' / '+number(v.get('baseline_pass')),number(v.get('repaired_failure'))+' / '+number(v.get('baseline_failure'))] for key,v in transitions.items()]))
            safety_rows=[]
            for key,label in metric_labels.items():
                values=ppo_pair.get('safety_metrics',{}).get(key,{})
                if values:safety_rows.append([label,fmt(values.get('IL_mean_percent'),6),fmt(values.get('PPO_FULL_mean_percent'),6),fmt(values.get('delta_percentage_points'),6)])
            sections.append(table(['配对指标（0–100）','IL70','PPO_FULL 1k','变化：百分点'],safety_rows))
            changed=ppo_pair.get('changed_scores',{})
            if changed:
                sections.append('<p>场景变化：改善 '+number(changed.get('improved'))+'、下降 '+number(changed.get('degraded'))+'、不变 '+number(changed.get('exactly_unchanged'))+'。整体收益包含严重失败修复：原零分修复 '+number(paired.get('repaired_zero'))+'，新增零分 '+number(paired.get('new_zero'))+'；收益并非所有场景普遍上升，下降场景也须保留诊断。</p>')
            categories=ppo_pair.get('scene_categories',{})
            if categories:
                labels={'score_down_EP_down_safety_unchanged':'EP下降，安全指标未变','score_down_with_new_NC_DAC_TTC_violation':'下降伴随新增NC/DAC/TTC违规','other_score_decrease':'其余分数下降'}
                total_loss=ppo_pair.get('G_minus_PDMS')
                category_rows=[]
                for key,values in categories.items():
                    loss=values.get('G_minus_PDMS')
                    fraction=loss/total_loss if finite(loss) and finite(total_loss) and total_loss<0 else None
                    category_rows.append([labels.get(key,key),number(values.get('scenes')),fmt(loss,6),fmt(fraction,1,percent=True)+'%' if fraction is not None else '待记录'])
                sections.append(table(['下降场景结构','场景数','G−贡献：PDMS点','占全部G−'],category_rows))
            log_rows=ppo_pair.get('log_concentration',{}).get('all_logs_sorted_by_G_minus',[])
            if log_rows:
                positive_logs=sum(finite(v.get('net_contribution_PDMS')) and v['net_contribution_PDMS']>0 for v in log_rows)
                negative_logs=sum(finite(v.get('net_contribution_PDMS')) and v['net_contribution_PDMS']<0 for v in log_rows)
                sections.append('<p>物理 log 净变化：'+number(positive_logs)+' 个改善、'+number(negative_logs)+' 个下降；后续应结合合法 TRAIN 的修复与退化反馈改进采样，不能把 NAVTEST 场景名单用于训练。</p>')
            verdict=('该配对区间高于零，支持这个固定 seed / 1k 阶段点的场景层面改善。' if interval[0]>0 else '该配对区间包含零，不能仅凭点差称为显著改善。' if interval[1]>=0 else '该配对区间低于零，应核对正向点差与配对证据是否一致。')
            sections.append('<p>'+verdict+' 这是当前完整阶段点，不代表已达到训练终局。'+relative_link(root,str(root/'analysis'/'PPO_FULL_u001000_vs_IL70.json'),'完整配对、失败修复与安全分析')+'</p>')
        else:
            sections.append('<p class="muted">CPU 配对 CI、G+/G−、失败修复和安全转移正在一次性分析，由唯一算法分析 owner 复用既有场景记录；不额外推理或评分。区间就绪前不补写显著性结论。</p>')
        sections.append('</section>')
    sections.append('<section><h2>当前任务与判断</h2><p>当前阶段：<strong>'+esc(phase)+'</strong>。本页将历史证据、本轮训练过程和本轮完整评测分开呈现；训练 reward 与部分评测均分不作为完整测试成绩。</p>')
    if campaign.get('owner'):
        owner=campaign['owner']
        sections.append('<p class="muted">研究 controller 收据：PID '+number(owner.get('pid'))+' / start ticks '+number(owner.get('start_ticks'))+'；'+relative_link(root,str(root/'owners'/'CAMPAIGN'/'STATE.json'),'当前 campaign STATE')+'。该 controller 与本页 HTML 更新进程分开。</p>')
    sections.append('<div class="priorities"><div><b>P0 · 可归因对照</b><p>最高分完整 IL 身份、数据人口、原生恢复、真实评测及资源归属。</p></div><div><b>P1 · 数据与行为保持</b><p>全量约 103k、困难与均匀覆盖、周期性未学好和退化场景再采样。</p></div><div><b>P2 · 目标与流程</b><p>CISPO、部署 ODE 对齐目标、采样与更新流程；按新候选数及计算成本对照。</p></div></div>')
    sections.append('<p class="note">用户已授权探索全量约 103k，不再限定困难集。若全量数据包含原 DEV，原 DEV 必须标为训练内诊断；独立评测人口以本轮登记为准。</p>')
    sections.append('<p>本轮高分起点为同架构原始 World IL70k，历史完整 Navtest <b>88.792910</b>。新算法应与同一父模型、同一观测及部署协议的基线比较；原 IL40k→RL 的分数只作历史参考，不归因到本轮新方法。</p>')
    stage_policy=snapshot.get('stage_policy_amendment',{})
    continuation_policy=snapshot.get('continuation_policy_amendment',{})
    ongoing_research=bool(continuation_policy or snapshot.get('report_config',{}).get('research_continuation_preference'))
    if ongoing_research:
        sections.append('<p class="note"><b>当前持续研究策略：</b>1k / 5k 诊断排名用于分析，不作硬淘汰；健康实验臂继续原定预算，不再限制为“对照＋最多两条挑战路线”。单臂故障隔离处理，不终止整体研究；合法 TRAIN 反馈就绪后可继续训练，完整 NAVTEST 异步完成。此前规划造成的屏障须修正后继续推进。</p>')
        if continuation_policy:
            sections.append('<p>'+relative_link(root,snapshot.get('continuation_policy_path'),'最新持续研究政策收据')+'；该收据覆盖旧的 5k 硬筛选口径。</p><details><summary>持续研究政策实施证据</summary><pre>'+esc(json.dumps(continuation_policy,ensure_ascii=False,indent=2))+'</pre></details>')
            if continuation_policy.get('implementation')=='APPLIED_INDEPENDENT_PIPELINES':
                sections.append('<p class="note"><b>全阶段独立 pipeline 已上线：</b>所有 gates 均不再等待全员或 NAVTEST；非自适应臂的 TRAIN 诊断也异步，自适应臂仅等待自己的合法 TRAIN 反馈。此次只接管 CPU owner，未向训练或评测子进程发信号；每条健康臂按原定 50k 预算独立推进。</p>')
            else:
                sections.append('<p class="note">实施范围：<b>1k 全局屏障已移除</b>，当前原生续训已派发；其他 gate 的全面异步、去等待改造仍在开发，尚未上线。政策意图与已经执行的改造分别记录，不将当前状态写成全部阶段均已改造完成。</p>')
        else:
            sections.append('<p class="muted">上述为用户最新要求；controller 正在应用，正式 CONTINUATION_POLICY_AMENDMENT 收据待落盘。报告收到后优先展示该收据，不把旧 5k 裁剪规则当作当前指令。</p>')
    elif stage_policy:
        sections.append('<p class="note"><b>阶段预算修订：</b>1,000 更新只作诊断；全部健康实验臂继续至 5,000 更新，5,000 才进行首次性能筛选，保留 PPO_FULL 对照及至多两条合格挑战路线。PPO_ADAPTIVE 的 stage 0 仍是均匀采样，须给后续真正自适应数据一次训练机会。此收据覆盖旧的 1k 性能裁剪口径；实际执行状态以 controller 的最新收据为准。</p><p>修订记录：'+esc(stage_policy.get('recorded_at','待记录'))+'；实施记录：'+esc(stage_policy.get('implementation','待记录'))+'。'+relative_link(root,str(root/'STAGE_POLICY_AMENDMENT_20261030.json'),'阶段预算修订收据')+'</p>')
    sections.append('<p>本轮保持训练 <b>seed42</b>、推理 <b>seed2101</b>；按用户要求不安排多种子实验。配对 bootstrap 的 seed42 是场景统计设置，不是额外训练种子。</p>')
    early_continuation=snapshot.get('early_5k_continuation',{})
    if early_continuation.get('dispatches'):
        dispatched_rows=[[esc(v.get('arm')),relative_link(root,v.get('job'),'续训 job / 收据'),
            relative_link(root,v.get('resume'),'原生 update1000 checkpoint'),relative_link(root,v.get('source'),'原 runtime')]
            for v in early_continuation['dispatches']]
        sections.append('<h3>已登记原生 1,000 → 5,000 续训派发</h3><p>下面是实际派发记录；恢复原 checkpoint 与 optimizer，不从新优化器起跑。派发不等同已经产生新增更新，训练进度仍以全 rank 日志为准。</p>'+table(['实验臂','派发','恢复点','科学 runtime'],dispatched_rows)+'<p>'+relative_link(root,str(root/'EARLY_5K_CONTINUATION_20261030.json'),'提前续训派发原始收据')+'</p>')
    cispo_expansion=snapshot.get('cispo_expansion',{})
    if cispo_expansion:
        cispo_run=next((r for r in runs if r.get('id')=='CISPO_FULL'),{})
        sections.append('<h3>已执行的评测扩容与续训恢复</h3><p>CISPO 的旧 world='+number(cispo_expansion.get('old_world'))+' 评测收到原生 STOP_REQUESTED 后在场景边界退出，随后扩为 '+number(cispo_expansion.get('new_world'))+' workers；保留 <b>'+number(cispo_expansion.get('preserved_scenes'))+'</b> 个成功 scene 的逐字节哈希，不重算成功场景。评测 identity、v3 科学 runtime、模型与推理 seed 保持不变。</p><p>'+relative_link(root,str(root/'owners'/'CAMPAIGN'/'async_adoption_and_cispo_expansion_20261030'/'RECEIPT.json'),'扩容与 CPU owner 交接收据')+' · '+relative_link(root,cispo_expansion.get('preserved_scene_hashes'),'保留场景原始 SHA256')+'</p>')
        refusal=snapshot.get('cispo_prechild_refusal',{})
        if refusal.get('child_not_started') is True and refusal.get('returncode')==75:
            sections.append('<p>CISPO 5k 续训 attempt001 在 zt4 只发生 pre-child 75 资源拒绝，模型子进程未启动；attempt002 已移至 zt5。当前全 rank 共同观察更新 '+number(cispo_run.get('updates'))+'，实际 job '+relative_link(root,cispo_run.get('job'),'attempt002 收据')+'。资源 owner guard 保持，没有为重试停止其他任务；原拒绝收据保留。</p>')
    adaptive_stage=snapshot.get('adaptive_stage_one',{})
    if adaptive_stage:
        adaptive_run=next((r for r in runs if r.get('id')=='PPO_ADAPTIVE'),{})
        sections.append('<h3>自适应数据 stage 1</h3><p>stage 1 名单已在 optimizer 阶段前发布；原生 1,000 → 2,000 续训已派发。已观察到的全 rank 共同更新仍按日志列为 <b>'+number(adaptive_run.get('updates'))+'</b>，不由“已启动”推算新增更新。stage 0 的均匀采样成绩不能当作 stage 1 自适应策略效果。</p><p>'+relative_link(root,str(root/'adaptive_stages'/'stage000001.json'),'stage 1 封存名单与合法 TRAIN 来源')+' · '+relative_link(root,adaptive_run.get('job'),'原生续训 job')+'</p>')
    parent = registration.get('parent') or registration.get('IL_parent') or registration.get('selected_parent')
    sections.append('<details open><summary>本轮父模型与数据登记</summary><pre>'+esc(json.dumps({'parent':parent, 'data':registration.get('data', registration.get('data_policy')), 'evaluation_protocol':registration.get('evaluation_protocol')},ensure_ascii=False,indent=2))+'</pre></details></section>')
    run_rows = []
    process_rows = []
    worker_rows = []
    for run in runs:
        run_id = run.get('id', '未命名')
        target = run.get('stage_budget_updates', run.get('target_updates'))
        updates = run.get('updates')
        progress_bar = ''
        if finite(updates) and finite(target) and target > 0:
            progress_bar = f'<progress max="{int(target)}" value="{min(updates,target)}"></progress>'
        status = '登记：'+esc(run.get('registry_status') or run.get('status', '待派发'))
        owner=run.get('owner_receipt',{})
        if owner:
            status+='<small>owner：'+esc(owner.get('status','状态待记录'))+'</small>'
        worker_status=run.get('worker_receipt_status')
        if worker_status:
            status+='<small>worker 收据：'+esc(worker_status)
            if run.get('worker_returncode') is not None:status+=' / code '+esc(run['worker_returncode'])
            status+='</small>'
        actual = '<small>'+esc(run.get('journal_state', '待观测'))+'</small>'
        run_rows.append([f'<b>{esc(run_id)}</b><small>{esc(run.get("algorithm","待登记"))}</small>',
                         esc(run.get('data_policy','待登记')), status+actual,
                         number(updates)+' / '+number(target)+progress_bar,
                         number(run.get('new_candidates')), esc(run.get('host') or '待分配'),
                         relative_link(root, run.get('checkpoint') or run.get('root'), '检查点 / run')])
        ratio_value=fmt(run.get('recent_clipfrac'),2,percent=True) if run.get('algorithm')!='nft' else '不适用（NFT）'
        blocked_value=fmt(run.get('recent_ppo_blocked_fraction'),2,percent=True) if run.get('algorithm')!='nft' else '不适用（NFT）'
        gradient=run.get('recent_gradient')
        gradient_value=(f'{gradient:.3e}' if finite(gradient) and 0<abs(gradient)<1e-4 else fmt(gradient,5))
        process_rows.append([esc(run_id), esc(str(run.get('rank_updates', '待记录'))),
                             fmt(run.get('recent_reward'),4,percent=True),ratio_value,blocked_value,
                             gradient_value,number(run.get('collections')),
                             utc(run.get('evidence_mtime')) if run.get('evidence_mtime') else '待记录'])
        identity=run.get('worker_receipt_identity',{})
        ids=identity.get('gpu_uuids') or []
        process_id=identity.get('worker_pid',identity.get('pid'))
        worker_rows.append([esc(run_id),esc(owner.get('status','owner收据待生成')),
                            esc(worker_status or '无 worker 启动收据')+('<small>returncode '+esc(run.get('worker_returncode'))+'</small>' if run.get('worker_returncode') is not None else ''),
                            esc(identity.get('host') or run.get('host') or '待分配')+'<small>PID '+number(process_id)+'</small>',
                            number(len(ids))+'<small>'+esc(', '.join(ids))+'</small>' if ids else '待核验',
                            relative_link(root,run.get('owner_state'),'owner')+' · '+relative_link(root,run.get('job'),'job / logs')+' · '+relative_link(root,run.get('source'),'source')])
    sections.append('<section><h2>本轮实验进度</h2>'+table(['实验 / 算法','数据策略','状态与证据','更新 / 阶段预算','新评分候选','主机','来源'],run_rows,empty='实验代码、父模型与资源核验中；尚未登记新训练，不宣称已经启动。'))
    sections.append('<h3>worker 与 owner 证据</h3><p class="muted">DISPATCHED 只表示登记派发。RUNNING.json 是启动收据，单独不能证明进程此刻存活；原生更新日志另外展示。RESULT.json 优先于历史 RUNNING.json；退出零也只表示该 worker 阶段结束，不自动代表 50k 完成。</p>'+table(['实验','owner阶段','worker收据','收据中的主机 / PID','GPU UUID数 / 明细','来源'],worker_rows))
    sections.append('<h3>训练过程观测</h3><p class="muted">更新数只在全部登记 rank 均有日志时取共同下界；候选数是日志里的全局计数，不再乘 rank 或 inner 复用次数。下表均值来自每个 rank 最多 64 条尾部记录，不代表固定场景对照。PPO方向阻断是给定PPO阈值的诊断，CISPO不会按此掩码丢弃梯度。</p>'+table(['实验','各 rank 最后更新','采样 PDMS 均值','ratio 越界 %','PPO方向阻断 %','梯度范数均值','fresh 全局批次数','最新日志时间'],process_rows)+'</section>')
    grouped = defaultdict(list)
    for row in complete:
        population_id=str(row.get('token_sha256') or 'count-only')[:8]
        grouped[f'{row["cohort"]} · {number(row.get("expected_scenes"))} scenes · {population_id}'].append(row)
    sections.append('<section><h2>本轮评测结果</h2><p>只有 <code>COMPLETE</code>、完整人口计数相符且 invalid=0 的结果才发布 PDMS；未完成项仅列进度。TRAIN_DIAGNOSTIC 的 1,024 / 4,096 场景是固定物理 log 分层 panel，只代表该声明子集，不是全量 TRAIN 的场景加权均分估计，也不自动构成 heldout。</p>')
    result_table = []
    for row in sorted(results,key=lambda r:(str(r.get('cohort')),str(r.get('arm')),r.get('updates',0) or 0)):
        point_updates=('IL '+number(row.get('IL_updates'))+' / RL 0' if row.get('evaluation_kind')=='IL_BASELINE' else number(row.get('updates')))
        point_label=esc(row.get('arm',row.get('label','待登记')))
        if row.get('evaluation_kind')=='IL_BASELINE':point_label+='<small>父模型基线，非 RL 收益</small>'
        result_table.append([point_label,point_updates,esc(row['cohort']),
                             esc(row['completion_label']),number(row.get('scenes'))+' / '+number(row.get('expected_scenes')),
                             '<b>'+fmt(row['display_PDMS'],6)+'</b>' if row['publishable_complete'] else '—',
                             relative_link(root,row.get('evidence') or row.get('summary'))])
    sections.append(table(['实验','更新','人口','完整性','已完成 / 应有场景','PDMS','证据'],result_table,empty='本轮尚无完整评测结果。历史成绩见下方，不能冒充本轮结果。'))
    for cohort, rows in grouped.items():
        series = defaultdict(list)
        if cohort.startswith('NAVTEST') and finite(pinned_reference) and not any(r.get('evaluation_kind')=='IL_BASELINE' for r in rows):
            series['IL70 pinned reference'].append((0,pinned_reference))
        for row in rows:
            if finite(row.get('rl_updates_for_plot')):
                series[str(row.get('arm','未命名'))].append((row['rl_updates_for_plot'],row['PDMS']))
        sections.append(chart([{'label':k,'points':v} for k,v in series.items()],title='本轮完整评测 · '+cohort,xlabel='RL optimizer updates (IL baseline = 0)',ylabel='PDMS (0–100)'))
    sections.append('</section>')
    narrative=[]
    observed=[r for r in runs if finite(r.get('updates'))]
    if observed:
        narrative.append(f'已有 {len(observed)} 条实验观察到各登记 rank 的原生更新日志。资格训练、阶段训练和完整预算须按各实验身份区分。')
    else:
        narrative.append('本轮尚无已汇总的全 rank 原生训练更新；准备与派发状态不等同于模型已经学习。')
    if complete:
        for cohort, rows in grouped.items():
            policy_rows=[r for r in rows if r.get('evaluation_kind')!='IL_BASELINE']
            baseline_rows=[r for r in rows if r.get('evaluation_kind')=='IL_BASELINE']
            if not policy_rows:
                baseline=baseline_rows[0]
                narrative.append(f'{cohort} 已完成父模型 IL 基线：{baseline["PDMS"]:.6f}；尚无同人口的本轮 RL 完整结果，这个分数不代表 RL 已提高性能。')
            else:
                best=max(policy_rows,key=lambda r:r['PDMS'])
                statement=f'{cohort} 已有 {len(policy_rows)} 个本轮 RL 完整评测点；当前已测最高 {best.get("arm","未命名")} / RL update {best.get("rl_updates_for_plot","待记录")}：{best["PDMS"]:.6f}。'
                if baseline_rows:statement+=f' 相对同人口 IL 基线为 {best["PDMS"]-baseline_rows[0]["PDMS"]:+.6f} 点。'
                statement+='这是已测点比较，不能据此宣称训练全过程最高或跨 seed 的稳定优势。'
                narrative.append(statement)
    else:
        narrative.append('尚无本轮完整评测分数，当前不能宣布算法、数据策略或最终性能已提高。')
    for result in complete:
        paired=result.get('paired',{})
        if isinstance(paired,dict) and isinstance(paired.get('CI95'),list) and len(paired['CI95'])==2:
            low,high=paired['CI95']
            if finite(low) and finite(high):
                verdict='区间大于零' if low>0 else '区间小于零' if high<0 else '区间跨零'
                narrative.append(f'{result.get("arm","未命名")} / {result["cohort"]} 的已登记配对区间为 [{100*low:+.4f}, {100*high:+.4f}] PDMS 点，{verdict}；仅覆盖该配对场景统计，不包含训练 seed 不确定性。')
    sections.append('<section><h2>阶段分析</h2><ul>'+''.join('<li>'+esc(v)+'</li>' for v in narrative)+'</ul><p>下一阶段重点区分：同场景 SDE/ODE 的采样失配、训练人口覆盖与成功行为退化、以及正负优势和逐步梯度的更新机制。健康臂继续原定预算，完整评测和成本收据用于分析及安排资源优先级。</p></section>')
    nft_analysis=snapshot.get('nft_first_stage_paired_analysis',{})
    paired=nft_analysis.get('paired',{})
    if nft_analysis.get('status')=='COMPLETE' and paired.get('scene_denominator')==12146:
        interval=nft_analysis.get('CI95_PDMS',[])
        interval_text='['+', '.join(f'{v:+.6f}' for v in interval)+']' if len(interval)==2 and all(finite(v) for v in interval) else '待记录'
        summary_rows=[['固定 IL70 父模型',fmt(nft_analysis.get('reference_PDMS'),6)],
            ['NFT · RL 1,000 更新',fmt(nft_analysis.get('RL_PDMS'),6)],
            ['配对差值（NFT − IL70）',fmt(nft_analysis.get('delta_PDMS'),6)],
            ['配对 95% 区间',interval_text],
            ['改善贡献 G+ / 损失贡献 G−',fmt(nft_analysis.get('G_plus_PDMS'),6)+' / '+fmt(nft_analysis.get('G_minus_PDMS'),6)]]
        sections.append('<section><h2>NFT 首个 1k 阶段：完整 Navtest 配对诊断</h2><p>同一完整人口 <b>12,146</b> 场景，与固定 IL70 父模型配对；按 '+number(paired.get('physical_groups'))+' 个物理 log 重采样 '+number(paired.get('bootstrap'))+' 次，bootstrap seed '+number(paired.get('bootstrap_seed'))+'。下表分数、差值和区间均为 PDMS 点（0–100）；G+、G−分别是正向与负向场景变化对全人口均分的贡献。</p>'+table(['统计量','PDMS 点'],summary_rows))
        if len(interval)==2 and all(finite(v) for v in interval):
            if interval[0]<=0<=interval[1]:
                verdict='配对区间包含零，当前不支持“显著退化”的表述，也不证明性能等效。'
            elif interval[1]<0:
                verdict='这一次场景配对区间低于零，提示该阶段存在下降；仍须区分场景统计与训练 seed 的不确定性。'
            else:
                verdict='这一次场景配对区间高于零，提示该阶段存在改善；仍须区分场景统计与训练 seed 的不确定性。'
            continuation_note=('1k / 5k 诊断不作硬淘汰，健康实验臂按当前持续研究政策继续原定预算。' if ongoing_research else '健康实验臂按登记的阶段预算继续研究。')
            sections.append('<p class="note">'+verdict+' 这是 <b>1k 早期观察</b>，不构成 NFT 最终优劣结论；'+continuation_note+'</p>')
        transition_rows=[['修复原零分',number(paired.get('repaired_zero'))+' / '+number(paired.get('baseline_zero_denominator'))],
            ['原非零新增零分',number(paired.get('new_zero'))+' / '+number(paired.get('baseline_nonzero_denominator'))],
            ['原高分（≥80）降为零分',number(paired.get('high_to_zero'))+' / '+number(paired.get('baseline_high_denominator'))]]
        sections.append('<h3>失败修复与安全变化</h3>'+table(['分数转移','场景数 / 对应 IL 人口'],transition_rows))
        names={'no_at_fault_collisions':'无责任碰撞 NC','drivable_area_compliance':'可行驶区域 DAC',
            'time_to_collision_within_bound':'碰撞时间 TTC','ego_progress':'行驶进度 EP',
            'comfort':'舒适性','driving_direction_compliance':'行驶方向合规'}
        safety_rows=[]
        for key,title in names.items():
            values=nft_analysis.get('safety_metrics',{}).get(key,{})
            if not values:continue
            safety_rows.append([title,fmt(values.get('IL_mean_percent'),6),fmt(values.get('NFT_mean_percent'),6),
                fmt(values.get('delta_percentage_points'),6),
                number(values.get('new_failure'))+' / '+number(values.get('baseline_pass')) if 'new_failure' in values else '—',
                number(values.get('repaired_failure'))+' / '+number(values.get('baseline_failure')) if 'repaired_failure' in values else '—'])
        sections.append(table(['指标（0–100）','IL70','NFT 1k','变化：百分点','新增失败 / 原通过','修复 / 原失败'],safety_rows))
        categories=nft_analysis.get('scene_categories',{})
        safety_down=categories.get('score_down_with_new_NC_DAC_TTC_violation',{})
        ep_down=categories.get('score_down_EP_down_safety_unchanged',{})
        total_loss=nft_analysis.get('G_minus_PDMS')
        if safety_down and ep_down and finite(total_loss) and total_loss<0:
            contribution=safety_down.get('G_minus_PDMS')
            share=100*contribution/total_loss if finite(contribution) else None
            sections.append('<p>损失结构：'+number(safety_down.get('scenes'))+' 个分数下降且伴随新增 NC/DAC/TTC 违规的场景贡献 '+fmt(contribution,6)+' 点，约占 G− 的 '+fmt(share,1)+'%；另有 '+number(ep_down.get('scenes'))+' 个安全指标未变、EP 降低的场景贡献 '+fmt(ep_down.get('G_minus_PDMS'),6)+' 点。后者解释了大部分下降场景的数量，前者贡献了主要分数损失；这是关联分解，未单独识别训练机制的因果作用。</p>')
        sections.append('<p class="muted">本分析仅用于完整 NAVTEST 的阶段诊断，不将这些场景 token 回流为训练名单；不修改已封存 SUMMARY，也不增加模型推理或评分。证据时间：'+esc(nft_analysis.get('utc','待记录'))+'。'+relative_link(root,str(root/'analysis'/'NFT_u001000_vs_IL70.json'),'完整配对分析与协议核对')+'</p></section>')
    reward_signal=snapshot.get('matched_reward_signal',{})
    if reward_signal.get('observations'):
        signal_rows=[]
        for arm,values in reward_signal['observations'].items():
            signal_rows.append([esc(arm),number(values.get('fresh_scene_groups')),
                number(values.get('reward_tied_groups')),fmt(values.get('reward_tied_fraction'),1,percent=True)+'%',
                number(values.get('tied_all_one')),number(values.get('tied_all_zero'))])
        sections.append('<section><h2>问题诊断：首 1,000 更新的奖励对比信号</h2><p>实际匹配首 1,000 次 optimizer 更新、全部 8 rank；每条路线各 2,000 个 fresh 场景组，每组只计一次，未将四次 inner 复用重复计数。</p>'+table(['路线','fresh 场景组','奖励同分组','同分比例','其中全 1','其中全 0'],signal_rows))
        sections.append('<p>NFT_FULL 的奖励同分组明显更多，其中大部分是全 1 组。这里的“同分”指缺少组内 reward 对比，<b>不表示总梯度为零</b>；NFT 仍可能通过 EMA 中心回拉产生梯度。</p><p class="note">这是算法族的过程观察：采样器、EMA behavior 和优化目标同时发生变化，不能归因为单一采样器因素。NFT 在固定 TRAIN1024 panel 的 +0.0069 点也不足以单独支持算法优劣结论。</p>')
        nft_test_complete=any(r.get('arm')=='NFT_FULL' and r['cohort'].upper() in ('NAVTEST','TEST','NAVTEST_V1') for r in complete)
        sections.append('<p>'+('完整 NFT Navtest 已列入上方结果；应结合完整分数和安全退化判断，不能由同分比例独自判定算法失效。' if nft_test_complete else '当前等待完整 NFT Navtest；不根据训练 panel 或同分比例预先宣称提分或算法失效。')+' '+relative_link(root,str(root/'analysis'/'first1000_reward_signal.json'),'实际首 1,000 更新诊断证据')+'</p></section>')
    training_series = defaultdict(list)
    for row in history:
        if finite(row.get('updates')) and finite(row.get('recent_reward')):
            training_series[row['id']].append((row['updates'],100*row['recent_reward']))
    sections.append('<section><h2>本轮过程曲线</h2><p class="muted">本图只读取 exploration.canonical_random_mean；SAFE_COMFORT 用于训练的塑形 rewards 不是 canonical PDMS，也不作为部署成绩。</p>'+chart([{'label':k,'points':v} for k,v in training_series.items()],title='训练随机候选分数 · 不作为部署成绩',xlabel='Common optimizer updates',ylabel='Sampled PDMS (0–100)')+'</section>')
    historical = snapshot['historical']
    historical_series = defaultdict(list)
    for row in historical.get('points',[]):
        if row.get('scenes') == 12146 and finite(row.get('updates')) and finite(row.get('PDMS')):
            historical_series[row['route']].append((row['updates'],row['PDMS']))
    historical_rows = []
    audit = snapshot['journal_audit'].get('runs',{})
    for arm, ps in historical_series.items():
        peak = max(ps,key=lambda p:p[1]); endpoint = next((y for x,y in ps if x==50000),None)
        a = audit.get(arm,{})
        fourth = a.get('by_inner',{}).get('4',{}).get('clipfrac',{}).get('mean')
        zero_pg = [v.get('zero_PG') for v in a.get('fresh_exploration_by_window',{}).values() if finite(v.get('zero_PG'))]
        historical_rows.append([esc(arm),fmt(endpoint,6),number(peak[0])+' / '+fmt(peak[1],6),
                                fmt(fourth,2,percent=True),f'{min(zero_pg)*100:.2f}–{max(zero_pg)*100:.2f}%' if zero_pg else '待记录'])
    sections.append('<section class="historical"><div class="tag">历史参考 · 非本轮新成绩</div><h2>上一轮 IL40k 起点的 World RL</h2><p>共同父模型 IL40k：87.972382；单候选 ODE K10 / shift5，完整 Navtest 12,146 场景，训练 seed42、推理 seed2101。已测最高点不等于全过程最高点。</p>'+table(['路线','固定 50k PDMS','已测最高：更新 / PDMS','第 4 次复用 ratio 越界','零 PG 组窗口范围'],historical_rows))
    sections.append(chart([{'label':k,'points':v} for k,v in historical_series.items()],title='历史完整 Navtest · IL40k 父模型',xlabel='RL optimizer updates',ylabel='PDMS (0–100)'))
    noise = snapshot['chain_audit'].get('physical_last_innovation_std',{})
    sections.append('<p>保存链诊断：最后一步创新噪声标准差，纵向 <b>'+fmt(noise.get('longitudinal_m'),3)+' m</b>、横向 <b>'+fmt(noise.get('lateral_m'),3)+' m</b>、航向 <b>'+fmt(noise.get('heading_rad'),4)+' rad</b>。这是训练 SDE 的采样特征，尚不能独立证明它造成 ODE 部署回落。</p><p>'+relative_link(root,str(HISTORICAL_CURVES),'完整历史曲线与配对统计')+' · '+relative_link(root,str(REVIEW_ROOT/'world_rl_journal_audit.json'),'120 万条更新日志统计')+' · '+relative_link(root,str(REVIEW_ROOT/'world_rl_archive_audit.json'),'768 份保存链诊断')+'</p></section>')
    sections.append('<section><h2>数据人口与采样阶段</h2>')
    if snapshot['data_audit']:
        audit=snapshot['data_audit']
        if 'full_NAVTRAIN_count' in audit:
            population_rows=[['全量 NAVTRAIN',number(audit.get('full_NAVTRAIN_count')),number(audit.get('full_NAVTRAIN_physical_logs'))+' 个物理 log'],
                             ['原始 native TRAIN',number(audit.get('original_native_TRAIN_count')),'全量数据来源的一部分'],
                             ['旧 DEV',number(audit.get('old_DEV_count')),'进入全量臂后是训练内诊断，不算 heldout'],
                             ['World 原 TRAIN',number(audit.get('world_TRAIN_count')),'先前物理 log 划分的人口'],
                             ['NAVTEST',number(audit.get('NAVTEST_count')),number(audit.get('NAVTEST_physical_logs'))+' 个独立物理 log，禁止进入训练'],
                             ['NAVTRAIN / NAVTEST 交集',number(audit.get('token_overlap'))+' tokens',number(audit.get('physical_log_overlap'))+' 个物理 log']]
            sections.append(table(['人口','场景数 / 交集','用途与边界'],population_rows))
            sections.append('<p class="note">全量臂实际人口为 <b>103,288</b>，旧 DEV 的 <b>1,696</b> 场景已在其中。另有按物理 log 预留的 RL_DEV 5,323 场景；只有使用排除这些场景的 <b>97,965</b> 训练臂，才可把该 RL_DEV 称为独立 RL heldout。全量 103,288 臂在同一人口上的评测只能作训练内诊断。</p>')
        sections.append('<details><summary>完整人口与资产审计</summary><pre>'+esc(json.dumps(audit,ensure_ascii=False,indent=2))+'</pre></details>')
    else:
        sections.append('<p class="muted">正在核对全量约 103k、当前 World TRAIN、旧 DEV、NAVTEST 及真实 metric cache 的人口关系。尚未把任何缺失场景静默排除或标为可训练。</p>')
    sections.append('<p>自适应选择须同时追踪当前低分、相对 IL 的退化和均匀覆盖；场景角色与每阶段名单应封存。选择依据来自合法训练人口的周期评测，不能把 Navtest token 纳入训练。</p></section>')
    resources = snapshot.get('active_resources',snapshot['resources'])
    resource_rows = []
    for row in resources:
        resource_rows.append([esc(row['host']),number(row.get('index')),esc(row.get('uuid') or '待记录'),
                              number(row.get('free_MiB',row.get('memory_free'))),number(row.get('utilization')),
                              number(row.get('reported_owners')),utc(row.get('inventory_at')) if row.get('inventory_at') else '待记录'])
    unique = len({r['uuid'] for r in resources if r.get('uuid')})
    withdrawal=snapshot.get('resource_withdrawal',{})
    hosts=withdrawal.get('remaining_hosts') or sorted({row['host'] for row in resources})
    historical_count=len({r['uuid'] for r in snapshot['resources'] if r.get('uuid')})
    sections.append('<section><h2>资源与等待原因</h2><p>当前授权范围为 <b>'+str(len(hosts))+'</b> 台主机，最近库存可识别 <b>'+str(unique)+'</b> 个唯一物理 GPU UUID；这不是全部空闲或本轮已部署的卡数。登记派发数、worker 启动收据数与真实更新证据分别列示。</p>')
    startup=snapshot.get('pipeline_startup',{})
    if startup.get('arms'):
        startup_arms={r['arm'] for r in startup['arms']}
        startup_hosts={r.get('host') for r in startup['arms'] if r.get('host')}
        model_ranks=sum(len(r.get('rank_updates',[])) for r in startup['arms'])
        training_uuids={gpu for run in runs if run['id'] in startup_arms
            for gpu in run.get('job_receipts',{}).get('LAUNCH.json',{}).get('gpu_uuids',[])}
        sections.append('<p class="note">七臂启动 / 接管快照 '+esc(startup.get('at','待记录'))+'：<b>'+number(model_ranks)+' 个模型 rank 进程</b>，分布在 <b>'+number(len(startup_hosts))+' 台训练主机 / '+number(len(training_uuids))+' 个唯一物理 GPU</b>；共享设备上的多个 rank 不重复计为物理卡。快照七臂全部 8 rank 均有更新且梯度有限；它是一次性证据，不是持续存活探测。其余主机既有任务与评测保留。'+relative_link(root,str(root/'STARTUP_SEVEN_PIPELINES_20261030.json'),'原始启动快照')+'</p>')
    evaluation_scheduling=snapshot.get('evaluation_scheduling',{})
    if evaluation_scheduling:
        sections.append('<p>新评测调度：NAVTEST 至少 '+number(evaluation_scheduling.get('minimum_new_NAVTEST_shards'))+'、最多 '+number(evaluation_scheduling.get('maximum_NAVTEST_shards'))+' shards；TRAIN 至少 '+number(evaluation_scheduling.get('minimum_new_TRAIN_shards'))+'、最多 '+number(evaluation_scheduling.get('maximum_TRAIN_shards'))+'。等待容量时训练继续，已发布 export 不因容量重试重复 hash；原生模型 restore 完整性校验保持。'+relative_link(root,str(root/'EVALUATION_SCHEDULING_POLICY.json'),'评测调度政策')+'</p>')
    if withdrawal:
        sections.append('<p class="note"><b>'+esc(withdrawal.get('withdrawn_host'))+' 已撤回，禁止新派发。</b> 收据时间 '+esc(withdrawal.get('utc','待记录'))+'；本轮在该主机已派发任务数 '+number(withdrawal.get('our_tasks_dispatched_on_host'))+'。撤回前库存曾含 '+str(historical_count)+' 个 UUID，已从下方当前授权设备表排除此主机。</p><p>'+relative_link(root,str(root/'resources'/'TSINGHUA_ZT2_WITHDRAWAL_20261030.json'),'撤回原始收据')+'</p>')
    sections.append('<p class="muted">库存时间是原始观测时间，不是报告刷新时间；旧库存不能证明设备目前仍空闲。新的准入由训练 controller 根据实际资源收据执行。</p>'+table(['授权主机','GPU','物理 UUID','库存空闲 MiB','库存利用率 %','库存 GPU owner 数','库存时间'],resource_rows,empty='资源清点中；不把未核验主机计作已部署。'))
    waits = state.get('waiting_reasons',state.get('resource_waits',[]))
    if isinstance(waits,list):
        waits=list(waits)
        for run in runs:
            owner=run.get('owner_receipt',{})
            reason=owner.get('waiting_reason',owner.get('wait_reason'))
            if reason:waits.append(dict(arm=run['id'],reason=reason))
    sections.append('<details open><summary>等待 / 不可用原因</summary><pre>'+esc(json.dumps(waits,ensure_ascii=False,indent=2) if waits else '尚无已登记等待原因；以资源和 worker 收据为准。')+'</pre></details></section>')
    throughput=snapshot.get('evaluation_throughput_snapshot',{})
    if throughput.get('evaluations'):
        throughput_rows=[]
        for item in throughput['evaluations']:
            if item.get('cohort')!='NAVTEST':continue
            throughput_rows.append([esc(item['label']),number(item.get('world')),
                number(item.get('scene_files'))+' / '+number(item.get('expected_scenes')),
                fmt(item.get('forward_seconds_per_new_scene'),3),fmt(item.get('scoring_seconds_per_new_scene'),3),
                fmt(item.get('scoring_share_of_loop_wall'),2,percent=True)+'%',
                fmt(item.get('completion_spread_seconds'),1)])
        sections.append('<section><h2>评测吞吐与阶段等待：一次性现状分析</h2><p>快照时间 <b>'+esc(throughput.get('captured_utc','待记录'))+'</b>；下表是已记录的 spec、worker 与原生 rank 计时收据，<b>不是当前实时设备状态</b>。仅列进度与耗时，不计算 partial 分数。</p>'+table(['完整人口评测','固定 rank 数','快照场景进度','前向秒 / 场景','scorer秒 / 场景','scorer占循环','首末 rank 完成差：秒'],throughput_rows))
        if snapshot.get('cispo_expansion'):
            sections.append('<p class="note">这是 CISPO 扩容前的旧快照；其中 world=1 的长尾风险已由原生场景边界退出、保留 1,661 个成功场景后扩为 16 workers 处理。当前评测进度看上方实时读取的收据，旧快照不作为当前仍只有单 worker 的证据。</p>')
        sections.append('<ul>'+''.join('<li>'+esc(item['finding'])+'</li>' for item in throughput.get('conclusions',[]) if item.get('id') in ('single_rank_navtest_tail','global_gate_wait','scorer_not_dominant','static_shard_heterogeneity'))+'</ul>')
        sections.append('<p class="note">优先修正诊断阶段的全局等待，让合法 TRAIN 反馈就绪的健康实验臂继续原定训练，完整 NAVTEST 异步完成；固定 world 与设备共享长尾优先于增加 scorer 并发。此处是该快照提出的诊断建议，实际执行以最新 controller 收据为准，未由报告重派或停止 worker。</p><p class="muted">前向计时不含 CPU 观察处理；其余循环开销尚未细分，不能把全部剩余耗时归为 CPU scorer。历史资源库存也不能证明设备当前空闲。'+relative_link(root,str(root/'analysis'/'evaluation_throughput_20261030.json'),'完整吞吐快照、逐 rank 证据与行动建议')+'</p></section>')
    normalization=snapshot.get('advantage_normalization',{})
    full_norm=normalization.get('arms',{}).get('PPO_FULL',{})
    if normalization.get('status')=='COMPLETE' and full_norm:
        norm_evidence=full_norm.get('normalization_evidence',{})
        scale=full_norm.get('raw_to_centered_scale',{})
        batch_registration=snapshot.get('batchstd_registration',{})
        batch_run=next((r for r in runs if r.get('id')=='PPO_BATCHSTD'),{})
        sections.append('<section><h2>优势归一化：现有实现与单因素实验</h2><p>实际首 1,000 更新诊断确认：基线<b>已经组内减均值，并除以全 8 rank 的 RAW reward std</b>，不是组内 std。已有技巧不重复登记为新增方法。</p>'+table(['口径','公式 / 证据'],[
            ['已存在的基线', '<code>'+esc(norm_evidence.get('current_formula','待记录'))+'</code>'],
            ['新臂唯一改变的分母' if batch_registration else '待探索的分母差异','<code>'+esc(norm_evidence.get('RFpp_baseline_candidate','待记录'))+'</code>'],
            ['新旧优势尺度倍率（TRAIN 观测）',fmt(scale.get('mean'),3)+' 均值；'+fmt(scale.get('min'),3)+'–'+fmt(scale.get('max'),3)+' 范围']]))
        sections.append('<p class="note">新方向是 REINFORCE++ 式的 <b>centered-advantages global std</b>：改变分母为全批已中心化优势的 std。当前 TRAIN 数据上尺度平均约 1.218 倍，这同时改变了更新尺度，不能提前解释为算法性能提升；已有 PPO_FULL 成绩不归因于它。</p>')
        if batch_registration:
            registered=batch_registration.get('arm',{})
            batch_scores=[v for v in complete if v.get('arm')=='PPO_BATCHSTD']
            sections.append('<p><b>PPO_BATCHSTD 已正式注册并派发</b>，以 PPO_FULL 为对照；FULL103288、IL70、8 rank、G8、4 inner、optimizer/LR/anchor 与 seed 保持一致。新 source '+relative_link(root,registered.get('source'),'v4_04a6798 runtime')+'；运行主机 '+esc(batch_run.get('host') or registered.get('host'))+' / '+esc(registered.get('lease','待记录'))+' lease。全 rank 共同观察更新 <b>'+number(batch_run.get('updates'))+'</b>，日志 rank '+number(batch_run.get('journal_ranks'))+' / '+number(batch_run.get('expected_ranks'))+'；尾部梯度范数均值 '+gradient_fmt(batch_run.get('recent_gradient'))+'。'+('已完整评测 '+number(len(batch_scores))+' 点，见结果表。' if batch_scores else '当前尚无该新臂完整评测成绩，不预先宣称提分。')+'</p><p>'+relative_link(root,str(root/'EXPERIMENT_PPO_BATCHSTD_REGISTRATION.json'),'正式单因素注册与 config 哈希')+' · '+relative_link(root,batch_run.get('job'),'原生训练 job')+'</p>')
        sections.append('<p>'+relative_link(root,str(root/'analysis'/'advantage_normalization_u001000.json'),'优势归一化实际诊断')+'；仅使用合法 TRAIN 银行与过程记录，NAVTEST 场景 token 不进入自适应训练 bank。</p></section>')
    comfort_registration=snapshot.get('comfort_registration',{})
    if comfort_registration:
        comfort_run=next((r for r in runs if r.get('id')=='PPO_SAFE_COMFORT'),{})
        registered=comfort_registration.get('arm',{})
        counter=snapshot.get('comfort_counterfactual',{})
        diagnostic=counter.get('stochastic_comfort_diagnostic',{})
        comfort_scores=[r for r in complete if r.get('arm')=='PPO_SAFE_COMFORT']
        sections.append('<section><h2>PPO_SAFE_COMFORT：训练奖励单因素实验</h2><p><b>已正式注册并开始原生训练</b>：'+esc(registered.get('host'))+' / '+esc(registered.get('lease'))+' lease，'+relative_link(root,registered.get('source'),'v5_15de3ab runtime')+'。全 rank 共同观察更新 '+number(comfort_run.get('updates'))+'，'+number(comfort_run.get('journal_ranks'))+' / '+number(comfort_run.get('expected_ranks'))+' rank；尾部梯度范数均值 '+gradient_fmt(comfort_run.get('recent_gradient'))+'。'+relative_link(root,str(root/'EXPERIMENT_PPO_SAFE_COMFORT_REGISTRATION.json'),'正式注册、config 哈希与对照')+'</p>')
        sections.append('<p class="note">该臂的<b>训练 reward 显式非 canonical</b>：仅当 NC=DAC=TTC=1 时，将训练奖励里的 Comfort 固定为 1，其余候选维持原 canonical reward。原始 <code>canonical_rewards</code> 与官方 feedback records 独立保存；保持原优势归一化方式，不叠加 centered-std 改动。部署仍使用相同 ODE 与官方 canonical scorer，训练塑形分不充当 PDMS。</p>')
        if diagnostic:
            pairs=diagnostic.get('pairwise_ranking',{})
            sections.append('<h3>提出实验的离线反事实证据</h3><p>原 TRAIN bank：'+number(counter.get('train_groups'))+' 组、'+number(counter.get('train_candidates'))+' 候选；NC/DAC/TTC 全过的候选中，'+number(diagnostic.get('safe_comfort_failures'))+' / '+number(diagnostic.get('safe_candidates'))+'（'+fmt(diagnostic.get('safe_comfort_failure_fraction'),1,percent=True)+'%）Comfort 失败。安全候选对中 EP / score 逆序 '+number(pairs.get('score_EP_discordant_pairs'))+' 对；Comfort 相同的 '+number(pairs.get('constant_comfort_distinct_EP_pairs'))+' 对中逆序为 '+number(pairs.get('constant_comfort_discordant_pairs'))+'。这些旧 SDE 样本统计不构成同 checkpoint 的 SDE / ODE 因果实验。</p>')
        variants=counter.get('variants',{})
        if variants:
            labels={'canonical':'原 canonical','small_safe_EP_bonus':'小 EP bonus（未安排新臂）','all_fixed_comfort1':'全部固定 Comfort=1（仅反事实）','PPO_SAFE_COMFORT':'安全门控 Comfort=1（已注册）'}
            rows=[]
            for key,values in variants.items():
                rows.append([labels.get(key,key),number(values.get('strict_old_winner_displaced'))+' / '+number(counter.get('train_groups')),
                    fmt(values.get('strict_old_winner_displaced_fraction'),1,percent=True)+'%',
                    number(values.get('current_positive_unsafe_candidates'))+' → '+number(values.get('new_positive_unsafe_candidates')),
                    number(values.get('unsafe_advantage_nonpositive_to_positive'))])
            sections.append(table(['旧 bank 离线变换','严格赢家更换','比例','unsafe正优势候选','新增unsafe正优势'],rows))
        gated=variants.get('PPO_SAFE_COMFORT',{})
        if gated:
            loss=gated.get('selected_raw_canonical_change_mean')
            sections.append('<p class="note"><b>离线反事实本身不能证明部署提分。</b>同一个旧 bank 中，按门控塑形重新选中的 SDE 候选，原 canonical 均分反而变化 '+fmt(loss,6)+'（0–1 reward；即 '+fmt(100*loss if finite(loss) else None,6)+' PDMS 点）。这不是新策略部署成绩。必须观察新模型 ODE 的官方 Comfort、安全与完整 PDMS，存在 Comfort 退化风险；离线赢家变化 25.8% 和 unsafe 正优势减少不能保证部署提升。</p>')
        sections.append('<p>'+('该臂已有 '+number(len(comfort_scores))+' 个完整评测点，见上方结果表。' if comfort_scores else '该臂尚无新的正式完整成绩。')+' 假设形成参考了既有 NAVTEST 诊断；本次数值反事实只使用 TRAIN，NAVTEST token 未用于训练选择。'+relative_link(root,str(root/'analysis'/'safe_comfort_counterfactual_u001000.json'),'离线反事实、风险与完整来源')+' · '+relative_link(root,str(root/'analysis'/'safe_EP_reward_u001000.json'),'基础奖励分解')+'</p></section>')
    first_match=snapshot.get('first_collection_match',{})
    if first_match:
        sections.append('<section><h2>新算法臂的起点公平性</h2><p>PPO_BATCHSTD 与 PPO_SAFE_COMFORT 对 PPO_FULL 的全部 8 rank，首个 fresh collection、更新前的 token、chain、old logp、raw canonical reward 与协议逐值完全相同；仅 CPU 读取既有记录，无新增 forward 或评分。这支持单因素实验从同一起点出发，不意味着后续 rollout 仍相同。'+relative_link(root,str(root/'analysis'/'ALGORITHM_ARMS_FIRST_COLLECTION_MATCH.json'),'首批匹配原始证据')+'</p></section>')
    failures = state.get('failures',state.get('failed_attempts',[]))
    if isinstance(failures,list):
        failures=list(failures)
        for run in runs:
            result=run.get('job_receipts',{}).get('RESULT.json',{})
            code=result.get('returncode',result.get('exit_code'))
            if result and (result.get('status')=='FAILED' or (code is not None and code!=0)):
                failures.append(dict(arm=run['id'],job=run.get('job'),worker_result=result))
        refusal=snapshot.get('cispo_prechild_refusal',{})
        if refusal.get('child_not_started') is True and refusal.get('returncode')==75:
            failures.append(dict(arm='CISPO_FULL',attempt='formal_to005000_attempt001',kind='PRE_CHILD_RESOURCE_REFUSAL',
                child_not_started=True,returncode=75,optimizer_updates_lost=0,
                receipt=str(root/'jobs'/'CISPO_FULL'/'formal_to005000_attempt001'/'RESULT.json'),reason=refusal.get('reason')))
    costs = state.get('costs',state.get('failed_costs'))
    sections.append('<section><h2>失败、成本与阶段决策</h2><p>失败尝试和物理开销保留，不用成功重试覆盖。未记录成本显示为未知，不补零；共享 GPU worker wall time 不等于独占 GPU 计算时间。</p><pre>'+esc(json.dumps({'failures':failures,'costs':costs,'decisions':state.get('decisions',[])},ensure_ascii=False,indent=2))+'</pre>')
    events = snapshot['events']
    sections.append('<h3>阶段事件</h3><pre>'+esc(json.dumps(events[-30:] if isinstance(events,list) else events,ensure_ascii=False,indent=2))+'</pre></section>')
    warnings = snapshot['warnings']
    sections.append('<section><h2>报告来源与可复核性</h2><p>本页仅做只读汇总；没有在渲染时训练、模型推理、评分或查询 GPU。渲染时间不代表原始证据时间。常规每 30 分钟生成；完整评测、阶段完成或重要异常通过 owner 状态文件事件提前触发，事件去重后更新，且不推迟常规节奏。事件监听不轮询训练日志。</p><p>'+relative_link(root,str(root/'reporting'/'REPORT_CONFIG.json'),'持久化报告更新偏好')+' · '+relative_link(root,str(root/'reporting'/'OWNER.json'),'唯一报告 watcher 与最近触发收据')+'</p>')
    if warnings:
        sections.append('<ul>'+''.join('<li>'+esc(w)+'</li>' for w in warnings)+'</ul>')
    source_rows = [[relative_link(root,r['path'],Path(r['path']).name),utc(r['mtime']),number(r['bytes']),'<code>'+r['sha256'][:16]+'…</code>'] for r in snapshot['inputs']]
    sections.append('<details><summary>输入文件时间与 SHA256</summary>'+table(['文件','修改时间','字节','SHA256 前缀'],source_rows)+'</details><p>'+relative_link(root,str(root/'reporting'/'report_snapshot.json'),'本次派生快照')+' · '+relative_link(root,str(root/'reporting'/'render_report.py'),'报告生成器源码')+' · '+relative_link(root,str(REVIEW_ROOT/'literature_and_recommendations.json'),'21 篇论文与建议索引')+'</p></section>')
    css = '''
    :root{color-scheme:light;--ink:#172b43;--muted:#617084;--line:#dce5ef;--blue:#1769aa}*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:var(--ink);font:15px/1.7 system-ui,-apple-system,"Noto Sans SC",sans-serif}header{background:#102d4a;color:white;padding:38px max(24px,calc((100vw - 1260px)/2));border-bottom:5px solid #24b59d}h1{font-size:29px;line-height:1.3;margin:0 0 12px}header p{margin:6px 0;color:#d7e3ef}main{max-width:1308px;margin:24px auto;padding:0 24px 60px}section{margin:20px 0;background:#fff;border:1px solid var(--line);border-radius:14px;padding:25px;box-shadow:0 4px 20px #13324b05}h2{font-size:21px;margin:0 0 15px}h3{font-size:17px;margin:24px 0 10px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:15px}.card{background:white;border-radius:12px;border:1px solid var(--line);padding:20px}.card span{display:block;color:var(--muted);font-size:13px}.card strong{display:block;font-size:29px;margin-top:7px}.priorities{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.priorities div{padding:16px;background:#f3f7fb;border-radius:9px}.priorities p{margin:6px 0 0}.note{padding:14px 17px;border-left:4px solid #23a890;background:#eff9f6}.muted,small{color:var(--muted)}small{display:block;font-size:12px;max-width:300px}a{color:#1769aa;text-decoration:none}a:hover{text-decoration:underline}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:11px 10px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}th{white-space:nowrap;background:#f1f5f9;font-weight:600}td{overflow-wrap:anywhere}code{font:12px/1.5 ui-monospace,monospace;overflow-wrap:anywhere}pre{font:12px/1.65 ui-monospace,monospace;background:#f4f7fa;padding:16px;border-radius:8px;white-space:pre-wrap;overflow-wrap:anywhere;max-height:420px;overflow:auto}details{margin-top:12px}summary{cursor:pointer;color:#315779}progress{display:block;width:130px;height:8px;margin-top:8px;accent-color:#1769aa}svg{display:block;width:100%;height:auto;margin-top:20px;font-family:system-ui,sans-serif}.axis{fill:#63758b;font-size:12px}.plot-title{fill:#172b43;font-size:16px;font-weight:600}.chart-empty{padding:32px;text-align:center;background:#f6f8fb;border:1px dashed #cbd8e5;border-radius:10px;color:var(--muted);margin-top:15px}.tag{font-size:12px;display:inline-block;padding:3px 10px;background:#fff3d9;color:#865c0f;border-radius:99px;margin-bottom:12px}.historical{border-left:5px solid #dab355}footer{text-align:center;color:var(--muted);font-size:12px;margin-top:25px}@media(max-width:850px){.cards{grid-template-columns:repeat(2,1fr)}.priorities{grid-template-columns:1fr}header{padding:28px 22px}main{padding:0 12px 30px}section{padding:18px}.card strong{font-size:23px}}'''
    return '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="1800"><title>Qwen-World RL 优化实验 · 持续报告</title><style>'+css+'</style></head><body><header><h1>Qwen-World RL 优化实验</h1><p>训练数据 · 自适应场景 · ODE 对齐 · 策略优化 · 行为保持</p><p>报告生成：'+esc(snapshot['generated_utc'])+'　｜　最新输入文件：'+esc(utc(newest) if newest else '待记录')+'</p></header><main><div class="cards">'+''.join('<div class="card"><span>'+esc(k)+'</span><strong>'+esc(v)+'</strong></div>' for k,v in cards)+'</div>'+''.join(sections)+'<footer>固定路径：'+esc(root/'report.html')+' · 本轮报告不改动训练或评测状态</footer></main></body></html>'


def main() -> None:
    global ARTIFACT_ROOT
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--once', action='store_true', help='Render one snapshot and exit; no daemon is launched.')
    parser.add_argument('--watch', action='store_true', help='Keep a single reporting owner; first render is immediate.')
    parser.add_argument('--interval', type=int, default=1800, help='Seconds between reporting reads in watch mode (default 1800).')
    args = parser.parse_args()
    if args.once and args.watch:
        parser.error('--once and --watch are mutually exclusive')
    if args.interval<60:
        parser.error('--interval must be at least 60 seconds; default observation cadence is 1800')
    root = args.root.resolve()
    ARTIFACT_ROOT=root
    os.environ.setdefault('MPLCONFIGDIR',str(root/'reporting'/'mpl-cache'))
    (root/'reporting').mkdir(parents=True,exist_ok=True)
    config_path=root/'reporting'/'REPORT_CONFIG.json'
    try:config=json.loads(config_path.read_text())
    except (OSError,ValueError):config={}
    config.update(schema='world_rl_report_config_v1',
        user_preference='HTML 常规30分钟；完整阶段结果可立即更新',
        regular_interval_seconds=args.interval,event_backend='linux_inotify',
        immediate_event_types=['evaluation_complete','training_stage_complete','important_exception',
            'stage_selection_complete','performance_selection_complete','campaign_complete','stage_policy_amended'],
        debounce_seconds=1.0,journal_polling=False,event_updates_reset_regular_deadline=False,
        administrative_event_validation_is_scientific_result=False)
    config.update(research_continuation_preference='1k/5k诊断不硬淘汰；健康臂继续原预算；单臂故障不终止整体；TRAIN反馈不等Test',
        training_seed=42,inference_seed=2101,multiple_training_seeds_requested=False,
        resource_preference='高效并行使用授权资源，修正运营规划错误后继续研究')
    atomic_json(config_path,config)
    def cycle():
        with (root/'reporting'/'render.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            snapshot=load_snapshot(root)
            history=update_history(root,snapshot)
            document=render(root,snapshot,history)
            atomic_json(root/'reporting'/'report_snapshot.json',snapshot)
            atomic_text(root/'report.html',document)
        record={'status':'RENDERED','report':str(root/'report.html'),'runs':len(snapshot['runs']),
                'complete_results':sum(r['publishable_complete'] for r in snapshot['results']),
                'warnings':snapshot['warnings'],'generated_utc':snapshot['generated_utc']}
        print(json.dumps(record,ensure_ascii=False),flush=True)
        return record
    if not args.watch:
        cycle();return
    owner_lock=(root/'reporting'/'watch.lock').open('a')
    try:
        fcntl.flock(owner_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('A report watch owner already holds the lock; do not start a duplicate')
    stopped=threading.Event()
    from report_events import OwnerStateEvents
    event_source=OwnerStateEvents(root,atomic_json)
    def request_stop(signum,frame):
        stopped.set()
        event_source.wake()
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    owner=dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
               started_utc=utc(),root=str(root),interval_seconds=args.interval,
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),status='RUNNING',
               event_backend='linux_inotify',journal_polling=False,
               initial_state_signatures=event_source.initial_signatures)
    atomic_json(root/'reporting'/'OWNER.json',owner)
    next_regular=time.monotonic()
    pending_trigger=dict(kind='startup',at=utc())
    pending_events=[]
    event_deadline=None
    try:
        while not stopped.is_set():
            now=time.monotonic()
            if pending_trigger is None:
                if now>=next_regular:
                    pending_trigger=dict(kind='regular_interval',at=utc(),events=pending_events)
                    pending_events=[];event_deadline=None
                elif event_deadline is not None and now>=event_deadline:
                    pending_trigger=dict(kind='owner_state_event',at=utc(),events=pending_events)
                    pending_events=[];event_deadline=None
                else:
                    deadline=min(next_regular,event_deadline if event_deadline is not None else next_regular)
                    new_events=event_source.wait(deadline-now)
                    if new_events:
                        pending_events.extend(new_events)
                        if event_deadline is None:event_deadline=time.monotonic()+config['debounce_seconds']
                    continue
            trigger=pending_trigger
            pending_trigger=None
            if trigger['kind'] in ('startup','regular_interval'):
                # Event renders never postpone the fixed regular schedule.
                while next_regular<=now:next_regular+=args.interval
            owner.update(last_trigger=trigger,next_regular_utc=utc(time.time()+next_regular-time.monotonic()))
            try:
                current_source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                if current_source_sha!=owner['source_sha256']:
                    # Keep this unique watcher and lock while a one-shot child
                    # loads the current renderer. This is not another watcher
                    # or a training/evaluation controller.
                    child=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--once'],
                        text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=False)
                    if child.returncode:
                        raise RuntimeError('Updated report renderer failed: '+child.stdout[-3000:])
                    record=None
                    for line in reversed(child.stdout.splitlines()):
                        try:parsed=json.loads(line)
                        except ValueError:continue
                        if isinstance(parsed,dict) and parsed.get('status')=='RENDERED':record=parsed;break
                    if record is None:raise RuntimeError('Updated renderer produced no completion receipt')
                    print(json.dumps(record,ensure_ascii=False),flush=True)
                    owner['renderer_source_sha256']=current_source_sha
                else:
                    record=cycle()
                owner.update(status='RUNNING',last_render=record,last_read_unix=time.time())
                owner['last_trigger']['rendered_utc']=record['generated_utc']
                owner['event_render_count']=owner.get('event_render_count',0)+(trigger['kind']=='owner_state_event')
            except BaseException as error:
                owner.update(status='REPORT_RENDER_FAILED',error=traceback.format_exc(),last_read_unix=time.time())
                print(json.dumps({'status':'REPORT_RENDER_FAILED','error':str(error)},ensure_ascii=False),flush=True)
            atomic_json(root/'reporting'/'OWNER.json',owner)
    finally:
        owner.update(status='STOPPED',ended_utc=utc())
        atomic_json(root/'reporting'/'OWNER.json',owner)
        event_source.close()
        fcntl.flock(owner_lock,fcntl.LOCK_UN);owner_lock.close()


if __name__ == '__main__':
    main()
