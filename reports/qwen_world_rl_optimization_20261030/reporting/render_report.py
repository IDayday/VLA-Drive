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
    train4096_state=read_json(root/'owners'/'TRAIN4096'/'STATE.json',warnings,inputs) or {}
    result_rows+=evaluation_entries(train4096_state)
    additional_probes=read_json(root/'ADDITIONAL_NAVTEST_PROBES.json',warnings,inputs) or {}
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
    portfolio=read_json(root/'analysis'/'PORTFOLIO_10K_PAIRED.json',warnings,inputs) or {}
    matched10=dict(portfolio,contrasts={})
    for key,value in portfolio.get('contrasts',{}).items():
        target=value.get('target','');baseline=value.get('baseline','')
        if target.endswith('_u010000_NAVTEST') and baseline.endswith('_u010000_NAVTEST'):
            matched10['contrasts'][target[:-len('_u010000_NAVTEST')]+'_minus_'+baseline[:-len('_u010000_NAVTEST')]]=value
    decisions={};stopped_receipts={}
    for path in (root/'owners'/'CAMPAIGN'/'arm_decisions').glob('*.json'):
        value=read_json(path,warnings,inputs) or {}
        if value.get('arm'):
            (stopped_receipts if path.name.endswith('.STOPPED.json') else decisions)[value['arm']]=value
    adaptive_process=read_json(root/'analysis'/'ADAPTIVE_TRAIN_PROCESS_20261030.json',warnings,inputs) or {}
    train4096_diagnosis=read_json(root/'analysis'/'TRAIN4096_DIAGNOSIS.json',warnings,inputs) or {}
    matched12=read_json(root/'analysis'/'MATCHED_12K_PPO_CONTRASTS.json',warnings,inputs) or {}
    allocation_policy=read_json(root/'RESEARCH_ALLOCATION_POLICY_20261030.json',warnings,inputs) or {}
    portfolio_recommendations=read_json(root/'analysis'/'PORTFOLIO_10K_RECOMMENDATIONS.json',warnings,inputs) or {}
    process10=read_json(root/'analysis'/'PROCESS_DIAGNOSTIC_10K.json',warnings,inputs) or {}
    clamp10=read_json(root/'analysis'/'ADV_CLAMP_NEAR10K.json',warnings,inputs) or {}
    anchor40_registration=read_json(root/'EXPERIMENT_PPO_ANCHOR40_REGISTRATION.json',warnings,inputs) or {}
    futility_input=read_json(root/'analysis'/'NFT_FUTILITY_REVIEW_20261030.json',warnings,inputs) or {}
    futility={k:v for k,v in futility_input.items() if k!='results'}
    matched_contrasts=read_json(root/'analysis'/'MATCHED_1K_PPO_CONTRASTS.json',warnings,inputs) or {}
    capacity_policy=read_json(root/'RESOURCE_CAPACITY_POLICY.json',warnings,inputs) or {}
    capacity_adoption=read_json(root/'owners'/'CAMPAIGN'/'resource_capacity_adoption_20261030'/'RECEIPT.json',warnings,inputs) or {}
    pending_additions=read_json(root/'reporting'/'PENDING_RESOURCE_ADDITIONS.json',warnings,inputs) or {}
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
    authorization_pointer=read_json(root/'resources'/'LATEST_RESOURCE_AUTHORIZATION.json',warnings,inputs) or {}
    authorization_path=authorization_pointer.get('record')
    authorization=(read_json(Path(authorization_path),warnings,inputs,optional=False) or {}) if authorization_path else {}
    migration_result_path=authorization_pointer.get('migration_result') or authorization_pointer.get('previous_withdrawal_migration_result')
    migration_result=(read_json(Path(migration_result_path),warnings,inputs,optional=False) or {}) if migration_result_path else {}
    addition_pointer=read_json(root/'resources'/'LATEST_RESOURCE_ADDITION.json',warnings,inputs) or {}
    addition_path=addition_pointer.get('request')
    addition=read_json(Path(addition_path),warnings,inputs) or {} if addition_path else {}
    addition_folder=Path(addition_path).parent if addition_path else None
    deployment_path=addition_pointer.get('deployment_status')
    addition_deployment=read_json(Path(deployment_path),warnings,inputs) or {} if deployment_path else {}
    preferred_path=addition_deployment.get('preferred_routes')
    addition_preferred=read_json(Path(preferred_path),warnings,inputs) or {} if preferred_path else {}
    addition_capacity=read_json(addition_folder/'INITIAL_CAPACITY.json',warnings,inputs) or {} if addition_folder else {}
    addition_second=read_json(addition_folder/'ZT2_CAPACITY_SECOND_SAMPLE.json',warnings,inputs) or {} if addition_folder else {}
    addition_dispatch=read_json(addition_folder/'STAGE1_FULL_BATCHSTD_DISPATCHES.json',warnings,inputs) or {} if addition_folder else {}
    collected_input=read_json(addition_folder/'STAGE1_FULL_BATCHSTD_COLLECTED.json',warnings,inputs) or {} if addition_folder else {}
    addition_collected={k:v for k,v in collected_input.items() if k!='moves'}
    addition_collected['moves']={arm:{key:v.get(key) for key in ('native','exact_old_transport_exited','new_host','new_lease','root')}
        for arm,v in collected_input.get('moves',{}).items()}
    withdrawal_receipts=[]
    for path in sorted((root/'resources').glob('*WITHDRAWAL*.json')):
        receipt=read_json(path,warnings,inputs) or {}
        if receipt.get('withdrawn_host') and receipt.get('new_dispatch_allowed') is False:
            withdrawal_receipts.append(dict(receipt,path=str(path)))
    withdrawn_hosts={receipt['withdrawn_host'] for receipt in withdrawal_receipts}
    if withdrawal.get('withdrawn_host'):withdrawn_hosts.add(withdrawal['withdrawn_host'])
    resources=flatten_resources(resource_input)
    if addition_capacity:
        rows={(r['host'],r.get('uuid')):r for r in resources}
        for row in flatten_resources(addition_capacity):
            row['initial_admission']='初次检查合格'
            rows[(row['host'],row.get('uuid'))]=row
        second_host=addition_second.get('host')
        second_inventory=addition_second.get('inventory',{})
        issues=addition_second.get('issues',[])
        for row in flatten_resources({'hosts':{second_host:second_inventory}}) if second_host else []:
            issue=issues[row['index']] if isinstance(row.get('index'),int) and row['index']<len(issues) else None
            row['initial_admission']='待清：'+issue if issue else '初次检查合格'
            rows[(row['host'],row.get('uuid'))]=row
        resources=list(rows.values())
    current_authorized_hosts=sorted({host for key in ('permanent_hosts','temporary_hosts','unbounded_temporary_hosts')
        for host in authorization.get(key,[]) if isinstance(host,str)}-withdrawn_hosts) if authorization else None
    remaining=current_authorized_hosts if current_authorized_hosts is not None else withdrawal.get('remaining_hosts')
    if isinstance(remaining,list):remaining=[host for host in remaining if host not in withdrawn_hosts]
    active_resources=([row for row in resources if row['host'] in remaining and row['host'] not in withdrawn_hosts]
                      if isinstance(remaining,list) else [row for row in resources if row['host'] not in withdrawn_hosts])
    return dict(generated_utc=utc(), generated_unix=datetime.now(timezone.utc).timestamp(),
                registration=registration, state=state,campaign_state=campaign, runs=runs, results=results,
                resources=resources,active_resources=active_resources,resource_raw=resource_input,
                resource_withdrawal=withdrawal,
                resource_authorization=authorization,resource_authorization_path=authorization_path,
                resource_authorization_pointer=authorization_pointer,current_authorized_hosts=remaining,
                host_withdrawal_migration=migration_result,host_withdrawal_migration_path=migration_result_path,
                resource_addition=addition,resource_addition_path=addition_path,
                resource_addition_capacity=addition_capacity,resource_addition_second=addition_second,
                resource_addition_dispatches=addition_dispatch,resource_addition_collected=addition_collected,
                resource_addition_deployment=addition_deployment,resource_addition_deployment_path=deployment_path,
                resource_addition_preferred_routes=addition_preferred,
                resource_withdrawals=withdrawal_receipts,withdrawn_hosts=sorted(withdrawn_hosts),
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
                editorial_analysis=dict(matched_10k=matched10,matched_10k_path=str(root/'analysis'/'PORTFOLIO_10K_PAIRED.json'),
                    decisions=decisions,stopped_receipts=stopped_receipts,nft_decision=decisions.get('NFT_FULL',{}),
                    nft_stopped=stopped_receipts.get('NFT_FULL',{}),adaptive_process=adaptive_process,
                    nft_comparisons={key:value for key,value in portfolio.get('contrasts',{}).items() if value.get('target','').startswith('NFT_FULL')},
                    allocation_policy=allocation_policy,nft_futility=futility,train4096=train4096_state,
                    train4096_diagnosis=train4096_diagnosis),
                matched_12k_analysis=matched12,
                portfolio_recommendations=portfolio_recommendations,process_diagnostic_10k=process10,
                advantage_clamp_near10k=clamp10,anchor40_registration=anchor40_registration,
                additional_navtest_probes=additional_probes,
                matched_first_stage_contrasts=matched_contrasts,
                resource_capacity_policy=capacity_policy,resource_capacity_adoption=capacity_adoption,
                pending_resource_additions=pending_additions,
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
    # Reload the external presentation on every read-only cycle. The unique
    # watch owner and all scientific evidence collection stay unchanged.
    import importlib
    import research_report
    importlib.reload(research_report)
    return research_report.render_research_report(root,snapshot,history,sys.modules[__name__])


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
