"""Publish sealed B/D checkpoint scores; no model, scorer or training-log reads.

Run from immutable operational source against a separate report-only checkout.
The existing living HTML owner is independent and remains the sole HTML writer.
"""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from .evidence import Evidence, score_row, FOUR, fingerprint
from .health import atomic

REPORT = 'reports/il_bd_navtest_updates_20261103'
BRANCH = 'reports/il-bd-navtest-live-20261103'
CONTROL = Path('/mnt/project/qwen-world-il-research-20261030')


def read(path, default=None):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return default


def collect():
    cadence = read(CONTROL/'navtest_cadence_20261031/STATE.json', {})
    jobs = {j['root']:j for j in cadence.get('jobs', [])}
    completed, pending = [], []
    for architecture in ('B','D'):
        for root in sorted(FOUR.glob(f'periodic_NAVTEST_ARCH_{architecture}_*')):
            pin = read(root/'PINNED.json', {}).get('models', {})
            if len(pin) != 1:raise ValueError('Ambiguous periodic checkpoint identity')
            label, bound = next(iter(pin.items()));directory=root/label
            if bound.get('architecture_id') != architecture:raise ValueError('Changed architecture pin')
            closure = read(root/'COMPLETE.json', {})
            if closure.get('status') == 'COMPLETE':
                native = read(directory/'COMPLETE.json', {})
                if closure.get('all_owned_workers_exited') is not True or native.get('all_owned_workers_exited') is not True:
                    raise ValueError('Unfinished workers cannot become a published checkpoint')
                evidence = Evidence();row=score_row(evidence,directory/'SUMMARY.json',label,method='ARCH_'+architecture)
                if row is None or row['updates'] != bound['updates']:
                    raise ValueError('Invalid sealed full score: '+str(evidence.warnings))
                completed.append(dict(architecture=architecture,**row))
            else:
                job=jobs.get(str(root), {})
                pending.append(dict(architecture=architecture,updates=bound['updates'],
                    status=job.get('status',read(root/'owner.json',{}).get('status','PENDING')),
                    root=str(root),PDMS=None))
    latest={a:read(FOUR/f'runs/ARCH_{a}/IL_seed42/LATEST.json',{}).get('updates')for a in ('B','D')}
    result=dict(schema='B_D_full_periodic_Navtest_readout_v1',completed=completed,pending=pending,
                latest_saved_IL_updates=latest,protocol=dict(scenes=12146,physical_logs=44,seed=2101,
                metric='canonical_NAVSIM_v1.1',inference='single_B1_cleanx_K10_shift5',invalid=0),
                scope='Already observed Test; no test labels used for training; partial means unpublished')
    result['content_sha256']=fingerprint(result)
    return result


def documents(snapshot):
    now=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    rows=snapshot['completed'];latest=snapshot['latest_saved_IL_updates']
    text=['# B / D IL：每 2K 检查点 Navtest 动态结果', '', '更新：'+now, '',
          '只发布完整 12,146 场景、44 个物理 log、invalid=0 的结果。统一 canonical NAVSIM v1.1，seed2101，B1 单候选 K10/shift5 ODE。', '',
          '最新保存的 IL 检查点：'+', '.join(f'{a} {latest[a]}'for a in ('B','D'))+'。训练过程中的 journal 不被本发布器轮询。', '',
          '| 架构 | checkpoint | PDMS | NC | DAC | TTC | EP | Comfort | DDC |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    buf=io.StringIO();writer=csv.writer(buf,lineterminator='\n');writer.writerow(['architecture','updates','PDMS','NC','DAC','TTC','EP','Comfort','DDC','summary_sha256'])
    keys=['no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','ego_progress','comfort','driving_direction_compliance']
    for row in sorted(rows,key=lambda r:(r['architecture'],r['updates'])):
        metrics=[row['means'][k]for k in keys]
        text.append('| '+row['architecture']+' | '+str(row['updates'])+' | '+' | '.join(f'{v:.4f}'for v in [row['PDMS'],*metrics])+' |')
        writer.writerow([row['architecture'],row['updates'],row['PDMS'],*metrics,row['summary_sha256']])
    text+=['','当前两组各自已完成评测的最高分：','']
    for a in ('B','D'):
        valid=[r for r in rows if r['architecture']==a]
        if valid:
            best=max(valid,key=lambda r:r['PDMS']);last=max(valid,key=lambda r:r['updates'])
            text.append(f'- {a}：最高 {best["PDMS"]:.4f}（{best["updates"]}）；最新完整点 {last["PDMS"]:.4f}（{last["updates"]}）。')
    text+=['','尚未完成的点：','','| 架构 | checkpoint | 状态 |','|---|---:|---|']
    for row in snapshot['pending']:text.append(f'| {row["architecture"]} | {row["updates"]} | {row["status"]} |')
    if not snapshot['pending']:text.append('| — | — | 无 |')
    text+=['','更新规则：每 60 秒仅检查封存结果与队列元数据；完整结果到达后自动更新本表并推送当前新报告分支。每 2K 评测登记由原 cadence 负责，checkpoint/scorer/训练源码与精度不在发布器中修改。', '',
           'D50k 曾有 7 个资源到期退出分片被误当完成，现已审核并保留原场景续测；禁止用部分分数填表。完整动态 HTML 仍由原单一 owner 更新。', '',
           '源码修复与验收：[修复分支](https://github.com/IDayday/VLA-Drive/tree/fix/il-bd-navtest-completion-20261103)。逐点身份与结果索引见 [results.json](results.json)。', '']
    value=dict(snapshot,generated_utc=now)
    return {'README.md':'\n'.join(text),'scores.csv':buf.getvalue(),'results.json':json.dumps(value,indent=2,ensure_ascii=False)+'\n'}


def git(checkout,*args):
    result=subprocess.run(['git',*args],cwd=checkout,text=True,capture_output=True,timeout=120)
    if result.returncode:raise RuntimeError(result.stderr)
    return result.stdout.strip()


def publish(checkout, snapshot):
    if git(checkout,'branch','--show-current')!=BRANCH:raise ValueError('Report-only branch required')
    prefix=REPORT+'/'
    dirty=git(checkout,'diff','--name-only').splitlines()+git(checkout,'diff','--cached','--name-only').splitlines()
    if any(not p.startswith(prefix)for p in dirty):raise ValueError('Unrelated edits cannot be published')
    destination=checkout/REPORT;destination.mkdir(parents=True,exist_ok=True)
    old=read(destination/'results.json',{})
    if old.get('content_sha256') != snapshot['content_sha256']:
        for name,value in documents(snapshot).items():
            path=destination/name;temporary=path.with_name(path.name+'.tmp.'+str(os.getpid()))
            temporary.write_text(value,encoding='utf-8');os.replace(temporary,path)
    git(checkout,'add','--',REPORT)
    staged=git(checkout,'diff','--cached','--name-only').splitlines()
    if any(not p.startswith(prefix)for p in staged):raise ValueError('Unrelated staged files')
    if staged:git(checkout,'commit','-m','Update sealed B D IL Navtest checkpoint results')
    head=git(checkout,'rev-parse','HEAD');remote=git(checkout,'ls-remote','origin','refs/heads/'+BRANCH).split()
    if not remote or remote[0]!=head:git(checkout,'push','-u','origin','HEAD:refs/heads/'+BRANCH)
    remote=git(checkout,'ls-remote','origin','refs/heads/'+BRANCH).split()[0]
    if remote!=head:raise RuntimeError('Remote SHA mismatch')
    return dict(status='PUBLISHED',head=head,remote_sha=remote,completed=len(snapshot['completed']),
                pending=snapshot['pending'],at=time.time(),content_sha256=snapshot['content_sha256'])


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkout',type=Path,required=True)
    p.add_argument('--state-root',type=Path,required=True);p.add_argument('--watch',action='store_true')
    p.add_argument('--interval',type=int,default=60);a=p.parse_args()
    if a.interval<60:p.error('Metadata checks no more often than once per minute')
    a.state_root.mkdir(parents=True,exist_ok=True);stop=threading.Event()
    signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
    with (a.state_root/'owner.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic(a.state_root/'OWNER.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
            source=str(Path(__file__).resolve()),checkout=str(a.checkout),interval=a.interval,started_unix=time.time()))
        last=None
        while not stop.is_set():
            try:
                snapshot=collect()
                if snapshot['content_sha256']!=last:
                    status=publish(a.checkout,snapshot);atomic(a.state_root/'STATUS.json',status);last=snapshot['content_sha256']
            except Exception as error:
                atomic(a.state_root/'STATUS.json',dict(status='PUBLICATION_REVIEW_OR_TRANSPORT_WAIT',error=repr(error),at=time.time(),model_or_scorer_calls=0))
                if not a.watch:raise
            if not a.watch:break
            stop.wait(a.interval)


if __name__=='__main__':main()
