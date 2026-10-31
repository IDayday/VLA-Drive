"""IL living report and bounded reviewed maintenance; one owner and atomic artifacts."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import tempfile
import threading
import time
import traceback
from .evidence import snapshot
from .diagrams import architecture_svg, NAMES
from .presentation import render
from .events import ILStateEvents
from .health import check as check_health

DEFAULT_ROOT=Path('/mnt/project/qwen-world-il-research-20261030')


def atomic(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=value if isinstance(value,str)else json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'w')as stream:
            stream.write(text);stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def update_history(root, state):
    path=root/'reporting/training_history.json'
    try:old=json.loads(path.read_text())
    except(OSError,ValueError):old=[]
    existing={(r['id'],r['updates']):r for r in old}
    for run in state['runs']:
        for point in run['recent_curve']:
            key=(run['id'],point['updates'])
            if key not in existing:
                existing[key]=dict(point,id=run['id'],observed_unix=state['generated_unix'])
    history=sorted(existing.values(),key=lambda r:(r['id'],r['updates']))
    atomic(path,history);return history


def cycle(root):
    with(root/'reporting/render.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=snapshot(root)
        state['health']=check_health(root,state['runs'])
        state.update(read_only=False,evidence_read_only=True,
                     maintenance_scope='reviewed_native_infrastructure_recovery_only')
        rl=Path('/mnt/project/qwen-world-rl-optimization-20261030')
        state['watch_directories']+=sorted(str(p)for p in [rl,*list((rl/'owners').glob('*'))]
                                           if p.is_dir()and str(p)not in state['watch_directories'])
        history=update_history(root,state)
        for key in NAMES:
            path=root/'figures'/f'architecture_{key}.svg';value=architecture_svg(key)
            if not path.exists()or path.read_text()!=value:atomic(path,value)
        document=render(state,history)
        atomic(root/'reporting/report_snapshot.json',state)
        atomic(root/'report.html',document)
    record=dict(status='RENDERED',at=time.time(),generated_utc=state['generated_utc'],
                report=str(root/'report.html'),runs=len(state['runs']),complete_results=len(state['scores']),
                warnings=state['warnings'],new_GPU_forwards=0,new_scores=0)
    print(json.dumps(record,ensure_ascii=False),flush=True)
    return record,state


def advance_regular(deadline, now, interval):
    """Events never move this deadline; advance only on a regular render."""
    while deadline<=now:deadline+=interval
    return deadline


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--watch',action='store_true');p.add_argument('--once',action='store_true')
    p.add_argument('--interval',type=int,default=1800);a=p.parse_args()
    if a.watch and a.once:p.error('Choose --watch or --once')
    if a.interval<60:p.error('Minimum interval60; registered normal interval1800')
    root=a.root.resolve();(root/'reporting').mkdir(parents=True,exist_ok=True)
    config=dict(schema='world_IL_live_report_config_v1',regular_interval_seconds=a.interval,
        event_backend='linux_inotify',debounce_seconds=1.0,journal_polling=False,
        event_updates_reset_regular_deadline=False,training_seed=42,inference_seed=2101,
        immediate_event_types=['evaluation_complete','training_stage_complete','important_exception',
                              'stage_selection_complete','resource_authorization_changed'],
        new_models_or_scores=False,training_control='reviewed_native_infrastructure_recovery_only',
        no_recipe_changes=True,no_restart_stopped_or_completed=True,
        root_sources=['/mnt/project/qwen-world-postcot-20261025','/mnt/project/qwen-world-four-frameworks-20261029'])
    atomic(root/'reporting/REPORT_CONFIG.json',config)
    if not a.watch:
        cycle(root);return
    lock=(root/'reporting/watch.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:p.error('A live IL report owner already holds watch.lock')
    try:prior=json.loads((root/'reporting/OWNER.json').read_text())
    except(OSError,ValueError):prior={}
    owner=dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
        status='STARTING',started_utc=datetime.now(timezone.utc).isoformat(),root=str(root),
        interval_seconds=a.interval,event_backend='linux_inotify',journal_polling=False,
        source_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in Path(__file__).parent.glob('*.py')})
    atomic(root/'reporting/OWNER.json',owner)
    stop=threading.Event();events=None
    def stopped(*_):
        stop.set()
        if events:events.wake()
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    # Presentation-only owner adoption also preserves the prior regular clock.
    remaining=prior.get('next_regular_unix',0)-time.time()
    next_regular=time.monotonic()+remaining if 0<remaining<=a.interval else time.monotonic()
    owner['previous_reporting_owner']=dict(pid=prior.get('pid'),start_ticks=prior.get('start_ticks'))
    pending=[];event_deadline=None;trigger='startup'
    try:
        while not stop.is_set():
            now=time.monotonic()
            if trigger is None:
                if now>=next_regular:trigger='regular_interval'
                elif event_deadline is not None and now>=event_deadline:trigger='owner_state_event'
                else:
                    wait_until=min(next_regular,event_deadline if event_deadline is not None else next_regular)
                    new=events.wait(wait_until-now)
                    if new:
                        pending.extend(new)
                        if event_deadline is None:event_deadline=time.monotonic()+1.0
                    continue
            if trigger in ('startup','regular_interval'):
                next_regular=advance_regular(next_regular,now,a.interval)
            owner.update(last_trigger=dict(kind=trigger,events=pending,at=time.time()),
                         next_regular_unix=time.time()+next_regular-time.monotonic())
            try:
                record,state=cycle(root)
                if events is None:events=ILStateEvents(root,atomic,state['watch_directories'])
                else:events.add_sources(state['watch_directories'])
                owner.update(status='RUNNING',last_render=record,
                    event_render_count=owner.get('event_render_count',0)+(trigger=='owner_state_event'))
            except Exception:
                owner.update(status='REPORT_RENDER_FAILED',error=traceback.format_exc())
                print(json.dumps(dict(status='REPORT_RENDER_FAILED',error=owner['error'])),flush=True)
                if events is None:events=ILStateEvents(root,atomic)
            atomic(root/'reporting/OWNER.json',owner)
            trigger=None;pending=[];event_deadline=None
    finally:
        owner.update(status='STOPPED',ended_unix=time.time());atomic(root/'reporting/OWNER.json',owner)
        if events:events.close()


if __name__=='__main__':main()
