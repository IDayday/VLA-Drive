"""Fresh CPU validation: owned synthetic supervisors only; no scientific dispatch."""
import argparse
from contextlib import ExitStack
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from unittest.mock import patch

sys.dont_write_bytecode = True
HELPER = Path(__file__).resolve().with_name('adopt_navsim_evaluation_policy.py')
s = importlib.util.spec_from_file_location('guard', HELPER)
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)


def rejected(call):
    try: call()
    except (RuntimeError, subprocess.CalledProcessError): return
    raise AssertionError('Unsafe input accepted')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--g1-manifest', type=Path, required=True)
    p.add_argument('--g3-manifest', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    records = []
    def case(name, call):
        call(); records.append({'case': name, 'passed': True})
    g1 = m.read(a.g1_manifest); g3 = m.read(a.g3_manifest)
    m.check_references(g1); m.check_references(g3)
    case('G1_active_original_command_exact', lambda: m.require(g1['current_stage']['command'] in [p['command'] for p in g1['training_plan']], 'Command differs'))
    case('G3_G2_all_eight_original_boundaries', lambda: m.require([(p['group'],p['target']) for p in g3['training_plan']] == [(g,t) for g in ('G3_EVENT_LOCAL','G2_EVENT_UNIFORM') for t in m.TARGETS], 'G2 lost'))
    case('G2_preserves_fresh_common_initialization_and_native_continuations', lambda: m.require('--resume' not in g3['training_plan'][4]['command'] and all('--resume' in p['command'] for p in g3['training_plan'][5:]) and all('--resume-origin-run' not in p['command'] for p in g3['training_plan'][4:]), 'G2 initialization changed'))
    case('SSH_route_shell_and_full_frozen_command_preserved', lambda: m.require(shlex.split(m.routed(g1,g1['current_stage']['command'])[-1]) == g1['route_prefix']+g1['current_stage']['command'], 'Route changed'))
    case('PID_reuse_rejected', lambda: rejected(lambda: m.verify_process({**m.process(os.getpid()),'start_ticks':-1})))
    case('stale_source_reference_rejected', lambda: rejected(lambda: m.check_references({'preserved_files_sha256':{str(HELPER):'0'*64}})))
    policy = m.read(m.POLICY)
    def reject_dev():
        with patch.object(m,'read',return_value={**policy,'dispatch_development_evaluation':True}):
            rejected(lambda: m.evaluation_policy(g1['groups']))
    case('DEV_policy_rejected', reject_dev)
    case('immutable_native_identity_contract_checked', lambda: m.require(m.native(g1['current_stage']['run'])['identity'] == '8f331622551d1f1c1c439a6d01411affe012a1777a63c61d8525470dc1e3ded5', 'Native differs'))
    test_queue = a.output/'newton_api_queue'
    queue_policy = {**policy, 'navtest_dispatch_output': str(test_queue)}
    plan = g1['training_plan'][0]
    def exact_queue():
        with patch.object(m,'evaluation_policy',return_value=queue_policy):
            first=m.queue_navtest(plan); second=m.queue_navtest(plan)
        assert first['request_sha256']==second['request_sha256']
        request=m.read(first['request'])
        assert set(('run','tag','updates','group','key')) <= request.keys()
    case('Newton_real_enqueue_API_and_identical_reenqueue', exact_queue)
    def queue_conflict():
        path=test_queue/'requests/G1_FULL_UNIFORM_0025000.json'
        value=m.read(path); value['training_identity']='foreign'; m.write(path,value)
        with patch.object(m,'evaluation_policy',return_value=queue_policy): rejected(lambda:m.queue_navtest(plan))
    case('Newton_conflicting_queue_request_rejected', queue_conflict)

    def lifecycle(code, future=False):
        root=a.output/f'owned_cpu_fixture_{code}_{future}'; root.mkdir()
        lockdir=root/'physical_gpu_locks';lockdir.mkdir(); lease=lockdir/'owned_fixture.lock'
        announcement=root/'CHILD.json'; dev=root/'DEV_DISPATCHED'; progress=root/'CHILD_PROGRESS'
        child_code='import time,sys;from pathlib import Path;time.sleep(1);Path(sys.argv[1]).write_text("progress");sys.exit(int(sys.argv[2]))'
        parent_code='import fcntl,json,subprocess,sys;from pathlib import Path;l=open(sys.argv[1],"a");fcntl.flock(l,fcntl.LOCK_EX);c=subprocess.Popen([sys.executable,"-c",sys.argv[2],sys.argv[3],sys.argv[4]]);a=Path(sys.argv[5]);t=a.with_suffix(".tmp");t.write_text(json.dumps({"pid":c.pid}));t.replace(a);c.wait();Path(sys.argv[6]).write_text("legacy DEV")'
        parent=subprocess.Popen([sys.executable,'-c',parent_code,str(lease),child_code,str(progress),str(code),str(announcement),str(dev)],start_new_session=True)
        stopped_checks=[]; signals=[]; dispatched=[]; queued=[]
        completed=25000
        try:
            deadline=time.monotonic()+5
            while not announcement.exists():
                assert time.monotonic()<deadline;time.sleep(.01)
            child=m.process(m.read(announcement)['pid']); pr=m.process(parent.pid)
            run=root/'run';run.mkdir()
            m.write(run/'RESUME_BOUNDARY_VERIFIED.json',{'passed':True,'identity':'owned_fixture','completed':25000})
            target=50000
            stage={'group':'G1_FULL_UNIFORM','target':25000,'run':str(run),'command':['owned_cpu_fixture']}
            future_plan={**stage,'target':target,'command':g1['training_plan'][1]['command']}
            manifest={'supervisor_host':'owned_cpu_fixture','training_host':'owned_cpu_fixture','old_supervisor':pr,'adopted_child':child,'groups':['G1_FULL_UNIFORM'],'current_stage':stage,'training_plan':[future_plan] if future else [],'original_locks':[str(lease)],'route_prefix':[],'training_environment':{},'adopted_resume_reference':None}
            original_verify=m.verify_process; original_signal=signal.pidfd_send_signal
            def verify(ref, stopped=False):
                result=original_verify(ref,stopped)
                if stopped and result['state'] in ('T','t'):
                    with lease.open('a') as file:
                        try:fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:stopped_checks.append(True)
                        else:raise AssertionError('Lock gap during live child')
                return result
            def send(fd,sig,*args):
                signals.append(int(sig));return original_signal(fd,sig,*args)
            def native(path,t=None):return {'completed':t if t is not None else completed,'identity':'owned_fixture'}
            def worker(command,**kwargs):
                nonlocal completed
                assert command==future_plan['command'];dispatched.append(command);completed=target
                return subprocess.CompletedProcess(command,0)
            with ExitStack() as patches:
                patches.enter_context(patch.object(m,'ART',root))
                patches.enter_context(patch.object(m,'check_references',lambda _:None))
                patches.enter_context(patch.object(m,'authorize',lambda _:None))
                patches.enter_context(patch.object(m,'evaluation_policy',lambda _:policy))
                patches.enter_context(patch.object(m,'native',native))
                patches.enter_context(patch.object(m,'queue_navtest',lambda p:queued.append(p) or 'owned_fixture_only'))
                patches.enter_context(patch.object(m,'wait_resource',lambda _:None))
                patches.enter_context(patch.object(m,'verify_process',verify))
                patches.enter_context(patch.object(signal,'pidfd_send_signal',send))
                patches.enter_context(patch.object(subprocess,'run',worker))
                if code:rejected(lambda:m.adopt(manifest,root/'guard'))
                else:m.adopt(manifest,root/'guard')
            parent.wait(timeout=5)
            assert not dev.exists() and progress.exists() and stopped_checks
            assert signals == [int(signal.SIGSTOP),int(signal.SIGKILL)]
            state=m.read(root/'guard/STATE.json')
            assert state['status']==('FAILED' if code else 'ORIGINAL_100K_TRAINING_COMPLETE_NAVTEST_REQUESTS_QUEUED')
            exits=[json.loads(s) for s in (root/'guard/executed_commands.jsonl').read_text().splitlines() if json.loads(s)['stage']=='adopted_child_exit']
            assert exits[0]['exit_code']==code
            assert bool(dispatched)==bool(future and not code)
            if future and not code:assert len(queued)==1
        finally:
            if parent.poll() is None:
                os.kill(parent.pid,signal.SIGKILL);parent.wait(timeout=5)
    case('owned_parent_STOP_live_child_continues_lock_retained_exit0',lambda:lifecycle(0))
    case('owned_child_real_exit7_persisted_and_no_future_dispatch',lambda:lifecycle(7))
    case('owned_child_exit0_then_exact_future_command_and_async_queue',lambda:lifecycle(0,True))
    m.write(a.output/'VALIDATION_RESULTS.json',{'scope':'CPU-only owned synthetic supervisors; real frozen plans and native metadata read; real Newton --enqueue to private fresh queue; no GPU/scientific workers dispatched','helper':str(HELPER),'helper_sha256':m.digest(HELPER),'cases':records,'passed':all(v['passed'] for v in records)})
    print(json.dumps({'cases':len(records),'passed':True,'evidence':str(a.output/'VALIDATION_RESULTS.json')}))


if __name__=='__main__':main()
