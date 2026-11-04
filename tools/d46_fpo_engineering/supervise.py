"""Recover only confirmed temporary-node loss; numerical failures need review."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from starVLA.rl.flow_grpo.transactions import atomic_json
from tools.d46_augustus_ops.event_wait import wait_for_events
from .dispatch import ROUND, ROOT, TASK, gang, stop_owned

SOURCE=Path(__file__).resolve().parents[2]
TEMPORARY=('training-vlawm-zt4','training-vla-zt')
STABLE=['training-vlawm-tsinghua-zt5','training-vlawm-tsinghua-zt']


def read(path,default=None):
    return json.loads(Path(path).read_text())if Path(path).is_file()else default


def alive(host):
    try:
        result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'hostname'],text=True,capture_output=True,timeout=12)
        return result.returncode==0 and result.stdout.strip()==host+'-worker-0'
    except subprocess.TimeoutExpired:return False


def main():
    from starVLA.rl.d46_fpo_engineering.reshard import prepare
    op=ROUND/'operations';failures={}
    with(op/'training_recovery.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_json(op/'TRAINING_RECOVERY_OWNER.json',dict(pid=os.getpid(),source=str(SOURCE),at=time.time()))
        while not(op/'STOP_TRAINING_RECOVERY').exists():
            policy=read('/mnt/project/server_dispatch_policy.json',{})
            authorized=policy.get('task_holds',{}).get(TASK,{}).get('new_training_authorized')is True
            route=read(op/'PLAN.json',{}).get('routes',{}).get('FPO_MATURE',{})
            if not authorized or route.get('world')!=32:
                wait_for_events([op],60);continue
            output=Path(route['output']);receipts=read(route['launch_receipts'],{}).get('receipts',[])
            if len(receipts)!=4:wait_for_events([op],30);continue
            latest=max([p.stat().st_mtime for p in output.glob('metrics_rank*.jsonl')]+[min(r['at']for r in receipts)])
            results=[read(Path(r['job'])/'RESULT.json',{})for r in receipts]
            broken=any(r.get('returncode')not in(None,0)for r in results)
            if not broken and time.time()-latest<300:
                wait_for_events([op,output]+[Path(r['job'])for r in receipts],60,ignore=('STATE.json','EVAL_STATE.json','report.html'));continue
            if any((Path(r['job'])/'PERFORMANCE_REVIEW_REASON.json').exists()for r in receipts):
                atomic_json(op/'RECOVERY_REVIEW.json',dict(at=time.time(),reason='Performance stop remains effective'));break
            down=[]
            for host in TEMPORARY:
                failures[host]=0 if alive(host)else failures.get(host,0)+1
                if failures[host]>=2:down.append(host)
            if not down:
                wait_for_events([op],30);continue
            hold=op/'EVALUATION_DISPATCH_HOLD';hold.touch()
            attempt='fallback16_'+str(int(time.time()))
            atomic_json(op/'TEMPORARY_TRAINING_NODE_LOSS.json',dict(at=time.time(),hosts=down,old_output=str(output),attempt=attempt,
                resume_from_last_atomic_checkpoint=True,unsaved_updates_may_be_lost=True,no_automatic_temporary_reuse=True))
            stop_owned(receipts,force=True)
            deadline=time.time()+120
            # Normal torchrun shutdown handles its owned children. Never
            # acquire survivor GPUs until their worker receipts confirm exit.
            while time.time()<deadline and not all((Path(r['job'])/'RESULT.json').exists()for r in receipts if r['host']not in down):
                wait_for_events([Path(r['job'])for r in receipts if r['host']not in down],30)
            if not all((Path(r['job'])/'RESULT.json').exists()for r in receipts if r['host']not in down):
                signals=stop_owned(receipts,force=True,signal_name='SIGKILL')
                atomic_json(op/'OWNED_STALLED_GANG_RELEASE.json',dict(at=time.time(),reason='Confirmed temporary-node loss; survivor gang cannot complete collective',signals=signals))
                wait_for_events([Path(r['job'])for r in receipts if r['host']not in down],30)
                if not all((Path(r['job'])/'RESULT.json').exists()for r in receipts if r['host']not in down):
                    atomic_json(op/'RECOVERY_REVIEW.json',dict(at=time.time(),reason='Surviving worker has not released GPUs; no resource reuse'));break
            if not all(alive(h)for h in STABLE):
                atomic_json(op/'RECOVERY_REVIEW.json',dict(at=time.time(),reason='Stable fallback node unavailable'));break
            cp=read(output/'latest.json')['checkpoint']
            new=ROOT/'runs'/('D46_FPO_MATURE_'+attempt)
            prepare(cp,route['source'],new,target_world=16)
            gang(SOURCE,new,STABLE,'172.16.199.244',attempt=attempt,port=29609)
            hold.unlink(missing_ok=True)
            atomic_json(op/'TRAINING_RECOVERY_RESULT.json',dict(status='DISPATCHED_NATIVE_FALLBACK',at=time.time(),world=16,output=str(new),origin_checkpoint=cp))
            # No automatic return to the revoked/interrupted temporary hosts.
            break


if __name__=='__main__':main()
