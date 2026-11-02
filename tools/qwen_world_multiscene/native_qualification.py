"""Finite owner-invoked eight-GPU exact-native replay qualification driver.

No SSH, scheduling, policy edits or resource acquisition. --run starts local
torchrun only inside a GPU allocation already held by the campaign owner.
"""
import argparse,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
import torch,yaml
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_world_multiscene.recipe import validate_recipe
from .native_review import compare


def process_identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        if fields[0]=='Z':return None
        return dict(pid=pid,pgid=int(fields[2]),start_ticks=int(fields[19]))
    except (OSError,ValueError,IndexError):return None


def descendants(pid,pgid):
    found={};todo=[pid]
    while todo:
        current=todo.pop();identity=process_identity(current)
        if identity is None or identity['pgid']!=pgid or current in found:continue
        found[current]=identity
        try:todo.extend(int(x)for x in Path(f'/proc/{current}/task/{current}/children').read_text().split())
        except OSError:pass
    return found


def launch_leg(command,log_path,env,*,timeout_seconds=1200,cleanup_seconds=60):
    """Every rank inherits owner PGID; clean only exact owned descendants."""
    owned={};pgid=os.getpgrp()
    with Path(log_path).open('xb')as log:
        child=subprocess.Popen(command,cwd=Path.cwd(),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=False)
        root=process_identity(child.pid)
        if root is None or root['pgid']!=pgid:raise RuntimeError('Qualification launcher escaped granted process group')
        try:
            deadline=time.monotonic()+timeout_seconds
            while child.poll()is None:
                owned.update(descendants(child.pid,pgid))
                if time.monotonic()>=deadline:raise subprocess.TimeoutExpired(command,timeout_seconds)
                time.sleep(.25)
            code=child.wait()
            if code:raise RuntimeError(f'Qualification subprocess failed rc={code}; original evidence retained')
            return code
        except BaseException:
            owned.update(descendants(child.pid,pgid))
            if process_identity(child.pid)==root:
                try:os.kill(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
            end=time.monotonic()+cleanup_seconds
            while time.monotonic()<end:
                owned.update(descendants(child.pid,pgid))
                alive=[x for x in owned.values()if process_identity(x['pid'])==x]
                if not alive:break
                time.sleep(.1)
            #Bounded exception cleanup, never a broad PGID/host/GPU signal.
            for x in owned.values():
                if process_identity(x['pid'])==x:
                    try:os.kill(x['pid'],signal.SIGKILL)
                    except ProcessLookupError:pass
            child.wait(timeout=10)
            atomic_json(Path(log_path).with_suffix('.cleanup.json'),dict(owner_pgid=pgid,exact_owned_descendants=list(owned.values())))
            raise


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('config','manifest','output','il-checkpoint','parent-selection','data'):p.add_argument('--'+key,required=True)
    p.add_argument('--run',action='store_true',help='Execute only under an already granted complete eight-GPU lease')
    p.add_argument('--timeout-per-leg-seconds',type=int,default=1200)
    a=p.parse_args();cfg=validate_recipe(yaml.safe_load(Path(a.config).read_text()))
    if cfg['research']['performance']['scenes_per_collection']!=128:raise ValueError('Use actual B128/G16 qualification cfg')
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    common=['--config',str(Path(a.config).resolve()),'--manifest',str(Path(a.manifest).resolve()),'--il-checkpoint',str(Path(a.il_checkpoint).resolve()),
        '--parent-selection',str(Path(a.parent_selection).resolve()),'--data',str(Path(a.data).resolve()),'--qualification','--save-every','2']
    if not 60<=a.timeout_per_leg_seconds<=3600:raise ValueError('Finite qualification timeout required')
    #Normal torchrun -m uses SubprocessHandler(start_new_session=True). The
    #callable run-path/spawn route keeps all eight ranks inside owner PGID.
    prefix=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=8',
        '--start-method','spawn','--run-path',str(Path('tools/qwen_world_multiscene/train.py').resolve())]
    continuous=out/'continuous';resumed=out/'resumed';cp2=resumed/'checkpoints/update000002_inner2'
    steps=[('continuous4',prefix+common+['--output',str(continuous),'--stop-after-update','4']),
        ('split2',prefix+common+['--output',str(resumed),'--stop-after-update','2']),
        ('resume2to4',prefix+common+['--output',str(resumed),'--resume',str(cp2),'--stop-after-update','4'])]
    plan=dict(schema='world_packed_native_8GPU_qualification_plan_v1',source=str(Path.cwd().resolve()),
        micro_scenes=cfg['research']['packed_execution']['micro_scenes'],B=128,G=16,M=4,world=8,
        commands=[dict(name=n,argv=c)for n,c in steps],actual_scoring_requests=2*128*16,
        timeout_per_leg_seconds=a.timeout_per_leg_seconds,maximum_cleanup_seconds=60,
        qualification_only=True,resource_acquisition='None: parent job_worker owns all eight GPU UUID leases',
        first_window_padding='Expected zero; comparator checks actual values and does not claim epoch-tail GPU evidence')
    atomic_json(out/'PLAN.json',plan)
    if not a.run:print('PREPARED '+str(out/'PLAN.json'));return
    if not torch.cuda.is_available()or torch.cuda.device_count()!=8:raise ValueError('Exactly eight owner-assigned visible CUDA devices required')
    env=dict(os.environ);env.setdefault('OMP_NUM_THREADS','1');env.setdefault('MKL_NUM_THREADS','1')
    def interrupted(signum,frame):raise RuntimeError('Qualification owner requested termination: '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    state=dict(status='RUNNING',started_unix=time.time(),steps=[]);atomic_json(out/'STATE.json',state)
    try:
        for name,command in steps:
            entry=dict(name=name,argv=command,status='RUNNING',started_unix=time.time());state['steps'].append(entry);atomic_json(out/'STATE.json',state)
            code=launch_leg(command,out/f'{name}.log',env,timeout_seconds=a.timeout_per_leg_seconds)
            entry.update(returncode=code,finished_unix=time.time(),status='COMPLETE'if code==0 else'FAILED')
            atomic_json(out/'STATE.json',state)
        results=[]
        for update,inner in ((2,2),(4,0)):
            value=compare(continuous/f'checkpoints/update{update:06}_inner{inner}',resumed/f'checkpoints/update{update:06}_inner{inner}')
            atomic_json(out/f'COMPARE_u{update:06}.json',value);results.append(value)
        receipt=dict(schema='world_packed_8rank_exact_native_qualification_v1',status='PASS',plan=plan,comparisons=results,
            finished_unix=time.time(),qualification_only=True,not_performance_evidence=True,
            GPU_tail_padding='No claim if actual pending_padding_scenes=0; separate CPU tests exercised zero-all-loss padding')
        atomic_json(out/'COMPLETE.json',receipt);state.update(status='COMPLETE',finished_unix=time.time());atomic_json(out/'STATE.json',state)
    except BaseException:
        if state['steps']and state['steps'][-1]['status']=='RUNNING':state['steps'][-1].update(status='FAILED',finished_unix=time.time())
        state.update(status='FAILED',finished_unix=time.time(),error=traceback.format_exc());atomic_json(out/'STATE.json',state);raise


if __name__=='__main__':main()
