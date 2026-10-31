"""Research-job receipt/locks and cooperative checkpoint guard.

Copied from the qualified World worker; only explicit research module/owner
registrations and this module's invocation namespace differ. Original worker
and running scientific snapshots remain unchanged.

No unrelated PID is signalled. Algorithms must poll boundary_stop_due() at
their native recoverable boundary. Long tasks are handed back to the permanent
host queue after a guarded exit; this worker never starts a second lineage.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from starVLA.rl.qwen_world_postcot.contracts import authorization,require_dispatch,file_sha

REGISTERED_LEARNERS = (
    'starVLA.rl.qwen_world_research_algorithms.safety_feasible_score_entry',
    'starVLA.rl.qwen_world_research_algorithms.success_memory_entry',
    'starVLA.rl.qwen_world_research_algorithms.fpo_entry',
    'tools.qwen_world_research.safety_component_probe',
    'starVLA.rl.qwen_world_research_algorithms.safety_lr_entry',
    'tools.qwen_world_research.train',
    'starVLA.rl.qwen_world_research_algorithms.train_nft',
    'starVLA.rl.qwen_world_research_algorithms.batchstd_entry',
    'starVLA.rl.qwen_world_research_algorithms.safe_comfort_entry',
    'starVLA.rl.qwen_world_research_algorithms.safety_efficiency_entry',
    'starVLA.rl.qwen_world_research_algorithms.preference_entry',
    'starVLA.rl.qwen_world_research_algorithms.lastdet_entry',
    'starVLA.rl.qwen_world_research_algorithms.rescue_entry',
)

REGISTERED_EVALUATORS = (
    'tools.qwen_world_research.evaluate',
    'tools.qwen_world_research.probe_zero_rollouts',
)

def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)


def task_ranks(state):
    """Verified descendants, including torchrun's separate rank sessions."""
    expected=state['command'];output=expected[expected.index('--output')+1] if '--output' in expected else None
    rows=[];processes={}
    known={row['pid']:row for row in state.get('known_rank_processes',[])+state.get('guard_rank_processes',[])}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            stat=(proc/'stat').read_text().split();processes[int(proc.name)]=(int(stat[3]),int(stat[21]),stat[2])
        except (OSError,ValueError,IndexError):continue
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            pid=int(proc.name)
            if pid not in processes or processes[pid][2] in ('Z','X'):continue
            old=known.get(pid);previous=old is not None and verified_process(pid,old['start_ticks'],old['pgid'])
            ancestor=pid;visited=set();descendant=False
            while ancestor in processes and ancestor not in visited:
                visited.add(ancestor)
                if ancestor==state['child_pid']:
                    descendant=processes[ancestor][1]==state['child_start_ticks'];break
                ancestor=processes[ancestor][0]
            if not descendant and not previous:continue
            args=[v for v in (proc/'cmdline').read_bytes().decode().split('\0') if v]
            entry=args[args.index('-m')+1] if '-m' in args else None
            if entry not in ('tools.qwen_world_postcot.train_il','tools.qwen_world_postcot.train_rl',
                             'tools.qwen_world_postcot.train_scope_rl',
                             'tools.qwen_world_postcot.train_selected_parent_rl')+REGISTERED_LEARNERS:continue
            if output is None or '--output' not in args or args[args.index('--output')+1]!=output:continue
            status=dict(line.split(':',1) for line in (proc/'status').read_text().splitlines() if ':' in line)
            rows.append(dict(pid=pid,start_ticks=int((proc/'stat').read_text().split()[21]),
                pgid=os.getpgid(pid),entry=entry,
                usr1_handler=bool(int(status['SigCgt'].strip(),16)&(1<<(signal.SIGUSR1-1)))))
        except (OSError,ValueError,IndexError,ProcessLookupError):continue
    return rows


def verified_process(pid,start_ticks,pgid):
    try:
        stat=(Path('/proc')/str(pid)/'stat').read_text().split()
        return stat[2] not in ('Z','X') and int(stat[21])==start_ticks and os.getpgid(pid)==pgid
    except (OSError,ValueError,ProcessLookupError):return False


def last_complete_checkpoint(state):
    command=state['command']
    if '--output' not in command:return None
    root=Path(command[command.index('--output')+1]);candidates=[]
    for name in ('progress.json','LATEST.json','latest.json'):
        try:
            value=json.loads((root/name).read_text());checkpoint=Path(value['checkpoint'])
            seal=json.loads((checkpoint/'COMPLETE').read_text());files=json.loads((checkpoint/'files.json').read_text())
            if seal['files_sha256']!=file_sha(checkpoint/'files.json'):continue
            if any(not (checkpoint/n).resolve().is_relative_to(checkpoint.resolve()) or not (checkpoint/n).is_file() for n in files):continue
            candidates.append(dict(path=str(checkpoint.resolve()),updates=seal['updates'],receipt=seal,
                inventory_seal_verified=True,full_tensor_hashes_deferred_to_strict_resume=True))
        except (OSError,ValueError,KeyError,TypeError):continue
    return max(candidates,key=lambda value:value['updates']) if candidates else None


def deadline_tick(state,child,root,stop,*,now=None):
    """Cooperative native stop, then explicit owned-group absolute-expiry fallback.

    Forced expiry preserves the last atomic recovery point but is never called
    lossless/exact. Injected times are solely for standalone child unit tests;
    production timestamps always come from the immutable task authorization.
    """
    if not state['temporary']:return
    now=time.time() if now is None else now
    if now>=state['guard_unix'] and not state.get('expiry_guard_requested'):
        stop.touch();state.update(expiry_guard_requested=True,stop_requested_unix=now)
        state['guard_rank_processes']=task_ranks(state)
        signalled=[]
        for row in state['guard_rank_processes']:
            if row['usr1_handler'] and verified_process(row['pid'],row['start_ticks'],row['pgid']):
                os.kill(row['pid'],signal.SIGUSR1);signalled.append(row)
        state['native_usr1_signals']=signalled
        write(root/'RUNNING.json',state)
    for seconds,label,sig in ((90,'SIGTERM',signal.SIGTERM),(30,'SIGKILL',signal.SIGKILL)):
        key='expiry_'+label
        if now<state['deadline_unix']-seconds or state.get(key):continue
        leader=verified_process(state['child_pid'],state['child_start_ticks'],state['child_pid'])
        ranks=[row for row in state.get('known_rank_processes',[])+state.get('guard_rank_processes',[]) if verified_process(row['pid'],row['start_ticks'],row['pgid'])]
        if not leader and not ranks:
            if child.poll() is None:raise RuntimeError('Owned process identity changed; refusing an unsafe expiry signal')
            continue
        state[key]=dict(at_unix=now,pgid=state['child_pid'],leader_identity_verified=leader,
            verified_rank_processes=ranks,last_atomic_checkpoint=last_complete_checkpoint(state))
        state['forced_expiry']=True
        state['recovery_contract']='FORCED_EXIT_REVIEW_REQUIRED; latest atomic checkpoint retained; uncommitted work may be lost'
        write(root/'FORCED_EXPIRY.json',state)
        groups=({state['child_pid']} if leader else set())|{row['pgid']for row in ranks}
        for pgid in groups:
            try:os.killpg(pgid,sig)
            except ProcessLookupError:pass


class ResourceRefused(RuntimeError):
    """A positively identified pre-child capacity/owner/lease change only."""


PRIMARY_LEASE_ROOT=Path('/mnt/project/runtime_gpu_leases')
SHARED_LEASE_ROOT=Path('/mnt/project/runtime_gpu_shared_workers')
RESEARCH_LEASE_ROOT=Path('/mnt/project/runtime_gpu_research_slots')
EVALUATION_LEASE_ROOT=Path('/mnt/project/runtime_gpu_research_evaluation_slots')


def _known_owners(rows,mapping,indices):
    for row in rows:
        if row['uuid']not in {mapping[i]for i in indices}:continue
        try:args=(Path('/proc')/str(row['pid'])/'cmdline').read_bytes().decode().split('\0')
        except (FileNotFoundError,ProcessLookupError,PermissionError)as error:
            raise ResourceRefused('Owner visibility changed before child startup')from error
        allowed_roots=('/mnt/project/qwen-sourceflow-', '/mnt/project/qwen-world-postcot-',
                       '/mnt/project/qwen-world-four-frameworks-20261029/',
                       '/mnt/project/qwen-world-rl-optimization-20261030/')
        permitted_entries={'tools.qwen_world_postcot.evaluate','tools.qwen_world_postcot.refresh_candidates',
                           'tools.qwen_world_postcot.train_il','tools.qwen_world_postcot.train_scope_rl',
                           'tools.qwen_world_postcot.train_selected_parent_rl','tools.qwen_world_compare.train_il',
                           'tools.qwen_world_compare.train_rl','tools.qwen_world_compare.evaluate',
                           }|set(REGISTERED_LEARNERS)|set(REGISTERED_EVALUATORS)
        entry=args[args.index('-m')+1] if '-m' in args else None
        output=args[args.index('--output')+1] if '--output' in args else None
        if entry not in permitted_entries or not output or not output.startswith(allowed_roots):
            raise ResourceRefused('GPU owner has no exact known research module/output identity')


def _admission_sample():
    devices={}
    for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True).splitlines():
        index,uuid,total,free,util=line.split(',')
        devices[int(index)]=dict(index=int(index),uuid=uuid.strip(),total_MiB=int(total),free_MiB=int(free),utilization=int(util))
    owners=[]
    for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory',
            '--format=csv,noheader,nounits'],text=True).splitlines():
        pid,uuid,memory=line.split(',')
        owners.append(dict(pid=int(pid),uuid=uuid.strip(),used_MiB=int(memory)if memory.strip().isdigit()else None))
    return devices,owners


def _gpu_admission(args,host,grant,indices,expected,mapping,locks):
    if not indices:return
    from starVLA.rl.qwen_world_postcot.resources import gpu_activity_issue
    def verify(devices,owners,*,locked):
        physical=[devices[i]['uuid']for i in indices]
        if expected is not None and physical!=expected:
            raise ResourceRefused('Physical GPU mapping changed after scheduling')
        if mapping and physical!=[mapping[i]for i in indices]:
            raise ResourceRefused('Physical GPU mapping changed under admission lock')
        mapping.update({i:devices[i]['uuid']for i in indices})
        if args.share_min_free_mib:
            permitted = list(grant['permanent_hosts'])+list(grant.get('shared_training_hosts',[]))
            if getattr(args,'primary_shared_training',False):
                permitted += grant.get('unbounded_temporary_hosts',[])
            if host not in permitted or args.share_min_free_mib<8192:
                raise ValueError('Sharing requires a permanent host and measured valid headroom')
            _known_owners(owners,mapping,indices)
        elif any(row['uuid']in physical for row in owners):
            raise ResourceRefused('GPU acquired another owner '+('under admission lock'if locked else'before dispatch'))
        for index in indices:
            issue=gpu_activity_issue(devices[index],owners)
            if issue:raise ResourceRefused(issue+(' under admission lock'if locked else' before dispatch'))
        required=args.share_min_free_mib or args.minimum_free_mib
        if required and any(devices[i]['free_MiB']<required for i in indices):
            raise ResourceRefused('Measured GPU headroom changed '+('under admission lock'if locked else'before dispatch'))
    devices,owners=_admission_sample();verify(devices,owners,locked=False)
    PRIMARY_LEASE_ROOT.mkdir(exist_ok=True);SHARED_LEASE_ROOT.mkdir(exist_ok=True);RESEARCH_LEASE_ROOT.mkdir(exist_ok=True);EVALUATION_LEASE_ROOT.mkdir(exist_ok=True)
    for index in indices:
        shared_worker=args.share_min_free_mib and not getattr(args,'primary_shared_training',False)
        lease_root=(EVALUATION_LEASE_ROOT if args.evaluation_shared_slot else RESEARCH_LEASE_ROOT if args.research_shared_slot else SHARED_LEASE_ROOT if shared_worker else PRIMARY_LEASE_ROOT)
        lease=(lease_root/(mapping[index]+'.lock')).open('a')
        try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            lease.close();raise ResourceRefused('GPU admission lease became busy before child startup')from error
        locks.append(lease)
    # A second actual total/free/utilization+per-process-memory sample under
    # the lock closes the invisible-container race before Popen or CUDA init.
    devices,owners=_admission_sample();verify(devices,owners,locked=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--receipt-root',required=True);parser.add_argument('--gpus',default='')
    parser.add_argument('--expected-gpu-uuids-json');parser.add_argument('--command-json',required=True);parser.add_argument('--cwd',required=True)
    parser.add_argument('--share-min-free-mib',type=int);parser.add_argument('--minimum-free-mib',type=int)
    parser.add_argument('--primary-shared-training',action='store_true',
        help='Own the primary training lease while cohosting known inference with measured headroom')
    parser.add_argument('--research-shared-slot',action='store_true',
        help='One additional research learner per physical GPU; all existing owners must be identified and measured headroom retained')
    parser.add_argument('--evaluation-shared-slot',action='store_true',
        help='One research evaluation per physical GPU with exact known owners and measured headroom')
    args=parser.parse_args();root=Path(args.receipt_root);root.mkdir(parents=True,exist_ok=True)
    command=json.loads(args.command_json)
    if not isinstance(command,list)or not command or not all(isinstance(value,str)for value in command):raise ValueError('Exact argument vector required')
    if args.primary_shared_training and (not args.share_min_free_mib or
            not any(entry in command for entry in REGISTERED_LEARNERS)):
        raise ValueError('Primary sharing is restricted to this explicitly registered research trainer')
    if args.research_shared_slot and (args.primary_shared_training or not args.share_min_free_mib
            or args.share_min_free_mib<32768 or not any(entry in command for entry in REGISTERED_LEARNERS)):
        raise ValueError('Additional research slot requires its exact learner and at least32768MiB measured headroom')
    if args.evaluation_shared_slot and (args.primary_shared_training or args.research_shared_slot
            or not args.share_min_free_mib or args.share_min_free_mib<20480
            or not any(entry in command for entry in REGISTERED_EVALUATORS)):
        raise ValueError('Evaluation slot requires its exact module and at least20480MiB measured headroom')
    indices=[]if not args.gpus else[int(value)for value in args.gpus.split(',')]
    if len(indices)!=len(set(indices)):raise ValueError('GPU indices must be unique')
    if args.minimum_free_mib is not None and args.minimum_free_mib<1:raise ValueError('Minimum headroom must be positive')
    expected=None if args.expected_gpu_uuids_json is None else json.loads(args.expected_gpu_uuids_json)
    if expected is not None and(not isinstance(expected,list)or len(expected)!=len(indices)or not all(isinstance(v,str)for v in expected)):
        raise ValueError('Expected physical GPU identities must match requested indices')
    lock=(root/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if(root/'RESULT.json').exists():raise ValueError('Invocation already finalized; do not duplicate it')
    host=socket.gethostname().removesuffix('-worker-0');mapping={};locks=[]
    state=dict(status='ADMISSION',host=host,worker_pid=os.getpid(),worker_start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
        start_unix=time.time(),gpus=indices,requested_gpu_uuids=expected,
        command=command,cwd=str(Path(args.cwd).resolve()),shared_headroom_MiB=args.share_min_free_mib,
        additional_research_slot=args.research_shared_slot,evaluation_slot=args.evaluation_shared_slot)
    try:
        try:require_dispatch(admit_long_job=True)
        except PermissionError as error:raise ResourceRefused('Host authorization expired or was withdrawn before child startup')from error
        grant,_=authorization();_gpu_admission(args,host,grant,indices,expected,mapping,locks)
    except ResourceRefused as error:
        state.update(status='REFUSED_RESOURCE',child_not_started=True,returncode=75,reason=str(error),end_unix=time.time(),
            gpu_uuids=[mapping[i]for i in indices]if all(i in mapping for i in indices)else[])
        if state['requested_gpu_uuids']is None:state['requested_gpu_uuids']=state['gpu_uuids']
        write(root/'REFUSED_RESOURCE.json',state);write(root/'RESULT.json',state)
        return 75
    stop=root/'STOP_REQUESTED'
    env=dict(os.environ,WORLD_POSTCOT_STOP_FILE=str(stop),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
        PYTHONPATH=str(Path(args.cwd).resolve())+(os.pathsep+os.environ['PYTHONPATH']if os.environ.get('PYTHONPATH')else''))
    if indices:env['CUDA_VISIBLE_DEVICES']=','.join(map(str,indices))
    state.update(status='RUNNING',gpu_uuids=[mapping[i]for i in indices],temporary=host not in grant['permanent_hosts'] and host not in grant.get('unbounded_temporary_hosts', []),
        guard_unix=grant['expires_unix']-grant['checkpoint_guard_seconds'],deadline_unix=grant['expires_unix'])
    def request(*_):
        stop.touch();state['stop_requested_unix']=time.time();write(root/'RUNNING.json',state)
    signal.signal(signal.SIGTERM,request);signal.signal(signal.SIGINT,request)
    with(root/'stdout.log').open('ab',buffering=0)as log:
        child=subprocess.Popen(command,cwd=args.cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state['child_pid']=child.pid;state['child_start_ticks']=int(Path(f'/proc/{child.pid}/stat').read_text().split()[21]);write(root/'RUNNING.json',state)
        while child.poll()is None or task_ranks(state):
            state['known_rank_processes']=task_ranks(state);deadline_tick(state,child,root,stop);time.sleep(5)
    state.update(status='FORCED_EXIT_REVIEW_REQUIRED' if state.get('forced_expiry') else 'EXITED' if child.returncode==0 else 'FAILED',
        returncode=child.returncode,end_unix=time.time(),migration_required=bool(stop.exists()and state['temporary']))
    if state.get('forced_expiry'):
        state['last_atomic_checkpoint']=last_complete_checkpoint(state);write(root/'FORCED_EXPIRY.json',state)
    write(root/'RESULT.json',state)
    return child.returncode


if __name__=='__main__':raise SystemExit(main())
