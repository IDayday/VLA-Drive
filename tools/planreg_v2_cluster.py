"""Explicit multi-node launcher, including unequal authorized GPU counts per node.

Only changes placement/admission. The child executes scripts/train_planreg_v2.py.
GB128 profiles are bounded by that production entry; formal launches require both gates.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
ENV_KEYS=('PYTHONNOUSERSITE','PYTHONPATH','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
    'NUPLAN_MAPS_ROOT','CUBLAS_WORKSPACE_CONFIG','NCCL_SOCKET_IFNAME','GLOO_SOCKET_IFNAME','NCCL_DEBUG',
    'PLANREG_BASE_VLM_PATH','PLANREG_VQA_VLM_PATH','PLANREG_V2_NORMALIZER','PLANREG_V2_SHARED_INIT')


def environment(node, extra=None):
    env={k:os.environ[k] for k in ENV_KEYS if k in os.environ}
    env.update(extra or {})
    # Prefer full physical UUIDs from the actual probe, avoiding CUDA/NVML index-order ambiguity.
    env['CUDA_VISIBLE_DEVICES']=','.join(map(str,node.get('gpu_uuids',node['gpus'])))
    env['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    env['PYTHONNOUSERSITE']='1'
    env['PYTHONPATH']=str(ROOT)+':/mnt/project/DriveVLA-M0-env/lib/python3.9/site-packages'
    return env


def command_on_node(node, command, extra=None):
    cmd=['env']+[k+'='+str(v) for k,v in environment(node,extra).items()]+list(map(str,command))
    if node['host']=='local':return cmd
    remote='cd '+shlex.quote(str(ROOT))+' && exec '+shlex.join(cmd)
    return ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','ServerAliveInterval=15',
        '-o','ServerAliveCountMax=3',node['host'],remote]


def inspect_nodes(layout):
    nodes=layout.get('nodes') or [dict(host='local',gpus=list(range(layout['gpus_per_node'])))]
    def inspect(node):
        cmd=command_on_node(node,[sys.executable,ROOT/'tools/planreg_v2_preflight.py','--hardware'])
        result=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=150,check=True)
        rows=[line for line in result.stdout.splitlines() if line.startswith('{')]
        if not rows:raise ValueError('No actual hardware/environment response from '+node['host'])
        actual=json.loads(rows[-1])
        if [g['physical_index'] for g in actual['hardware']]!=node['gpus']:
            raise ValueError('CUDA physical indices differ from authorized card list on '+node['host'])
        return node['host'],actual
    with ThreadPoolExecutor(len(nodes)) as pool:return dict(pool.map(inspect,nodes))


def training_commands(spec):
    nodes=spec['nodes'];world=sum(len(n['gpus']) for n in nodes)
    if world*spec['microbatch']*spec['accumulate']!=128:raise ValueError('Effective global batch must remain 128')
    mode=spec['mode']
    if mode not in ('formal','profile','communication'):raise ValueError('Unknown cluster launch mode')
    if mode=='formal' and os.getenv('START_FORMAL_AFTER_PREFLIGHT')!='1':
        raise ValueError('Formal training requires START_FORMAL_AFTER_PREFLIGHT=1')
    commands=[]
    for rank,node in enumerate(nodes):
        cmd=[sys.executable,'-m','torch.distributed.run','--nnodes',str(len(nodes)),
             '--nproc_per_node',str(len(node['gpus'])),'--node_rank',str(rank),
             '--master_addr',spec['master_addr'],'--master_port',str(spec['master_port']),
             '--max_restarts','0']
        if mode=='communication':
            cmd += [str(ROOT/'tools/planreg_v2_cluster.py'),'--communication-worker','--result-root',spec['log_dir']]
        else:
            cmd += ['scripts/train_planreg_v2.py','--config',spec['config'],'--manifest',spec['manifest'],
                '--output',spec['run_output'],'--microbatch',str(spec['microbatch']),
                '--accumulate',str(spec['accumulate']),'--workers',str(spec['workers']),'--seed',str(spec.get('seed',0))]
            if mode=='profile':cmd += ['--profile-only']
            else:
                cmd += ['--layout-lock',spec['layout']]
                if spec.get('resume_checkpoint'):cmd += ['--resume',spec['resume_checkpoint']]
        extra=dict(spec.get('environment',{}))
        if mode=='formal':extra['LAUNCH_FORMAL']='1'
        commands.append(command_on_node(node,cmd,extra))
    return commands


def coordinate(spec):
    commands=training_commands(spec);logs=Path(spec['log_dir'])
    logs.mkdir(parents=True,exist_ok=True)
    if (logs/'coordinator.json').exists():raise FileExistsError('Unique cohort log directory required')
    processes=[];handles=[]
    report=dict(mode=spec['mode'],nodes=spec['nodes'],commands=commands,start_time=time.time(),status='RUNNING')
    (logs/'coordinator.json').write_text(json.dumps(report,indent=2))
    try:
        for i,command in enumerate(commands):
            handle=(logs/('node%02d.log'%i)).open('x');handles.append(handle)
            processes.append(subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True))
        report['local_transport_pids']=[p.pid for p in processes]
        (logs/'coordinator.json').write_text(json.dumps(report,indent=2))
        while True:
            codes=[p.poll() for p in processes]
            if any(c not in (None,0) for c in codes):raise RuntimeError('One cohort member failed: '+str(codes))
            if all(c==0 for c in codes):break
            time.sleep(2)
        report.update(status='COMPLETED',exit_codes=codes)
    except BaseException as exc:
        # Stop only this unique, recorded torchrun cohort, including remote parents.
        # Closing an SSH transport alone need not terminate its remote CUDA workers.
        target=spec['log_dir'] if spec['mode']=='communication' else spec['run_output']
        for node in spec['nodes']:
            try:
                subprocess.run(command_on_node(node,[sys.executable,ROOT/'tools/planreg_v2_cluster.py',
                    '--stop-cohort',target]),cwd=ROOT,timeout=20,check=True,capture_output=True)
            except (OSError,subprocess.SubprocessError):pass
        # Only our recorded transport process groups; never kill a GPU-wide PID list.
        for p in processes:
            if p.poll() is None:
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        report.update(status='FAILED',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        for handle in handles:handle.close()
        report['end_time']=time.time();(logs/'coordinator.json').write_text(json.dumps(report,indent=2))


def launch_formal(request, admission):
    if os.getenv('START_FORMAL_AFTER_PREFLIGHT')!='1' or admission.get('READY_FOR_FORMAL_TRAINING') is not True:
        raise ValueError('Both explicit start and successful actual preflight required')
    spec=dict(mode='formal',nodes=admission['nodes'],config=request['config'],manifest=request['manifest'],
        layout=admission['layout_path'],run_output=request['run_output'],microbatch=admission['microbatch'],
        accumulate=admission['accumulate'],workers=request['workers'],master_addr=request['master_addr'],
        master_port=request['master_port'],log_dir=request['launch_log_dir'],environment=request['environment'],seed=request.get('seed',0))
    if request.get('resume_checkpoint'):spec['resume_checkpoint']=request['resume_checkpoint']
    path=Path(request['launch_spec_path'])
    if path.exists():raise FileExistsError('Explicit new launch attempt path required')
    path.write_text(json.dumps(spec,indent=2))
    stdout=Path(request['launch_log_dir']+'.log').open('x')
    proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--coordinate',str(path)],cwd=ROOT,
                          stdout=stdout,stderr=subprocess.STDOUT,start_new_session=True)
    stdout.close()
    return dict(launch_requested=True,training_started=False,optimizer_steps_observed=0,
                coordinator_pid=proc.pid,launch_spec_path=str(path),run_output=request['run_output'])


def stop_cohort(target):
    if not Path(target).is_absolute() or len(Path(target).parts)<5:
        raise ValueError('An exact unique cohort output path is required')
    stopped=[]
    for process in Path('/proc').glob('[0-9]*'):
        try:
            argv=process.joinpath('cmdline').read_bytes().decode().split('\0')
            if 'torch.distributed.run' not in argv or target not in argv:continue
            idx=argv.index(target)
            if idx==0 or argv[idx-1] not in ('--output','--result-root'):continue
            env=process.joinpath('environ').read_bytes().decode().split('\0')
            if 'CUDA_VISIBLE_DEVICES='+os.environ['CUDA_VISIBLE_DEVICES'] not in env:continue
            pid=int(process.name);os.kill(pid,signal.SIGTERM);stopped.append(pid)
        except (OSError,UnicodeError,ValueError):continue
    print(json.dumps(dict(exact_cohort=target,stopped_torchrun_pids=stopped)))


def communication_worker(result_root):
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl',timeout=timedelta(seconds=120))
    rank=dist.get_rank();world=dist.get_world_size()
    x=torch.tensor([rank+1.],device='cuda');dist.all_reduce(x)
    if float(x)!=world*(world+1)/2:raise AssertionError('Rank assignment/reduction failed')
    payload=torch.ones(8*1024*1024,device='cuda')
    for _ in range(2):dist.all_reduce(payload);payload.fill_(1)
    torch.cuda.synchronize();start=time.perf_counter()
    for _ in range(8):dist.all_reduce(payload);payload.fill_(1)
    torch.cuda.synchronize()
    Path(result_root, 'rank%03d.json'%rank).write_text(json.dumps(dict(status='PASS',rank=rank,world=world,
        local_rank=int(os.environ['LOCAL_RANK']),hostname=__import__('socket').gethostname(),
        visible=os.environ['CUDA_VISIBLE_DEVICES'],all_reduce_32mib_seconds=(time.perf_counter()-start)/8)))
    dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--coordinate');p.add_argument('--inspect')
    p.add_argument('--stop-cohort')
    p.add_argument('--communication-worker',action='store_true');p.add_argument('--result-root');p.add_argument('--output')
    a=p.parse_args()
    if a.stop_cohort:stop_cohort(a.stop_cohort)
    elif a.communication_worker:communication_worker(a.result_root)
    elif a.coordinate:coordinate(json.loads(Path(a.coordinate).read_text()))
    elif a.inspect:
        report=inspect_nodes(json.loads(Path(a.inspect).read_text()))
        Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    else:p.error('Select an explicit bounded/profile/formal operation')
