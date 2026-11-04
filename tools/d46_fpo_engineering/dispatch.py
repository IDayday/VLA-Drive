"""Explicit gang launch and exact owned-process shutdown for native recovery."""
import json
from pathlib import Path
import subprocess
import time
from starVLA.rl.flow_grpo.transactions import atomic_json
from tools.d44_native_rl.launch import launch

ROUND=Path('/mnt/project/d46-fpo-two-routes-20261104')
ROOT=Path('/mnt/project/d46-priority-rl-20261103')
PYTHON='/mnt/project/flow-safe-improvement-20261013/venv/bin/python'
TASK='d46_fpo_two_routes_20261104'


def gang(source,output,hosts,master_address,*,attempt,port):
    output=Path(output);source=Path(source)
    identity=json.loads((output/'identity.json').read_text())
    if len(hosts)*8!=identity['world'] or len(set(hosts))!=len(hosts):raise ValueError('World/host mismatch')
    resume=json.loads((output/'latest.json').read_text())['checkpoint']
    receipts=[]
    receipt_path=ROUND/'operations'/('MATURE_'+attempt+'_LAUNCHES.json')
    for rank,host in enumerate(hosts):
        command=['env','NCCL_IB_DISABLE=1','NCCL_SOCKET_IFNAME=eth0',PYTHON,'-m','torch.distributed.run',
            '--nnodes='+str(len(hosts)),'--nproc_per_node=8','--node_rank='+str(rank),
            '--master_addr='+master_address,'--master_port='+str(port),'--max_restarts=0',
            '-m','tools.d46_fpo_engineering.train','--config',str(ROOT/'registered_configs/D3_FPO.yaml'),
            '--output',str(output),'--resume',resume]
        receipt=launch(host,source,ROUND/'jobs/FPO_MATURE'/attempt/('node'+str(rank)),command,list(range(8)),
            share_min_free_mib=45000,worker_source=source,task_root=ROOT,task=TASK)
        receipts.append(receipt);atomic_json(receipt_path,dict(receipts=receipts))
    plan_path=ROUND/'operations/PLAN.json';plan=json.loads(plan_path.read_text())
    route=plan['routes']['FPO_MATURE']
    route.update(source=str(source),output=str(output),world=identity['world'],hosts=hosts,
        launch_receipts=str(receipt_path),start_update=identity['performance_execution']['origin_updates'])
    proof=ROUND/'operations/ACTUAL_START_FPO_MATURE.json'
    if proof.exists():proof.rename(proof.with_name('ACTUAL_START_FPO_MATURE_'+attempt+'_previous.json'))
    atomic_json(plan_path,plan)
    return receipts


def stop_owned(receipts,*,force=False,signal_name='SIGTERM'):
    if signal_name not in('SIGTERM','SIGKILL'):raise ValueError('Invalid owned recovery signal')
    results=[]
    for receipt in receipts:
        job=Path(receipt['job']);(job/'STOP_REQUESTED').touch()
        if not force or not(job/'RUNNING.json').exists():continue
        state=json.loads((job/'RUNNING.json').read_text())
        code='''import os,json,signal,pathlib
s=STATE
targets=[dict(pid=s['child_pid'],start_ticks=s['child_start_ticks'],pgid=s['child_pid'])]+s.get('known_rank_processes',[])
for row in targets:
 p=pathlib.Path('/proc')/str(row['pid'])
 if not p.exists():continue
 try:
  v=(p/'stat').read_text().split()
  if int(v[21])!=row['start_ticks']or os.getpgid(row['pid'])!=row['pgid']:continue
  cmd=(p/'cmdline').read_bytes().decode().split(chr(0))
  if '--output'in cmd and cmd[cmd.index('--output')+1]==s['command'][s['command'].index('--output')+1]:
   os.killpg(row['pgid'],getattr(signal,SIGNAL));print(SIGNAL+' owned group');break
 except(ProcessLookupError,FileNotFoundError):continue
'''.replace('STATE',repr(state)).replace('SIGNAL',repr(signal_name))
        process=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',receipt['host'],'/usr/bin/python3','-'],input=code,text=True,capture_output=True,timeout=20)
        results.append(dict(host=receipt['host'],returncode=process.returncode,stdout=process.stdout,stderr=process.stderr))
    return results
