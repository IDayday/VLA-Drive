"""Plan two admitted workers, or exec one participant inside its held leases.

No SSH, resource acquisition, process signalling or controller mutations. Root
must admit BOTH hosts/8 UUIDs via the updated worker before running participants.
The participant execs torchrun in its inherited PGID, so no detached launcher.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from starVLA.rl.qwen_world_compare.contracts import file_sha,require_dispatch
from starVLA.rl.d46_world16.runtime import receipt


def build(root,config,source,hosts,port,stop,master_address=None):
    root=Path(root).resolve();source=Path(source).resolve();identity=json.loads((root/'identity.json').read_text());receipt(root,identity)
    if len(hosts)!=2 or len(set(hosts))!=2 or any(not x.startswith('training-')for x in hosts)or not 1024<=port<=65535:raise ValueError('Two explicit hosts and legal rendezvous port required')
    if identity['recipe']!=__import__('yaml').safe_load(Path(config).read_text()):raise ValueError('Config changed')
    origin=identity['native_world_transition']['origin_updates']
    if not origin<stop<=origin+10:raise ValueError('Gang tool is finite qualification only, at most10 inherited Adam updates')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip():raise ValueError('Freeze source before gang preparation')
    master_address=master_address or hosts[0]
    if any(x.isspace()for x in master_address)or not master_address:raise ValueError('Invalid explicit master address')
    resume=json.loads((root/'latest.json').read_text())['checkpoint'];participants=[]
    for node in range(2):
        command=[sys.executable,'-m','torch.distributed.run','--nnodes=2','--nproc_per_node=8',f'--node_rank={node}',f'--master_addr={master_address}',f'--master_port={port}',
            '--max_restarts=0','--rdzv_conf=timeout=120','-m','tools.d46_world16.train','--config',str(Path(config).resolve()),'--output',str(root),'--resume',resume,'--stop-after-update',str(stop)]
        if identity['qualification']:command.append('--qualification')
        participants.append(dict(host=hosts[node],node_rank=node,command=command,environment={'NCCL_IB_DISABLE':'1','NCCL_SOCKET_IFNAME':'eth0'},physical_GPUs='0..7 only as mapped/leased by Root worker',lease_must_be_held_by_worker=True))
    return dict(schema='d46_two_node_world16_finite_gang_plan_v1',source=str(source),source_commit=head,identity_sha256=identity['sha256'],config_file_sha256=file_sha(config),world=16,
        global_scenes=64,group=16,inner_updates=2,origin_updates=origin,stop_after_update=stop,participants=participants,
        no_local_dispatch=True,worker_requirement='updated exact tools.d46_world16.train ownership+native16 admission, both host grants/UUID/leases/headroom; old world8 owner must not adopt',
        no_detached_subprocess=True,qualified_throughput_claim=False)


def participate(path,node,expected):
    if file_sha(path)!=expected:raise ValueError('Pinned gang plan changed')
    plan=json.loads(Path(path).read_text());part=plan['participants'][node]
    if plan.get('world')!=16 or part['node_rank']!=node or require_dispatch()!=part['host']:raise ValueError('Wrong admitted participant host')
    source=Path(plan['source']);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    if head!=plan['source_commit']or subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip():raise ValueError('Frozen gang source changed')
    if len(os.environ.get('CUDA_VISIBLE_DEVICES','').split(','))!=8:raise ValueError('Existing worker must expose exactly8 leased GPUs')
    if os.environ.get('WORLD16_GANG_PLAN_SHA256')!=expected:raise ValueError('Root worker must explicitly opt in to this exact gang')
    environment=os.environ.copy();environment.update(part['environment'])
    os.chdir(source);os.execvpe(part['command'][0],part['command'],environment)


def main():
    p=argparse.ArgumentParser(__doc__);s=p.add_subparsers(dest='op',required=True)
    q=s.add_parser('prepare');q.add_argument('--root',required=True);q.add_argument('--config',required=True);q.add_argument('--source',required=True);q.add_argument('--hosts',nargs=2,required=True);q.add_argument('--port',type=int,required=True);q.add_argument('--master-address');q.add_argument('--stop-after-update',type=int,required=True);q.add_argument('--output',required=True)
    r=s.add_parser('participate');r.add_argument('--plan',required=True);r.add_argument('--node-rank',type=int,choices=(0,1),required=True);r.add_argument('--expected-sha256',required=True)
    a=p.parse_args()
    if a.op=='participate':participate(a.plan,a.node_rank,a.expected_sha256);return
    out=Path(a.output)
    if out.exists():raise FileExistsError('Keep earlier gang plan')
    value=build(a.root,a.config,a.source,a.hosts,a.port,a.stop_after_update,a.master_address);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'plan':str(out),'sha256':file_sha(out),'dispatched':False}))


if __name__=='__main__':main()
