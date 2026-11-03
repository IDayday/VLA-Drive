"""Single CPU owner: adopt AUGUSTUS q1 -> own q2 -> independent IL46 formal.

At most two fixed routes. Adopts exact qualified UUID/lease worker receipts,
never launches a new GPU job, signals processes, or touches old CAMPAIGN. Ambiguous/failed gangs
are retained for review; a durable launch intent is never blindly repeated.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import traceback
import yaml
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_world_compare.contracts import allowed_hosts,file_sha,fingerprint

ROOT=Path('/mnt/project/d46-priority-rl-20261103')
PYTHON='/mnt/project/flow-safe-improvement-20261013/venv/bin/python'
TASK='d46_priority_rl_20261103'
SCI_SHA='78cea8f14907762f790bed87684cf87612a42456'
WORKER_SHA='81c997e8aa927e3eba70ffc56e42fa9204047fde'
MODEL_SHA='386a0a27bd197cb002501bcb90c343659001d2e6d4f7bad0cbd5e31a91f8477d'
SCHEMA='d46_augustus_two_route_native_owner_v1'
TERMINAL={'REVIEW_REQUIRED','COMPLETE_6E','STOPPED_OWNER'}


def read(path,default=None):return json.loads(Path(path).read_text())if Path(path).is_file()else default

def head(path):return subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()


def command(route,node,phase):
    resume=phase=='qualification_resume';qual=phase!='formal'
    output=route['output']if qual else route['formal_output']
    value=['/usr/bin/env','NCCL_IB_DISABLE=1','NCCL_SOCKET_IFNAME=eth0',PYTHON,'-m','torch.distributed.run',
        f"--nnodes={len(route['hosts'])}",'--nproc_per_node=8',f'--node_rank={node}',f"--master_addr={route['master_addr']}",
        f"--master_port={route.get('phase_ports',{}).get(phase,route['port'])}",'--rdzv_conf=timeout=180','--max_restarts=0','-m','tools.d46_augustus.train',
        '--config',route['config'],'--output',output]
    if qual:value+=['--qualification']
    if resume:value+=['--resume',str(Path(route['output'])/'checkpoints/update000001_inner0')]
    if qual:value+=['--stop-after-update','2'if resume else'1']
    return value


def receipt_matches(receipt,route,node,phase,job):
    if(receipt.get('host')!=route['hosts'][node]or Path(receipt.get('job','/missing')).resolve()!=Path(job).resolve()
        or receipt.get('command')!=command(route,node,phase)or Path(receipt.get('source','/missing')).resolve()!=Path(route['source']).resolve()
        or receipt.get('source_sha')!=SCI_SHA or receipt.get('worker_source_sha')!=WORKER_SHA
        or Path(receipt.get('worker_source','/missing')).resolve()!=Path(route['worker_source']).resolve()
        or receipt.get('gpus')!=list(range(8))or len(receipt.get('expected_gpu_uuids',[]))!=8):
        raise ValueError('Foreign/mismatched AUGUSTUS node launch receipt')
    return receipt


def load(plan_path,launches_path,root=ROOT):
    root=Path(root).resolve();plan=read(plan_path);launches=read(launches_path)
    if plan.get('maximum_simultaneous_routes')!=2 or not 1<=len(plan.get('routes',[]))<=2:raise ValueError('MAX2 AUGUSTUS routes')
    ids=[r['experiment']for r in plan['routes']]
    if len(set(ids))!=len(ids)or set(ids)-{'AUG1_FULL','AUG2_LORA'}:raise ValueError('Only two named AUGUSTUS experiments')
    out=[];seenhosts=set()
    for original in plan['routes']:
        r=dict(original);r['formal_output']=str(root/'runs'/(r['experiment']+'_v1'))
        cfg=yaml.safe_load(Path(r['config']).read_text())
        expectedworld=32 if r['experiment']=='AUG1_FULL'else 16
        if(r['world']!=expectedworld or len(r['hosts'])!=expectedworld//8 or set(r['hosts'])&seenhosts
            or r['qualification']is not True or r['target_update']!=1 or cfg.get('experiment_id')!=r['experiment']
            or cfg.get('inner_updates')!=1 or cfg.get('target_epochs')!=6. or cfg.get('group')!=8
            or file_sha(r['config'])!=r['config_sha256']or head(r['source'])!=SCI_SHA or not SCI_SHA.startswith(r['source_sha'])
            or head(r['worker_source'])!=WORKER_SHA):raise ValueError('Changed fixed source/config/world/host plan')
        seenhosts.update(r['hosts']);nodes=[]
        for node,host in enumerate(r['hosts']):
            rows=[x for x in launches.get('receipts',[])if x.get('host')==host and '--output'in x.get('command',[])and x['command'][x['command'].index('--output')+1]==r['output']]
            if len(rows)!=1:raise ValueError('Missing/duplicate existing qualification1 receipt')
            job=root/'jobs/AUGUSTUS_20261103'/r['experiment']/'qualification1'/f'node{node}'
            receipt=receipt_matches(rows[0],r,node,'qualification1',job)
            nodes.append(dict(host=host,job=str(job),command=receipt['command'],phase='DISPATCHED',receipt=receipt))
        out.append(dict(experiment=r['experiment'],route=r,formal_output=r['formal_output'],status='QUALIFICATION1',phase='qualification1',
            stages={'qualification1':dict(nodes=nodes,adopted=True)},proofs={},events=[]))
    return dict(schema=SCHEMA,root=str(root),plan_path=str(Path(plan_path).resolve()),plan_sha256=file_sha(plan_path),
        launches_path=str(Path(launches_path).resolve()),launches_sha256=file_sha(launches_path),status='RUNNING',routes={r['experiment']:r for r in out})


_NATIVE=r'''
import json,sys,torch,hashlib
from pathlib import Path
source,cp,cfgpath,world,updates,qualification=sys.argv[1:];sys.path.insert(0,source)
from starVLA.rl.d46_augustus.runtime import AugustusBoundary
from starVLA.rl.d46_augustus.config import read,SCHEMA
from starVLA.rl.qwen_world_compare.contracts import fingerprint,file_sha
from starVLA.rl.qwen_world_compare.recovery_state import tensor_digest
cp=Path(cp);cfg=read(cfgpath);world=int(world);updates=int(updates);qualification=qualification=='True'
seal=json.loads((cp/'COMPLETE').read_text());files=json.loads((cp/'files.json').read_text());identity=json.loads((cp/'identity.json').read_text())
body=dict(identity);digest=body.pop('sha256')
assert fingerprint(body)==digest and identity['algorithm_schema']==SCHEMA and identity['recipe']==cfg
assert identity['world']==world and identity['qualification']is qualification and identity['fresh_Adam']is True
assert identity['parent_model_sha256']=='386a0a27bd197cb002501bcb90c343659001d2e6d4f7bad0cbd5e31a91f8477d'
assert seal['schema']==2 and seal['contract']=='qwen_world_abcd_research_checkpoint_v1'and seal['world']==world
assert seal['phase']=='READY'and seal['next_inner']==0 and seal['updates']==updates and seal['files_sha256']==file_sha(cp/'files.json')
assert {'identity.json','optimizer.pt','transaction.json'}|{f'rank{r}.pt'for r in range(world)}<=files.keys()
for name,sha in files.items():
 assert Path(name).name==name and(cp/name).resolve().is_relative_to(cp)and file_sha(cp/name)==sha
for name,sha in identity['source_files'].items():assert file_sha(Path(source)/name)==sha
common=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False,mmap=True)
assert set(common['adapters'])==set(identity['scope']['trainable_names'])
assert common['scheduler']['last_epoch']==updates
assert bool(common['optimizer']['state']) is(updates>0)
tx=json.loads((cp/'transaction.json').read_text());assert len(tx['ranks'])==world
rows=[];rng=[];states=[]
for rank in range(world):
 own=torch.load(cp/f'rank{rank}.pt',map_location='cpu',weights_only=False,mmap=True)
 assert own['identity_sha256']==digest and own['rollout']is None
 row=AugustusBoundary.validate_boundary(own['state'],None,identity,common['optimizer'],common['scheduler'])
 assert row==tx['ranks'][rank]and row['updates']==updates
 assert set(own['rng'])=={'python','numpy','torch','cuda','rollout_generator'}
 torch.Generator().set_state(own['rng']['rollout_generator']);rng.append(tensor_digest(own['rng']))
 if updates==0:
  assert own['state']['scene_cursor']==0 and own['state']['new_candidates']==0 and own['state']['collections']==0
  assert torch.equal(own['rng']['rollout_generator'],torch.Generator().manual_seed(420000+rank).get_state())
 rows.append(row);states.append(row['valid_scene_exposures'])
AugustusBoundary.validate_ranks(rows)
print(json.dumps(dict(status='PASS',checkpoint=str(cp),files_sha256=seal['files_sha256'],identity_sha256=digest,
 world=world,updates=updates,phase='READY',next_inner=0,scope=identity['scope'],source_files=identity['source_files'],
 optimizer_empty=not bool(common['optimizer']['state']),scheduler_updates=common['scheduler']['last_epoch'],
 allrank_RNG_sha256=rng,valid_scene_exposures=states[0],qualification=qualification)))
'''


def review_native(route,output,updates,qualification):
    cp=Path(output)/'checkpoints'/f'update{updates:06}_inner0'
    value=subprocess.run([PYTHON,'-c',_NATIVE,route['source'],str(cp),route['config'],str(route['world']),str(updates),str(qualification)],
        cwd=route['source'],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=route['source']),text=True,capture_output=True,timeout=1800)
    if value.returncode:raise RuntimeError('Native CPU validation failed: '+value.stderr[-5000:])
    return json.loads(value.stdout)


class Owner:
    def __init__(self,state,directory):self.state=state;self.directory=Path(directory);self.lock=threading.RLock();self.stop=threading.Event()
    def persist(self):
        with self.lock:
            self.state['updated']=time.time();review=any(r['status']=='REVIEW_REQUIRED'for r in self.state['routes'].values());active=any(r['status']not in TERMINAL for r in self.state['routes'].values())
            self.state['status']='RUNNING_WITH_REVIEW_REQUIRED'if review and active else'REVIEW_REQUIRED'if review else'RUNNING'if active else'COMPLETE'
            atomic_json(self.directory/'STATE.json',self.state)
    def event(self,row,name,**kw):row['events'].append(dict(at=time.time(),event=name,**kw));self.persist()
    def peer_stop(self,row):
        for stage in row['stages'].values():
            for node in stage['nodes']:
                if node.get('receipt')and not(Path(node['job'])/'RESULT.json').exists():(Path(node['job'])/'STOP_REQUESTED').touch()
    def fail(self,row,error):
        row.update(status='REVIEW_REQUIRED',error=str(error));self.event(row,'review_required',error=str(error))
        # Metadata/controller bugs must not interrupt healthy native learners.
        actual=row.get('actual_child_failure',{})
        if actual.get('phase')==row['phase']and actual.get('returncode')not in(None,0):self.peer_stop(row)
    def external_resume(self,row,phase='qualification_resume'):
        path=self.directory/('FORMAL_LAUNCHES.json'if phase=='formal'else'QUALIFICATION_RESUME_LAUNCHES.json');value=read(path,{})
        receipts=value if isinstance(value,list)else value.get('receipts',[])
        route=row['route'];nodes=[]
        for n,host in enumerate(route['hosts']):
            job=Path(self.state['root'])/'jobs/AUGUSTUS_20261103'/route['experiment']/phase/f'node{n}'
            found=[r for r in receipts if r.get('job')==str(job)]
            actual=found[0]if len(found)==1 else read(job/'LAUNCH.json')
            if len(found)>1:raise ValueError('Duplicate external qresume receipts')
            if actual:
                port=int(next(x.split('=',1)[1]for x in actual['command']if x.startswith('--master_port=')))
                oldport=route.setdefault('phase_ports',{}).setdefault(phase,port)
                if oldport!=port:raise ValueError('Mixed rendezvous ports within the exact adopted gang')
                receipt_matches(actual,route,n,phase,job)
            nodes.append(dict(host=host,job=str(job),command=command(route,n,phase),phase='DISPATCHED'if actual else'NOT_LAUNCHED',receipt=actual))
        return nodes
    def new_stage(self,row,phase):
        route=row['route'];nodes=self.external_resume(row,phase)
        if not nodes:
            for n,host in enumerate(route['hosts']):
                job=Path(self.state['root'])/'jobs/AUGUSTUS_20261103'/route['experiment']/phase/f'node{n}'
                nodes.append(dict(host=host,job=str(job),command=command(route,n,phase),phase='NOT_LAUNCHED'))
        if phase=='formal':
            output=Path(row['formal_output'])
            if output.exists()and any(output.iterdir())and not any(n.get('receipt')for n in nodes):raise ValueError('Formal fresh root already contains files without an owner intent')
        row['stages'][phase]=dict(nodes=nodes,created=time.time());row['phase']=phase;row['status']='PREPARING_'+phase.upper()
        self.event(row,'stage_intent',phase=phase)
    def reconcile(self,row,node,n):
        if node.get('receipt'):return
        job=Path(node['job']);actual=read(job/'LAUNCH.json')
        if actual:
            node.update(receipt=receipt_matches(actual,row['route'],n,row['phase'],job),phase='DISPATCHED');self.persist();return
        if node['phase']=='LAUNCH_INTENT':raise RuntimeError('Unresolved durable launch intent; never guess or redo remote Popen')
        if any((job/x).exists()for x in('RUNNING.json','RESULT.json','DISPATCH.json','ADMISSION_WAIT.json')):raise RuntimeError('Remote worker evidence lacks exact LAUNCH; review required')
    def launch_stage(self,row):
        # Current campaign is already launched by Root: admission is adoption
        # only. No remote dispatch, host probes, or guessed missing-node retry.
        stage=row['stages'][row['phase']]
        refreshed=self.external_resume(row,row['phase'])
        for n,node in enumerate(stage['nodes']):
            if not node.get('receipt')and refreshed[n].get('receipt'):node.update(refreshed[n])
            self.reconcile(row,node,n)
        row['status']=('FORMAL_RUNNING'if row['phase']=='formal'else row['phase'].upper())if all(n.get('receipt')for n in stage['nodes'])else'WAITING_ROOT_RECEIPTS'
        self.persist()
    def completed_nodes(self,row):
        results=[]
        for node in row['stages'][row['phase']]['nodes']:
            value=read(Path(node['job'])/'RESULT.json')
            if value:
                if value.get('command')!=node['command']or value.get('cwd')!=row['route']['source']or value.get('host')!=node['host']:
                    raise ValueError('Foreign node RESULT identity')
                code=value.get('returncode')
                if type(code)is not int:raise ValueError('RESULT has no actual integer child exit code')
                if code!=0:
                    row['actual_child_failure']=dict(phase=row['phase'],returncode=code,path=str(Path(node['job'])/'RESULT.json'),
                        file_sha256=file_sha(Path(node['job'])/'RESULT.json'),observed=time.time())
                    raise RuntimeError('Preserved child failure/resource refusal: '+str(Path(node['job'])/'RESULT.json'))
            results.append(value)
        return bool(results)and all(x is not None for x in results)
    def qualification(self,row,updates):
        route=row['route'];output=Path(route['output']);results=[]
        for rank in range(route['world']):
            p=output/f'result_rank{rank}.json';r=read(p,{})
            if(r.get('rank')!=rank or r.get('updates')!=updates or r.get('status')!='QUALIFICATION_STOPPED'or r.get('qualification')is not True
                or not r.get('frozen_before')or r['frozen_before']!=r.get('frozen_after')or r.get('phase')!='READY'or r.get('next_inner')!=0):raise ValueError('Missing/failed allrank qualification/frozen hash')
            results.append(dict(rank=rank,file_sha256=file_sha(p),frozen_before=r['frozen_before'],frozen_after=r['frozen_after']))
        proof=review_native(route,output,updates,True)
        if updates==2 and row['proofs']['q1'].get('rank_results')and [r['frozen_before']for r in results]!=[r['frozen_before']for r in row['proofs']['q1']['rank_results']]:raise ValueError('Frozen parent/reference changed across own-native qresume')
        proof['rank_results']=results;row['proofs']['q'+str(updates)]=proof;self.event(row,'qualification_complete',updates=updates)
    def progress(self,row):
        route=row['route'];root=Path(row['formal_output']);point=read(root/'latest.json',{})
        if point:
            cp=Path(point['checkpoint']).resolve();needroot=root.resolve()/'checkpoints'
            if not cp.is_relative_to(needroot):raise ValueError('Foreign formal native pointer')
            seal=read(cp/'COMPLETE',{});files=read(cp/'files.json',{});identity=read(cp/'identity.json',{})
            if(seal.get('files_sha256')!=file_sha(cp/'files.json')or files.get('identity.json')!=file_sha(cp/'identity.json')or identity.get('qualification')is not False
                or identity.get('recipe')!=yaml.safe_load(Path(route['config']).read_text())or identity.get('world')!=route['world']):raise ValueError('Changed formal native metadata')
            import torch
            if file_sha(cp/'rank0.pt')!=files.get('rank0.pt'):raise ValueError('Changed latest native rank0 bytes')
            ns=torch.load(cp/'rank0.pt',map_location='cpu',weights_only=False,mmap=True)['state']
            row['latest_native']=dict(checkpoint=str(cp),updates=seal['updates'],next_inner=seal['next_inner'],phase=seal['phase'],files_sha256=seal['files_sha256'],valid_scene_exposures=ns['valid_scene_exposures'],scene_cursor=ns['scene_cursor'])
        rows=[]
        for rank in range(route['world']):
            p=root/f'metrics_rank{rank}.jsonl'
            if not p.exists():rows.append(dict(rank=rank,updates=0));continue
            with p.open('rb')as f:f.seek(0,2);n=f.tell();f.seek(max(0,n-16384));lines=f.read().splitlines()
            last=None
            for line in reversed(lines):
                try:last=json.loads(line);break
                except ValueError:continue
            rows.append(dict(rank=rank,updates=last['updates'],scene_exposures=last['scene_exposures'],at=last['at'])if last else dict(rank=rank,updates=0))
        row['actual_allrank_progress']=dict(min_updates=min(x['updates']for x in rows),max_updates=max(x['updates']for x in rows),ranks=rows)
        cp0=root/'checkpoints/update000000_inner0'
        if 'formal_native0'not in row['proofs']and(cp0/'COMPLETE').exists():
            proof=review_native(route,root,0,False)
            if not proof['optimizer_empty']or proof['scheduler_updates']!=0:raise ValueError('Formal run was not an empty-Adam IL46 fresh start')
            row['proofs']['formal_native0']=proof;self.event(row,'fresh_formal_native0_complete')
        self.persist()
    def tick(self,row):
        if row['status']in TERMINAL:return
        route=row['route']
        if file_sha(route['config'])!=route['config_sha256']or head(route['source'])!=SCI_SHA or head(route['worker_source'])!=WORKER_SHA:raise ValueError('Sealed source/config changed')
        phase=row['phase'];stage=row['stages'][phase]
        if phase=='qualification1'and all(n.get('receipt')for n in self.external_resume(row)):
            if not self.completed_nodes(row):return
            proof=review_native(route,route['output'],1,True)
            proof['rank_results']=[];proof['q1_result_files_overwritten_by_explicit_own_resume']=True
            proof['frozen_guard_evidence']='all original q1 workers EXIT0 under pinned source strict frozen-before/after assertion; current rank results belong to q2'
            row['proofs']['q1']=proof;self.new_stage(row,'qualification_resume');return
        for n,node in enumerate(stage['nodes']):self.reconcile(row,node,n)
        # Harvest failures before any remaining gang launches.
        done=self.completed_nodes(row)
        if phase=='formal':self.progress(row)
        if not all(n.get('receipt')for n in stage['nodes']):self.launch_stage(row);return
        if not done:return
        if phase=='qualification1':self.qualification(row,1);self.new_stage(row,'qualification_resume');self.launch_stage(row)
        elif phase=='qualification_resume':self.qualification(row,2);self.new_stage(row,'formal');self.launch_stage(row)
        else:
            if 'formal_native0'not in row['proofs']:raise ValueError('Formal terminal missing initial native0 proof')
            if(row.get('latest_native',{}).get('phase')!='READY'or row['latest_native'].get('valid_scene_exposures',0)<6*103288):raise ValueError('Formal terminal before registered6E; preserve native for review')
            row['status']='COMPLETE_6E';self.event(row,'formal_finished_no_auto_restart')
    def route_loop(self,row,interval):
        while not self.stop.is_set()and not(self.directory/'STOP_OWNER').exists()and row['status']not in TERMINAL:
            try:self.tick(row)
            except BaseException:self.fail(row,traceback.format_exc());break
            self.stop.wait(interval)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=('prepare','run','once'));p.add_argument('--root',default=str(ROOT));p.add_argument('--directory');p.add_argument('--interval',type=float,default=20.);a=p.parse_args()
    root=Path(a.root).resolve();directory=Path(a.directory).resolve()if a.directory else root/'operations/AUGUSTUS_CAMPAIGN_20261103'
    directory.mkdir(parents=True,exist_ok=True)
    with(directory/'owner.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        initial=load(directory/'QUALIFICATION_PLAN.json',directory/'QUALIFICATION_LAUNCHES.json',root)
        state=read(directory/'STATE.json')or initial
        if(state.get('schema')!=SCHEMA or state['plan_sha256']!=initial['plan_sha256']or state['launches_sha256']!=initial['launches_sha256']):raise ValueError('Owner state bound to another initial plan')
        atomic_json(directory/'OWNER.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),host=socket.gethostname(),
            source=str(Path(__file__).resolve()),source_file_sha256=file_sha(__file__),started=time.time(),maximum_routes=2))
        owner=Owner(state,directory);owner.persist()
        if a.action=='prepare':return
        if a.action=='once':
            for row in state['routes'].values():
                try:owner.tick(row)
                except BaseException:owner.fail(row,traceback.format_exc())
            return
        with ThreadPoolExecutor(max_workers=2)as pool:
            futures=[pool.submit(owner.route_loop,row,max(2.,a.interval))for row in state['routes'].values()]
            try:
                while not all(f.done()for f in futures):time.sleep(1)
            finally:owner.stop.set();owner.persist()


if __name__=='__main__':main()
