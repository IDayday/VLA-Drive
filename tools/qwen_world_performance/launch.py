"""Registered fresh-IL70 performance campaign; no dispatch unless `run` is explicit.

The existing qualified Owner, transport, scorer and native trainer are reused.
Exposure gates differ across M2/M4 and S5's B64->B128 stage; no Adam resets.
"""
import argparse,copy,fcntl,hashlib,json,math,os,socket,subprocess,threading,time,traceback
from pathlib import Path
import yaml

ROOT=Path('/mnt/project/qwen-world-performance-rl-20261103')
TASK='qwen_world_safety_exploration_20261031'
SOURCE=ROOT/'runtime/formal_v1'
OWNER='PERFORMANCE_NATIVE'
ORDER=['S1_NOMINAL_RESCUE','S5_PERFORMANCE_MAX','S0_LARGE_SAFETYEP','S2_CONSERVATIVE_AWFM','S3_LARGE_FPO','S4_TEMPORAL_RESCUE','S0_LOW_LR']
EPOCHS=[.25,.5,1,2,3,4,6,8,12,20]
PROTECTED={'training-vla-zt','training-vlawm-tsinghua-zt'}

def read(p,default=None):return json.loads(Path(p).read_text())if Path(p).exists()else default
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def publish(p,value):
    from starVLA.rl.flow_grpo.transactions import atomic_json
    atomic_json(p,value)
def bootstrap():
    os.environ.update(WORLD_COMPARE_ROOT=str(ROOT),WORLD_COMPARE_TASK=TASK,CUDA_VISIBLE_DEVICES='')
    from starVLA.rl.qwen_world_safety_exploration import contracts
    if contracts.ROOT.resolve()!=ROOT:raise RuntimeError('Start in a fresh Python process; imported legacy ROOT differs')


def exposure_gates(cfg):
    """Simulate only scene counters, including epoch padding and stage B/M."""
    p=cfg['research']['performance'];n=p['population'];m=p['inner_updates'];u=valid=offset=epoch=0;out={};wanted=list(p['evaluation_epochs'])
    while wanted:
        stage=max((s for s in p['stages']if s['start_epoch']<=valid/n+1e-12),key=lambda s:s['start_epoch'])
        b=stage['B'];valid+=min(b,n-offset);offset+=b;u+=m
        if offset==math.ceil(n/b)*b:offset=0;epoch+=1
        while wanted and valid+1e-9>=wanted[0]*n:
            e=wanted.pop(0);out[str(e)]=dict(epoch=e,updates=u,valid_scene_exposures=valid,collections=u//m)
    if u!=cfg['max_steps']:raise ValueError('Config max_steps does not match the declared20epoch B/M clock')
    return out


def key(epoch):return str(float(epoch))if float(epoch).is_integer()else str(epoch)
def point(mapping,epoch):
    return next(v for v in mapping.values()if v['epoch']==epoch)


def prepare(source=SOURCE):
    """Only new-root config/registration metadata; never starts any worker."""
    from .configure import recipes
    from starVLA.rl.qwen_world_performance.contracts import validate_recipe
    from starVLA.rl.qwen_world_performance.data_protocol import read_sealed,fingerprint
    old=read('/mnt/project/qwen-world-safety-exploration-20261031/standard_training_20261101/RUNS.json')
    if(ROOT/'RUNS.json').exists():raise FileExistsError('Do not overwrite an existing campaign registration')
    manifest=ROOT/'data/NAVTRAIN_SCENES.json';audit=ROOT/'data/AUDIT.json'
    if read_sealed(audit)['status']!='COMPLETE':raise ValueError('Real full asset audit must be COMPLETE')
    cfgdir=ROOT/'configs';cfgdir.mkdir(parents=True,exist_ok=True)
    source=Path(source).resolve();pending=not source.exists()
    commit=None if pending else subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    from starVLA.rl.qwen_world_safety_exploration.contracts import allowed_hosts
    routes=[dict(host=h,lease_slot='primary',minimum_free_MiB=32768)for h in allowed_hosts(admit_long_job=True)if h not in PROTECTED]
    rows=[]
    for name,cfg in recipes(str(manifest),str(audit)):
        validate_recipe(cfg);arm=cfg['research']['arm'];path=cfgdir/(arm+'.yaml')
        if path.exists():raise FileExistsError('Config bytes must never be overwritten: '+str(path))
        path.write_text(yaml.safe_dump(cfg,sort_keys=False));mapping=exposure_gates(cfg)
        native=[point(mapping,e)['updates']for e in EPOCHS if e<=6]
        nav=[point(mapping,e)['updates']for e in EPOCHS if e<=6]
        rows.append(dict(id=arm,architecture_id='QWEN_WORLD_MEMORY',root=str(ROOT/'runs'/arm),source=str(source),
            source_pending=pending,source_commit=commit,worker_source=str(source),config=str(path),config_sha256=sha(path),
            parent_checkpoint=old['runs'][0]['parent_checkpoint'],group_size=cfg['research']['performance']['group_size'],
            inner_updates=cfg['research']['performance']['inner_updates'],exposure_mapping=mapping,
            native_gates=native,gates=nav,navtest_epochs=[e for e in EPOCHS if e<=6],
            NavtestSelected=True,target_updates=native[-1],target_epochs=6.,minimum_epochs=3.,
            optional_epoch_cap=20.,optional_extensions=[8.,12.,20.],trainer_module='tools.qwen_world_performance.train',
            save_every=400,owner_state=str(ROOT/'owners'/OWNER/'STATE.json'),status='REGISTERED',
            priority=ORDER.index(arm),deployment_candidates=copy.deepcopy(routes)))
    rows.sort(key=lambda r:r['priority'])
    r=dict(schema='world_performance_fresh_IL70_campaign_v1',root=str(ROOT),task=TASK,status='CONFIGURED_SOURCE_PENDING'if pending else'CONFIGURED_NOT_STARTED',
        created_unix=time.time(),training_population=103288,manifest=str(manifest),manifest_sha256=sha(manifest),
        observation_data=old['observation_data'],parent_selection=old['parent_selection'],
        evaluation=copy.deepcopy(old['evaluation']),exposure_clock='valid fresh scenes; source padding excluded',
        initial_target_epochs=6.,optional_epoch_cap=20.,minimum_epochs=3.,
        continuation='Numerically healthy own native progress continues through3/6 regardless early scores;6 requires review before extension',runs=rows)
    r['registration_sha256']=fingerprint({k:v for k,v in r.items()if k!='registration_sha256'})
    publish(ROOT/'RUNS.json',r);publish(ROOT/'REGISTRATION.json',r);return r


def seal_source():
    r=read(ROOT/'RUNS.json');source=Path(r['runs'][0]['source'])
    if not source.is_dir():raise ValueError('Root must freeze the real source first')
    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).splitlines()
    if any(line not in ('?? FROZEN_SOURCE.json','?? FROZEN_SOURCE')for line in dirty):raise ValueError('Scientific source must be clean before admission')
    for run in r['runs']:
        if Path(run['source'])!=source:raise ValueError('Mixed source binding')
        run.update(source_pending=False,source_commit=commit)
    r.update(status='SOURCE_FROZEN_NOT_STARTED',source_frozen_unix=time.time())
    publish(ROOT/'RUNS.json',r);publish(ROOT/'REGISTRATION.json',r);return validate_registry(r)


def validate_registry(r,*,allow_pending=False):
    from starVLA.rl.qwen_world_performance.contracts import validate_recipe
    if(r['task']!=TASK or Path(r['root']).resolve()!=ROOT or r['training_population']!=103288
       or{v['id']for v in r['runs']}!=set(ORDER)or len(r['runs'])!=7):raise ValueError('Wrong explicit7arm performance campaign')
    if sha(r['manifest'])!=r['manifest_sha256']:raise ValueError('Registered data manifest changed')
    for run in r['runs']:
        cfg=validate_recipe(yaml.safe_load(Path(run['config']).read_text()));mapping=exposure_gates(cfg);cap=run['target_epochs']
        nav_epochs=run['navtest_epochs']
        if sorted(set(nav_epochs))!=nav_epochs or not set(nav_epochs)<=set(e for e in EPOCHS if e<=cap)or cap not in nav_epochs:
            raise ValueError('Explicit registered full-NAV exposure cadence required')
        if cap not in(6,8,12,20)or run['optional_epoch_cap']!=20:raise ValueError('Only registered6initial/20optional cap')
        if(cfg['research']['arm']!=run['id']or sha(run['config'])!=run['config_sha256']
           or Path(run['root']).resolve()!=ROOT/'runs'/run['id']
           or run['owner_state']!=str(ROOT/'owners'/OWNER/'STATE.json')
           or run['trainer_module']!='tools.qwen_world_performance.train'
           or run['native_gates']!=[point(mapping,e)['updates']for e in EPOCHS if e<=cap]
           or run['gates']!=[point(mapping,e)['updates']for e in nav_epochs]
           or run['target_updates']!=point(mapping,cap)['updates']or run['exposure_mapping']!=mapping):raise ValueError('Run/config/clock/owner contract changed')
        if not allow_pending:
            if run.get('source_pending')or not run.get('source_commit')or not Path(run['source']).is_dir():raise ValueError('Actual frozen source required, no fabricated commit')
        if any(route['lease_slot']!='primary'for route in run['deployment_candidates']):raise ValueError('Native learners use free8GPU primary slots only')
    return r


def inspect_native(run,cp):
    """Lightweight publication/ownership check. Full tensor restore remains native."""
    cp=Path(cp).resolve();root=Path(run['root']).resolve()
    if not cp.is_relative_to(root/'checkpoints')or'.incomplete'in cp.name:raise ValueError('Foreign/incomplete checkpoint')
    seal=read(cp/'COMPLETE');files=read(cp/'files.json');identity=read(cp/'identity.json');tx=read(cp/'transaction.json')
    body=dict(identity);saved=body.pop('sha256',None)
    if hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=saved:raise ValueError('Native identity seal changed')
    required={'optimizer.pt','identity.json','transaction.json'}|{f'rank{r}.pt'for r in range(8)}
    if not required<=set(files)or any(Path(n).is_absolute()or not(cp/n).resolve().is_relative_to(cp)or not(cp/n).is_file()for n in files):raise ValueError('Native checkpoint inventory incomplete/escaping')
    if(seal.get('schema')!=2 or seal['world']!=8 or seal['files_sha256']!=sha(cp/'files.json')
       or identity!=read(root/'identity.json')or identity['recipe']!=yaml.safe_load(Path(run['config']).read_text())
       or identity.get('purpose')!='formal'or identity['world']!=8):raise ValueError('Native publication/identity changed')
    for name in sorted(required-{'optimizer.pt'}):
        if sha(cp/name)!=files[name]:raise ValueError('Native metadata payload changed')
    if(len(tx['ranks'])!=8 or [x.get('rank')for x in tx['ranks']]!=list(range(8))
       or any(x['updates']!=seal['updates']or x['next_inner']!=seal['next_inner']or x['phase']!=seal['phase']or x['run']!=identity['sha256']for x in tx['ranks'])):raise ValueError('Eight-rank native transaction mismatch')
    m=run['inner_updates']
    if not 0<=seal['next_inner']<m or seal['updates']%m!=seal['next_inner']:raise ValueError('Wrong M2/M4 inner cursor')
    return dict(path=str(cp),updates=seal['updates'],next_inner=seal['next_inner'],phase=seal['phase'],world=8,
        files_sha256=seal['files_sha256'],identity_sha256=sha(cp/'identity.json'),
        valid_scene_exposures=tx['ranks'][0]['valid_scene_exposures'])


def resource_return(run,node,result,latest,target,*,authorized_hosts,grant,now=None):
    if result.get('returncode')!=0 or result.get('forced_expiry'):return None
    cp=inspect_native(run,latest['checkpoint']);u=cp['updates'];host=node['host'];job=Path(node['job'])
    if not job.resolve().is_relative_to(ROOT/'jobs'/run['id'])or result.get('host')!=host:raise ValueError('Foreign native job return')
    now=time.time()if now is None else now
    due=host not in authorized_hosts or(host in grant.get('temporary_hosts',[])and now>=grant['expires_unix']-grant['checkpoint_guard_seconds'])
    if not due or not 0<=u<target:return None
    if result.get('cwd')!=run.get('execution_source',run['source']):raise ValueError('Native resource return source changed')
    for rank in range(8):
        rr=read(Path(run['root'])/f'result_rank{rank}.json',{})
        if rr.get('updates')!=u or rr.get('checkpoint')!=cp['path']or rr.get('rank')!=rank or rr.get('host')!=host:raise ValueError('Incomplete8rank cooperative return')
    return dict(job=str(job),result=result,checkpoint=cp,reason='EXACT_NATIVE_RESOURCE_BOUNDARY',
        full_tensor_hashes_and_Adam_RNG_restore='required by unchanged strict native restore',at_unix=now)


def owner_class():
    bootstrap()
    from tools.qwen_world_safety_exploration.owner import Owner,StopBeforeDispatch
    from tools.qwen_world_safety_exploration import dispatch,eval_owner
    from starVLA.rl.qwen_world_safety_exploration.resources import probe,MODEL_PYTHON,gpu_activity_issue
    from starVLA.rl.qwen_world_safety_exploration.contracts import allowed_hosts,authorization
    class PerformanceOwner(Owner):
        def __init__(self):
            super().__init__(selected_ids=ORDER,owner_name=OWNER);validate_registry(self.registry)
            self.immutable={r['id']:{k:r[k]for k in('id','root','source','source_commit','worker_source','config','config_sha256','parent_checkpoint','inner_updates','group_size','owner_state','trainer_module')}for r in self.runs}
            self.runs.sort(key=lambda r:r['priority']);self.handoff=False
            publish(self.directory/'OWNER.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),host=socket.gethostname(),source=__file__,source_sha256=sha(__file__),started_unix=time.time()))
        def current(self,arm):
            registry=validate_registry(read(ROOT/'RUNS.json'));run=next(r for r in registry['runs']if r['id']==arm)
            if any(run[k]!=v for k,v in self.immutable[arm].items()):raise ValueError('Frozen native identity/config/parent binding changed')
            return run
        def candidate(self,run):
            # Admit the two designated primary studies before lower priorities,
            # without preempting any already running learner.
            for higher in self.runs:
                if higher['priority']>=run['priority']:continue
                status=self.state['runs'].get(higher['id'],{}).get('status')
                if status is None or status in('WAITING_RESOURCE','WAITING_AUTHORIZED_RESOURCE'):
                    return None
            claims=read(ROOT/'resources/NATIVE_HOST_CLAIMS.json',{})
            hosts={h for h,c in claims.items()if'job'in c and not(Path(c['job'])/'RESULT.json').exists()}
            known=eval_owner.KNOWN_ROOTS+(str(ROOT)+'/',)
            for route in run['deployment_candidates']:
                host=route['host']
                if host in PROTECTED or host in hosts or host not in allowed_hosts(admit_long_job=True):continue
                try:
                    s=probe(host);g=s['gpus']
                    if(len(g)!=8 or len({x['uuid']for x in g})!=8 or not s['worker_python_exists']or s['MemAvailable_bytes']<100*2**30 or s['load1']>s['CPU_affinity']*1.5):continue
                    if any(gpu_activity_issue(x,s['owners'])for x in g)or min(x['free_MiB']for x in g)<route['minimum_free_MiB']:continue
                    if any(not o.get('visible')or not isinstance(o.get('output'),str)or not o['output'].startswith(known)for o in s['owners']):continue
                    if any(not eval_owner.lease_free(eval_owner.LEASES['primary']/(x['uuid']+'.lock'))for x in g):continue
                    return route,s
                except(Exception):continue
            return None
        def run_stage(self,run,target,resume):
            arm=run['id'];node=self.stage(arm,target)
            if node.get('complete'):return inspect_native(run,node['checkpoint'])['path']
            while True:
                if(self.directory/'STOP_OWNER').exists():self.handoff=True;return None
                run=self.current(arm)
                if not node.get('job')and((ROOT/'STOP_REQUESTED').exists()or run.get('disposition')):raise StopBeforeDispatch('Explicit stop/reserve prevents new work')
                if not node.get('job'):
                    exact=Path(run['root'])/'checkpoints'/f'update{target:06}_inner0'
                    if(exact/'COMPLETE').exists():
                        cp=inspect_native(run,exact)
                        with self.mutex:
                            node.update(complete=True,checkpoint=cp['path'],adopted_complete_seal=True);self.update(arm,status='STAGE_COMPLETE',updates=target,checkpoint=cp['path'])
                        return cp['path']
                    latest=read(Path(run['root'])/'latest.json')
                    if latest:
                        cp=inspect_native(run,latest['checkpoint']);resume=cp['path']
                        if cp['updates']>=target:raise ValueError('Actual native progress exceeds a missing earlier gate; explicit review required')
                    self.update(arm,status='WAITING_RESOURCE',stage_target_updates=target)
                    lockpath=ROOT/'resources/NATIVE_DISPATCH.lock';lockpath.parent.mkdir(parents=True,exist_ok=True)
                    with lockpath.open('a')as lock:
                        fcntl.flock(lock,fcntl.LOCK_EX);chosen=self.candidate(run)
                        if chosen is not None:
                            route,s=chosen;attempt=int(node.get('attempt',0))+1;name=f'{arm}/formal_to{target:06}_attempt{attempt:03}'
                            command=[MODEL_PYTHON,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=8','-m',run['trainer_module'],'--config',run['config'],'--manifest',self.registry['manifest'],'--il-checkpoint',run['parent_checkpoint'],'--data',self.registry['observation_data'],'--parent-selection',self.registry['parent_selection'],'--output',run['root'],'--stop-after-update',str(target),'--save-every',str(run['save_every'])]
                            if resume:command+=['--resume',resume]
                            with self.mutex:
                                node.update(attempt=attempt,job=str(ROOT/'jobs'/name),host=route['host'],resume=resume,command=command,status='DISPATCHING');self.update(arm,status='DISPATCHING',job=node['job'],host=route['host'])
                            claims=read(ROOT/'resources/NATIVE_HOST_CLAIMS.json',{});claims[route['host']]=dict(job=node['job'],arm=arm,at_unix=time.time());publish(ROOT/'resources/NATIVE_HOST_CLAIMS.json',claims)
                            try:dispatch.launch(route['host'],run['source'],command,name,gpus='0,1,2,3,4,5,6,7',expected_uuids=[x['uuid']for x in s['gpus']],share_minimum=route['minimum_free_MiB'],lease_slot='primary',worker_source=run.get('worker_source',run['source']))
                            except Exception:
                                with self.mutex:
                                    node['dispatch_error']=traceback.format_exc();self.update(arm,status='DISPATCH_TRANSPORT_REVIEW',job=node['job'])
                                raise
                    if chosen is None:time.sleep(30);continue
                result=dispatch.job_state(node['job']);latest=read(Path(run['root'])/'latest.json');u=latest['updates']if latest else 0
                self.update(arm,status='TRAINING'if result.get('status')=='RUNNING'else result.get('status','STARTING'),updates=u,stage_target_updates=target,job=node['job'],host=node['host'])
                if result.get('returncode')==75 and result.get('child_not_started')is True:
                    with self.mutex:
                        node.setdefault('resource_refusals',[]).append(dict(job=node.pop('job'),result=result));self.update(arm,status='WAITING_RESOURCE')
                    time.sleep(30);continue
                if'returncode'not in result:time.sleep(10);continue
                if result['returncode']!=0:raise RuntimeError('Real child failure retained, no scientific retry: '+node['job'])
                if not latest:raise ValueError('EXIT0 without a sealed native checkpoint')
                cp=inspect_native(run,latest['checkpoint'])
                if cp['updates']!=target:
                    if(ROOT/'STOP_REQUESTED').exists()or run.get('disposition'):raise StopBeforeDispatch('Explicit cooperative stop saved')
                    grant,_=authorization();returned=resource_return(run,node,result,latest,target,authorized_hosts=allowed_hosts(admit_long_job=True),grant=grant)
                    if returned is None:raise ValueError('Unexpected early native exit, preserve evidence')
                    with self.mutex:
                        node.setdefault('resource_returns',[]).append(returned);node.pop('job');resume=cp['path'];self.update(arm,status='WAITING_AUTHORIZED_RESOURCE',checkpoint=resume,updates=cp['updates'])
                    continue
                for rank in range(8):
                    rr=read(Path(run['root'])/f'result_rank{rank}.json',{})
                    if rr.get('updates')!=target or rr.get('checkpoint')!=cp['path']:raise ValueError('Missing8rank stage completion')
                with self.mutex:
                    node.update(complete=True,checkpoint=cp['path'],end_unix=time.time());self.update(arm,status='STAGE_COMPLETE',updates=target,checkpoint=cp['path'],valid_scene_exposures=cp['valid_scene_exposures'])
                return cp['path']
        def run_arm(self,initial):
            arm=initial['id'];resume=None
            try:
                while True:
                    run=self.current(arm)
                    for target in run['native_gates']:
                        resume=self.run_stage(run,target,resume)
                        if self.handoff:return
                        if(ROOT/'STOP_REQUESTED').exists():raise StopBeforeDispatch('Global cooperative stop')
                    self.update(arm,status='COMPLETED_AT_REGISTERED_CAP',updates=run['target_updates'],exposure_epochs=run['target_epochs'],review='6epoch complete; no automatic20epoch commitment');return
            except StopBeforeDispatch:self.update(arm,status='STOPPED_AT_SAVED_BOUNDARY')
            except Exception:self.update(arm,status='FAILED_REVIEW_REQUIRED',error=traceback.format_exc())
        def run(self):
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=7)as pool:list(pool.map(self.run_arm,self.runs))
            self.state['status']='CPU_OWNER_QUIESCED_CHILDREN_PRESERVED'if self.handoff else'COMPLETED_AT_REGISTERED_CAP'if all(r.get('status')=='COMPLETED_AT_REGISTERED_CAP'for r in self.state['runs'].values())else'FINISHED_WITH_REVIEW_REQUIRED'
            if not self.handoff:self.state['native_owner_finished_unix']=time.time()
            publish(self.state_path,self.state)
    return PerformanceOwner


def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('action',choices=['prepare','seal-source','check','run']);ap.add_argument('--source',default=str(SOURCE));a=ap.parse_args()
    if a.action=='prepare':r=prepare(Path(a.source))
    elif a.action=='seal-source':r=seal_source()
    elif a.action=='check':r=validate_registry(read(ROOT/'RUNS.json'),allow_pending=True)
    else:owner_class()().run();return
    print(json.dumps(dict(status=r['status'],arms=len(r['runs']),source_pending=any(x.get('source_pending')for x in r['runs']),root=str(ROOT))))
if __name__=='__main__':main()
