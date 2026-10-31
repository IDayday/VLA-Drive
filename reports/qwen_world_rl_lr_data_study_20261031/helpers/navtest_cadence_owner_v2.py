"""Independent administrative NAVTEST cadence; never trains/stops/promotes arms.

Extra points12k,14k,... up to the explicit RUNS research cap, excluding legacy
GATES owned by the original controllers. Watch COMPLETE metadata, not journals.
New evaluations use up-to32 lawful routes; existing layouts remain immutable.
"""
from __future__ import annotations
import argparse,fcntl,hashlib,json,os,select,struct,subprocess,sys,threading,time,traceback,uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import ctypes
SOURCE=Path('/mnt/project/VLA-Drive-qwen-world-rl-research-20261030')
sys.path.insert(0,str(SOURCE))
os.environ['WORLD_POSTCOT_TASK']='qwen_world_rl_optimization_20261030'
from tools.qwen_world_research import dispatch as D
from tools.qwen_world_research.campaign_owner import CampaignOwner,atomic_json,read_json,now,native_checkpoint,job_path
R=Path('/mnt/project/qwen-world-rl-optimization-20261030')
A=R/'analysis';O=R/'owners/NAVTEST_CADENCE';PLAN=A/'NAVTEST_CADENCE_V2_LAUNCH_PLAN_20261031.json'
POLICY=R/'NAVTEST_CADENCE_POLICY_20261031.json'
LEGACY_GATES=(1000,5000,10000,20000,35000,50000)
INFERENCE=dict(sampler='ODE',steps=10,shift=5.0,seed=2101,scene_seed_namespace='native_qwen_sourceflow_ODE',candidates=1)
V3=R/'runtime/v3_4e61823bb3a8'
SCHEMA='independent_NAVTEST_even2k_cadence_owner_v1'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def label_for(arm,u):return f'{arm}_u{u:06}_NAVTEST'
def targets(cap):return [u for u in range(12000,int(cap)+1,2000) if u not in LEGACY_GATES]
def allowed(row):
    status=str(row.get('status','')).upper()
    # Missing explicit approval is not inferred from a historic50k ceiling.
    return (int(row.get('research_budget_updates',0))>10000 and bool(row.get('job'))
        and bool(row.get('source')) and bool(row.get('root'))
        and not any(x in status for x in ('STOP','FAIL','CANCEL','REGISTERED','NOT_STARTED','HELD','PAUSED','DEFER')))
def eligible_rows(root=R):return {x['id']:x for x in read_json(root/'RUNS.json')['runs'] if allowed(x)}
def seal_policy():
    fields=dict(schema='NAVTEST_even2k_cadence_policy_v1',user_instruction='获准长训任务10k后每2k做完整NAVTEST，尽可能并行分片',
        starts_after_updates=10000,interval_updates=2000,requested_maximum_world=32,minimum_new_world=8,max_active_models=2,
        eligibility='explicit RUNS research_budget_updates>10000,first job dispatched,not STOP/FAIL/CANCEL/REGISTERED/HELD/PAUSED/DEFER',
        legacy_gates_owned_elsewhere=list(LEGACY_GATES),frozen_world_for_existing_specs=True,full_NAVTEST_scenes=12146,
        inference=INFERENCE,scorer_source=str(V3),TRAIN_frequency_unchanged=True,training_budget_unchanged=True,
        stopped_arms_excluded=True,unstarted_arms_excluded=True,native_training_never_waits_for_this_owner=True,
        checkpoint_requirement='native COMPLETE with exact even2k updates,next_inner0',
        lifetime='continuous event watcher; no arbitrary short deadline;1800s metadata heartbeat',
        provenance='root delegated preparation for latest user instruction; startup requires root review',
        scientific_result=False)
    if POLICY.exists():
        old=read_json(POLICY)
        if any(old.get(k)!=v for k,v in fields.items()):raise ValueError('Sealed cadence policy differs; explicit root review required')
        return old
    record=dict(fields,sealed_at=now());atomic_json(POLICY,record);return record
def refresh(status,evidence):
    atomic_json(R/'reporting/refresh_request.json',dict(id='NAVTEST_CADENCE_'+uuid.uuid4().hex,
        scientific_result=False,evidence_kind='completed_supplemental_analysis',reason='10k后每2k原生NAVTEST评测：'+status,evidence=str(evidence)))


class Events:
    """Small Linux metadata watcher, also used for exact worker capacity exits."""
    HEADER=struct.Struct('iIII')
    def __init__(self):
        self.libc=ctypes.CDLL(None,use_errno=True);self.libc.inotify_init1.restype=ctypes.c_int
        self.libc.inotify_add_watch.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_uint32]
        self.fd=self.libc.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC)
        if self.fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
        self.paths={};self.names={}
    def add(self,path,names=None):
        path=Path(path)
        if not path.is_dir():return
        if path in self.names:
            if self.names[path] is not None:
                self.names[path]=None if names is None else self.names[path]|set(names)
            return
        wd=self.libc.inotify_add_watch(self.fd,os.fsencode(path),0x8|0x80|0x100)
        if wd<0:raise OSError(ctypes.get_errno(),str(path))
        self.paths[wd]=path;self.names[path]=None if names is None else set(names)
    def drain(self):
        changed=[]
        while True:
            try:raw=os.read(self.fd,262144)
            except BlockingIOError:break
            pos=0
            while pos<len(raw):
                wd,mask,cookie,length=self.HEADER.unpack_from(raw,pos);pos+=self.HEADER.size
                name=os.fsdecode(raw[pos:pos+length].split(b'\0',1)[0]);pos+=length
                if mask&0x4000:changed.append('INOTIFY_OVERFLOW');continue
                p=self.paths.get(wd)
                if p is not None and (self.names[p] is None or name in self.names[p]):changed.append(str(p/name))
        return changed
    def wait(self,timeout,extra=()):
        # Ignored directory writes are not a timeout. In particular STATE.json
        # written by our heartbeat must not become another immediate heartbeat.
        deadline=time.monotonic()+max(0,timeout)
        while True:
            remaining=deadline-time.monotonic()
            if remaining<=0:return []
            ready,_,_=select.select([self.fd,*extra],[],[],remaining)
            if not ready:return []
            changed=self.drain() if self.fd in ready else []
            if any(fd!=self.fd for fd in ready):changed.append('EXACT_PROCESS_EXIT')
            if changed:return changed
    def close(self):os.close(self.fd)


def validated_summary(spec,u):
    p=Path(spec['output']);s=read_json(p/'SUMMARY.json',{})
    if not s:return None
    identity=read_json(p/'identity.json')
    if (s.get('status')!='COMPLETE' or s.get('cohort')!='NAVTEST' or s.get('scenes')!=12146
        or s.get('invalid_scenes')!=0 or s.get('RL_updates')!=u or s.get('inference')!=INFERENCE
        or s.get('train_feedback_allowed') is not False or s.get('physical_logs')!=44
        or identity.get('native_RL_checkpoint')!=str(Path(spec['rl_checkpoint']).resolve())
        or identity.get('inference')!=INFERENCE or identity.get('seed')!=2101
        or s.get('evaluation_identity_sha256')!=identity.get('sha256')):
        raise ValueError('Existing summary is not this complete native checkpoint/population: '+spec['label'])
    return dict(s,evidence=str(p/'SUMMARY.json'))


def validate_spec(spec,arguments):
    for k in ('label','parent_checkpoint','rl_checkpoint','cohort','manifest','data'):
        if spec.get(k)!=str(arguments[k]):raise ValueError('Semantic evaluation label collision: '+k)
    if spec.get('source')!=str(V3):raise ValueError('Cadence requires frozen original v3 scorer')
    jobs=[job_path(j) for j in spec.get('jobs',[])]
    if int(spec.get('world',0))<1 or len(jobs)!=int(spec['world']) or len(set(jobs))!=len(jobs):raise ValueError('Missing/duplicate immutable rank receipts')
    return spec


def choose_routes(routes,inventory,maximum=32,rank0_extra=True):
    """Use existing owner/UUID/headroom proof, then limit host CPU/RAM pressure."""
    candidates=[];remaining={};counters={}
    for route in routes:
        e=inventory.get(route['host'],{});cores=int(e.get('CPU_affinity',0));load=float(e.get('load1',1e9));mem=int(e.get('MemAvailable_bytes',0))
        # Actual measured headroom: at most3 spare logical CPUs and6GiB per
        # added scorer; reserve12GiB host RAM. No util==0 permission shortcut.
        count=min(max(0,int((cores-load)//3)),max(0,(mem-12*1024**3)//(6*1024**3)))
        remaining[route['host']]=count
        owners=[o for o in e.get('owners',[]) if o.get('uuid')==route['uuid']]
        training=any(o.get('entry')!='tools.qwen_world_research.evaluate' for o in owners)
        free=next((g['free_MiB'] for g in e.get('gpus',[]) if g['uuid']==route['uuid']),0)
        candidates.append(((bool(owners),training,load/max(cores,1),-free),route,free))
    result=[]
    for _,route,free in sorted(candidates,key=lambda x:x[0]):
        host=route['host']
        if counters.get(host,0)>=remaining[host]:continue
        required=32768 if rank0_extra and not result else 20480
        if free<required:continue
        result.append(dict(route,minimum_free=required));counters[host]=counters.get(host,0)+1
        if len(result)==maximum:break
    return result


def start_claimed_worker(root,claims,label,rank,job,route,command):
    receipt=D.start_worker(root,job,host=route['host'],source=V3,command=command,gpus=[route['gpu']],uuids=[route['uuid']],lease='evaluation',minimum_free=route['minimum_free'])
    claims[route['uuid']]=dict(receipt=receipt,label=label,rank=rank,claimed_unix=time.time())
    # Publish each successful reservation while still under the global lock.
    # A later launch failure cannot expose already dispatched sibling GPUs.
    D.write(Path(root)/'resources/EVALUATION_CLAIMS.json',claims)
    return receipt


class NoLongerAuthorized(RuntimeError):pass


class CadenceDispatch:
    """Narrow32-route adapter; all admission/claims/worker/scorer helpers unchanged."""
    def __init__(self,owner):self.owner=owner
    def __getattr__(self,name):return getattr(D,name)
    def dispatch_evaluation(self,**arguments):
        root=Path(arguments['root']);label=arguments['label'];sp=root/'evaluation_specs'/f'{label}.json'
        # This is the same per-label cross-owner lock used by dispatch.py.
        with D.locked(sp.with_suffix('.dispatch.lock')):
            if sp.exists():
                spec=validate_spec(read_json(sp),arguments)
                refused=[]
                for job in spec['jobs']:
                    rr=read_json(Path(job_path(job))/'RESULT.json',{})
                    if rr.get('returncode')==75 and rr.get('child_not_started') is True:refused.append(job)
                if not refused:return spec  # Existing16/32 layout, exact adoption.
                self.owner.require_allowed(label)
                # Existing helper only replaces proven pre-child75 receipts;
                # it preserves good siblings and the already frozen world.
                return D._dispatch_evaluation(**arguments)
            family=root/'jobs'/('EVAL_'+label)
            if family.exists() and any(family.glob('rank*_attempt*')):
                raise ValueError('Unpublished prior dispatch receipts exist; preserve and review, never redispatch: '+label)
            self.owner.require_allowed(label)
            if D.operator_source(root)!=V3:raise ValueError('Frozen evaluation source pointer changed')
            output=D.TASK_STORAGE/'evaluations'/label;export=D.TASK_STORAGE/'exports'/label
            logroot=root/'exports'/label;logroot.mkdir(parents=True,exist_ok=True)
            # Only our two active points serialize CPU export. No global owner
            # handoff and no waiting native training/optimizer boundaries.
            with D.locked(O/'CPU_EXPORT.lock'):
                if (export/'COMPLETE').exists():
                    from tools.qwen_world_research.evaluate import verify_export_binding
                    seal=read_json(export/'COMPLETE');meta=export/'export.json'
                    if sha(meta)!=seal['export_sha256']:raise ValueError('Published export metadata changed')
                    verify_export_binding(read_json(meta),parent_checkpoint=arguments['parent_checkpoint'],rl_checkpoint=arguments['rl_checkpoint'])
                else:
                    cmd=[D.MODEL,'-m','tools.qwen_world_research.evaluate','--export-only','--parent-checkpoint',str(arguments['parent_checkpoint']),
                         '--rl-checkpoint',str(arguments['rl_checkpoint']),'--export',str(export),'--output',str(output)]
                    env=dict(os.environ,PYTHONPATH=str(V3),WORLD_POSTCOT_TASK=D.TASK,CUBLAS_WORKSPACE_CONFIG=':4096:8',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
                    with (logroot/'export.log').open('ab',buffering=0) as log:subprocess.run(cmd,cwd=V3,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            self.owner.require_allowed(label)
            with D.locked(root/'resources/EVALUATION_DISPATCH.lock'):
                raw,claims=D._eval_routes(root,maximum=128)
                inventory=read_json(root/'resources/LATEST_EVALUATION_INVENTORY.json')['hosts']
                routes=choose_routes(raw,inventory)
                if len(routes)<8:raise BlockingIOError(f'Cadence needs at least8 lawful GPU/CPU/RAM routes; {len(routes)} available, requested up-to32')
                jobs=[]
                for rank,route in enumerate(routes):
                    job=root/'jobs'/('EVAL_'+label)/f'rank{rank:03}_attempt001'
                    command=[D.MODEL,'-u','-m','tools.qwen_world_research.evaluate','--parent-checkpoint',str(arguments['parent_checkpoint']),
                        '--rl-checkpoint',str(arguments['rl_checkpoint']),'--export',str(export),'--data',str(arguments['data']),
                        '--manifest',str(arguments['manifest']),'--cohort','NAVTEST','--output',str(output),'--rank',str(rank),'--world',str(len(routes)),'--seed','2101']
                    if rank==0:command+=['--native-parity-output',str(output/'native_export_parity.json')]
                    jobs.append(start_claimed_worker(root,claims,label,rank,job,route,command))
                spec=dict(label=label,parent_checkpoint=str(arguments['parent_checkpoint']),rl_checkpoint=str(arguments['rl_checkpoint']),cohort='NAVTEST',
                    manifest=str(arguments['manifest']),data=str(arguments['data']),baseline=None,output=str(output),export=str(export),jobs=jobs,
                    world=len(routes),source=str(V3),training_bank=None,requested_world=32,administrative_owner=str(O),
                    route_policy='existing UUID/visible-owner/admission proof, plus measured CPU/RAM pressure')
                D.write(sp,spec);return spec


class CadenceOwner(CampaignOwner):
    def __init__(self):
        # Deliberately no CampaignOwner.__init__, CAMPAIGN lock, RUNS write, or
        # per-arm STATE mirror: original owners retain all training state.
        if read_json(PLAN)['bindings']!=bindings():raise ValueError('Reviewed cadence source/policy binding changed')
        O.mkdir(parents=True,exist_ok=True);self.lock=(O/'owner.lock').open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.root=R;self.owner_root=O;self.mutex=threading.RLock();self.capacity_changed=threading.Condition(self.mutex);self.capacity_epoch=0
        self.parent=str(R/'IL_PARENT');self.navtest_manifest=str(R/'data/panels/NAVTEST_012146.json');self.navtest_data='/mnt/project/qwen-sourceflow-native-20261018/data/NAVTEST'
        self.dispatcher=CadenceDispatch(self);self.resource_retries=0
        self.state=read_json(O/'STATE.json',{})
        if self.state and self.state.get('schema')!=SCHEMA:raise ValueError('Other cadence owner STATE')
        self.state.setdefault('schema',SCHEMA);self.state.setdefault('evaluations',{});self.state.setdefault('events',[]);self.state.setdefault('results',[])
        self.state.update(status='WATCHING',policy=str(POLICY),policy_sha256=sha(POLICY),requested_world=32,max_active_models=2,training_budget_mutation=False,legacy_gates_excluded=list(LEGACY_GATES))
        self.events=Events();self.pool=ThreadPoolExecutor(max_workers=2,thread_name_prefix='cadence_eval');self.futures={}
        atomic_json(O/'OWNER.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),source_file=str(Path(__file__).resolve()),
            source_sha256=sha(__file__),started_at=now(),role='CPU NAVTEST observer/dispatcher only; no native training actions'))
        self.flush()
    def flush(self):
        with self.mutex:
            self.state['updated_at']=now();atomic_json(O/'STATE.json',self.state)
            atomic_json(O/'RESULTS.json',dict(schema='NAVTEST_cadence_complete_result_index_v1',status=self.state['status'],updated_at=self.state['updated_at'],
                evaluations={k:dict(arm=v['arm'],updates=v['updates'],checkpoint=v['checkpoint'],summary=v['summary'],spec=v.get('spec'))
                    for k,v in self.state['evaluations'].items() if v.get('status')=='COMPLETE'}))
    def require_allowed(self,label):
        n=self.state['evaluations'][label];row=eligible_rows().get(n['arm'])
        if not row or n['updates'] not in targets(row['research_budget_updates']):raise NoLongerAuthorized('No longer approved for new cadence evaluation: '+label)
        if str(Path(row['root']).resolve())!=n['run_root'] or row['source']!=n['training_source']:raise ValueError('Registered logical run/source changed while queued')
    def wait_capacity(self,node,*,status,label,arm=None,since=None):
        if since is not None and since!=self.capacity_epoch:return
        node['status']=status;self.event('waiting_real_capacity',label=label,status=status)
        ev=Events();fds=[]
        try:
            ev.add(R,{'RUNS.json'});ev.add(R/'resources',{'EVALUATION_CLAIMS.json'});ev.add(O,{'STOP_OWNER'});ev.add(Path('/mnt/project'),{'server_dispatch_policy.json'})
            jobs={r['job'] for r in read_json(R/'RUNS.json')['runs'] if r.get('job')}
            jobs.update(x['receipt'] for x in read_json(R/'resources/EVALUATION_CLAIMS.json',{}).values() if x.get('receipt'))
            for job in jobs:
                p=Path(job);ev.add(p,{'RESULT.json'})
                if (p/'RESULT.json').exists():continue
                launch=read_json(p/'LAUNCH.json',{})
                try:
                    pid=int(launch['transport_pid']);st=Path(f'/proc/{pid}/stat').read_text().split()
                    if int(st[21])!=launch['transport_start_ticks'] or st[2] in ('Z','X'):continue
                    fd=os.pidfd_open(pid) if hasattr(os,'pidfd_open') else ctypes.CDLL(None,use_errno=True).syscall(434,pid,0)
                    if fd>=0:fds.append(fd)
                except (OSError,KeyError,ValueError):pass
            if ev.wait(1800,fds):self.capacity_released()
            else:self.event('capacity_1800s_heartbeat',label=label)
        finally:
            ev.close()
            for fd in fds:os.close(fd)
        if (O/'STOP_OWNER').exists():raise NoLongerAuthorized('Administrative cadence stop; preserve existing eval children')
        self.require_allowed(label)
    def evaluate(self,label):
        node=self.state['evaluations'][label];u=node['updates']
        try:
            arguments=dict(label=label,parent_checkpoint=self.parent,rl_checkpoint=node['checkpoint'],cohort='NAVTEST',manifest=self.navtest_manifest,data=self.navtest_data)
            sp=R/'evaluation_specs'/f'{label}.json'
            if sp.exists():
                spec=validate_spec(read_json(sp),arguments);node['spec']=spec;self.flush()
                summary=validated_summary(spec,u)
                if summary:
                    node.update(status='COMPLETE',summary=summary,completed_at=now(),adopted_complete=True);self.flush();refresh('ADOPTED_COMPLETE',summary['evidence']);return
            elif (D.TASK_STORAGE/'evaluations'/label/'SUMMARY.json').exists():
                # Durable complete output may outlive its administrative spec.
                # Adopt only after exact identity/CP/inference validation.
                output=D.TASK_STORAGE/'evaluations'/label
                spec=dict(arguments,source=str(V3),output=str(output),world=read_json(output/'layout.json')['world'],jobs=[])
                summary=validated_summary(spec,u)
                node.update(status='COMPLETE',spec=spec,summary=summary,completed_at=now(),adopted_complete_without_dispatch_spec=True)
                self.flush();refresh('ADOPTED_COMPLETE',summary['evidence']);return
            node['status']='EVALUATING';self.flush()
            # Shared helper handles exact child waits, retained75 attempts,
            # event-based capacity wait and complete12146 validation.
            summary=CampaignOwner.run_evaluation(self,label,node['checkpoint'],cohort='NAVTEST',node=node)
            summary=validated_summary(node['spec'],u)
            if summary is None:raise ValueError('Expected durable canonical summary missing')
            node.update(status='COMPLETE',summary=summary,completed_at=now());self.flush();refresh('COMPLETE',summary['evidence'])
        except NoLongerAuthorized as exc:
            node.update(status='HELD_NO_NEW_EVAL_AUTHORIZATION',reason=str(exc));self.flush()
        except Exception:
            node.update(status='FAILED_REQUIRES_REVIEW',error=traceback.format_exc(),failed_at=now());self.flush();refresh('FAILED_REQUIRES_REVIEW',O/'STATE.json')
    def discover(self):
        rows=eligible_rows();self.state['approved_arms']={a:dict(eligible_budget=int(r['research_budget_updates']),currently_dispatched_target=r.get('stage_budget_updates'),run_root=r['root'],source=r['source'],status=r.get('status')) for a,r in rows.items()}
        self.events.add(R,{'RUNS.json'});self.events.add(O,{'STOP_OWNER','WORK_COMPLETED'});self.events.add(R/'evaluation_specs')
        for arm,row in rows.items():
            root=Path(row['root']);ts=targets(row['research_budget_updates']);names={f'update{u:06}_inner0' for u in ts}
            self.events.add(root.parent,{root.name});self.events.add(root,{'checkpoints'});self.events.add(root/'checkpoints',names)
            for u in ts:
                cp=root/'checkpoints'/f'update{u:06}_inner0';self.events.add(cp,{'COMPLETE'});label=label_for(arm,u)
                if label in self.state['evaluations']:continue
                try:checkpoint=native_checkpoint(root,u)
                except FileNotFoundError:continue
                if checkpoint['updates']!=u:raise ValueError('Cadence native boundary mismatch')
                identity=root/'identity.json'
                self.state['evaluations'][label]=dict(arm=arm,updates=u,checkpoint=checkpoint['path'],checkpoint_complete_sha256=sha(cp/'COMPLETE'),
                    run_root=str(root.resolve()),training_source=row['source'],native_identity_sha256=sha(identity),status='READY',discovered_at=now())
        self.flush()
    def run_loop(self):
        self.event('cadence_started',policy='native12k/even2k; no legacy GATES; dynamic explicit approved budgets; no training actions')
        try:
            while not (O/'STOP_OWNER').exists():
                self.discover()
                for label,f in list(self.futures.items()):
                    if f.done():f.result();del self.futures[label]
                available=2-len(self.futures)
                nodes=sorted(self.state['evaluations'].items(),key=lambda x:(x[1]['updates'],x[0]))
                for label,n in nodes:
                    if available<=0:break
                    if label in self.futures or n['status'] in ('COMPLETE','FAILED_REQUIRES_REVIEW','HELD_NO_NEW_EVAL_AUTHORIZATION'):continue
                    try:self.require_allowed(label)
                    except NoLongerAuthorized:n['status']='HELD_NO_NEW_EVAL_AUTHORIZATION';continue
                    self.futures[label]=self.pool.submit(self.evaluate,label);available-=1
                self.flush()
                # Completion callback emits one administrative wake, no logs or
                # minute-level full-state polling. RUNS promotions also wake us.
                for f in self.futures.values():
                    if not getattr(f,'_cadence_callback',False):
                        f.add_done_callback(lambda _:atomic_json(O/'WORK_COMPLETED',dict(at=now(),id=uuid.uuid4().hex)));f._cadence_callback=True
                if not self.events.wait(1800):self.event('cadence_1800s_heartbeat')
            self.state['status']='STOPPED_CPU_OBSERVER_ONLY';self.flush()
        finally:
            self.pool.shutdown(wait=False);self.events.close()


def bindings():
    return {str(p):sha(p) for p in (Path(__file__).resolve(),POLICY,SOURCE/'tools/qwen_world_research/campaign_owner.py',SOURCE/'tools/qwen_world_research/dispatch.py',V3/'tools/qwen_world_research/evaluate.py')}
def prepare():
    seal_policy()
    rows=eligible_rows();ready=[]
    for a,r in rows.items():
        for u in targets(r['research_budget_updates']):
            try:cp=native_checkpoint(r['root'],u)
            except FileNotFoundError:continue
            ready.append(dict(arm=a,updates=u,checkpoint=cp['path'],existing_spec=(R/'evaluation_specs'/f'{label_for(a,u)}.json').exists()))
    value=dict(schema='NAVTEST_cadence_launch_plan_v1',status='READY_FOR_ROOT_REVIEW_NOT_STARTED',prepared_at=now(),started=False,
        script=str(Path(__file__).resolve()),policy=str(POLICY),policy_sha256=sha(POLICY),bindings=bindings(),model_python=D.MODEL,
        launch_argv=[D.MODEL,str(Path(__file__).resolve()),'--launch'],owner=str(O/'OWNER.json'),state=str(O/'STATE.json'),results=str(O/'RESULTS.json'),launch=str(O/'LAUNCH.json'),
        requested_world=32,minimum_routes=8,max_active_models=2,cpu_export='one at a time under independent CPU_EXPORT.lock',
        cpu_route_policy=dict(spare_logical_cpus_per_rank=3,available_RAM_per_rank_GiB=6,host_RAM_reserve_GiB=12,prioritize='no GPU owner; then measured load/free memory; original owner accounting still required'),
        legacy_gates_excluded=list(LEGACY_GATES),cadence='12000,14000,... explicit RUNS research cap; COMPLETE inner0 only',
        approved_arms={a:r['research_budget_updates'] for a,r in rows.items()},currently_sealed_extra_points=ready,
        protocol=dict(cohort='NAVTEST',scenes=12146,inference=INFERENCE,scorer_source=str(V3)),
        preserves=['training jobs/optimizer/checkpoints','main FIRST14 owner lock/state/source','all TRAIN frequencies','old eval spec world/layout/receipts','shared exact GPU claims'],
        never=['train or raise budgets','revive stopped/unstarted arms','kill native/eval children','use NAVTEST tokens for train selection','repartition already dispatched evaluations'])
    atomic_json(PLAN,value);return value

def launch():
    plan=read_json(PLAN)
    if plan['bindings']!=bindings():raise ValueError('Root-reviewed source bindings changed; prepare/review again')
    if str(Path(sys.executable).resolve())!=str(Path(D.MODEL).resolve()):raise ValueError('Launch with the model venv, not system Python')
    O.mkdir(parents=True,exist_ok=True)
    with (O/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for old in (read_json(O/'OWNER.json',{}),read_json(O/'LAUNCH.json',{})):
            if not old:continue
            try:
                st=Path(f"/proc/{old['pid']}/stat").read_text().split()
                if int(st[21])==old['start_ticks'] and st[2] not in ('Z','X'):raise RuntimeError('Existing exact cadence owner')
            except FileNotFoundError:pass
        cmd=[D.MODEL,'-u',str(Path(__file__).resolve()),'--run']
        with (O/'stdout.log').open('ab',buffering=0) as log:
            proc=subprocess.Popen(cmd,cwd=A,env=dict(os.environ,WORLD_POSTCOT_TASK=D.TASK,PYTHONPATH=str(SOURCE),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record=dict(pid=proc.pid,start_ticks=int(Path(f'/proc/{proc.pid}/stat').read_text().split()[21]),argv=cmd,source_sha256=sha(__file__),started_at=now(),plan=str(PLAN),bindings=plan['bindings'])
        atomic_json(O/'LAUNCH.json',record);plan.update(status='ROOT_STARTED',started=True,actual_launch=record);atomic_json(PLAN,plan);print(json.dumps(record))

def self_check():
    import tempfile
    from unittest.mock import patch
    assert targets(10000)==[] and targets(20000)==[12000,14000,16000,18000]
    assert targets(50000)[-1]==48000 and not set(targets(50000))&set(LEGACY_GATES)
    base=dict(id='MOCK',research_budget_updates=20000,job='job',root='run',source='source',status='TRAINING_DISPATCHED')
    assert allowed(base) and not allowed(dict(base,research_budget_updates=5000))
    for st in ('STOPPED_FUTILITY','REGISTERED','FAILED_REQUIRES_REVIEW','HELD_BY_STOP_REQUEST'):assert not allowed(dict(base,status=st))
    # Every scene belongs to exactly one rank for both immutable16 and new32.
    for world in (16,32):
        shards=[list(range(12146))[rank::world] for rank in range(world)]
        assert sum(map(len,shards))==12146 and sorted(x for q in shards for x in q)==list(range(12146))
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);p=root/'evaluation_specs';p.mkdir();args=dict(root=root,label='MOCK_u012000_NAVTEST',parent_checkpoint='parent',rl_checkpoint='cp',cohort='NAVTEST',manifest='manifest',data='data')
        spec=dict(**{k:str(v) for k,v in args.items() if k!='root'},source=str(V3),world=16,jobs=[str(root/f'job{rank}') for rank in range(16)],output=str(root/'out'))
        atomic_json(p/(args['label']+'.json'),spec)
        class Owner:
            def require_allowed(self,label):raise AssertionError('Existing live spec should not redispatch')
        with patch.object(D,'_dispatch_evaluation',side_effect=AssertionError('Duplicate dispatch')):
            result=CadenceDispatch(Owner()).dispatch_evaluation(**args)
            assert result==spec and result['world']==16
        bad=dict(spec,jobs=spec['jobs'][:-1]+[spec['jobs'][0]])
        try:validate_spec(bad,args)
        except ValueError:pass
        else:raise AssertionError('Duplicate rank receipts not rejected')
        route=dict(host='MOCK_HOST',gpu=0,uuid='MOCK_UUID_0',minimum_free=32768);claims={}
        with patch.object(D,'start_worker',side_effect=[str(root/'accepted'),OSError('mock launch refusal')]):
            start_claimed_worker(root,claims,args['label'],0,root/'accepted',route,[])
            try:start_claimed_worker(root,claims,args['label'],1,root/'refused',dict(route,uuid='MOCK_UUID_1'),[])
            except OSError:pass
            else:raise AssertionError('Mock failure absent')
        assert set(read_json(root/'resources/EVALUATION_CLAIMS.json'))=={'MOCK_UUID_0'}
        wrong=dict(args,rl_checkpoint='wrong')
        try:CadenceDispatch(Owner()).dispatch_evaluation(**wrong)
        except ValueError:pass
        else:raise AssertionError('Checkpoint collision not rejected')
    result=dict(status='PASS',checked_at=now(),scope='CPU-only target/budget/stopped guard,16/32 complete partitions,existing16 spec adoption/unique rank count/checkpoint collision,partial-launch accepted sibling claim preservation; no real dispatch/model/probe',script_sha256=sha(__file__))
    atomic_json(A/'NAVTEST_CADENCE_CPU_CHECK_20261031.json',result);return result

def self_wake_check():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp);events=Events();events.add(p,{'MATCH.json'})
        def writes(match):
            for i in range(6):
                (p/'STATE.json').write_text(str(i));time.sleep(.01)
            if match:(p/'MATCH.json').write_text('ready')
        try:
            worker=threading.Thread(target=writes,args=(False,));worker.start();start=time.monotonic()
            got=events.wait(.18);elapsed=time.monotonic()-start;worker.join()
            assert got==[] and elapsed>=.16,(got,elapsed)
            worker=threading.Thread(target=writes,args=(True,));worker.start();start=time.monotonic()
            got=events.wait(1);matched_elapsed=time.monotonic()-start;worker.join()
            assert set(got)=={str(p/'MATCH.json')} and .04<=matched_elapsed<1,(got,matched_elapsed)
            # Simultaneous ignored inotify and an exact worker-exit fd must
            # still wake capacity wait immediately instead of losing the exit.
            readfd,writefd=os.pipe()
            try:
                (p/'STATE.json').write_text('ignored');os.write(writefd,b'x')
                got=events.wait(.5,[readfd]);assert 'EXACT_PROCESS_EXIT' in got,got
            finally:os.close(readfd);os.close(writefd)
        finally:events.close()
    result=dict(status='PASS',checked_at=now(),script_sha256=sha(__file__),scope='CPU inotify ignored self-writes preserve actual monotonic timeout; later matching event wakes; simultaneous extra fd preserved',
        ignored_timeout_seconds=elapsed,matched_event_seconds=matched_elapsed,signals_sent=False,owners_started=False)
    atomic_json(A/'NAVTEST_CADENCE_V2_SELFWAKE_CHECK_20261031.json',result);return result


def main():
    p=argparse.ArgumentParser(__doc__);g=p.add_mutually_exclusive_group(required=True)
    for arg in ('prepare','self-check','self-wake-check','launch','run'):g.add_argument('--'+arg,action='store_true')
    a=p.parse_args()
    if a.prepare:print(json.dumps(prepare(),indent=2))
    elif a.self_check:print(json.dumps(self_check()))
    elif a.self_wake_check:print(json.dumps(self_wake_check()))
    elif a.launch:launch()
    else:CadenceOwner().run_loop()
if __name__=='__main__':main()
