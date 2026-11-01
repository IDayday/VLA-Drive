"""Independent revival CPU campaign; F control is read-only external evidence.

Qualified native admission/dispatch/receipts and async eval/review are reused.
No score-based pruning before10k; fixed50k scheduler never follows runtime caps.
run is an explicit root-reviewed launch action; check/self-check never dispatch.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import traceback

ROOT_PATH=Path('/mnt/project/qwen-world-safety-exploration-20261031/revival_training_20261101')
CONTROL_ROOT=Path('/mnt/project/qwen-world-safety-exploration-20261031/standard_training_20261101')
PROTECTED={'training-vlawm-tsinghua-zt','training-vla-zt'}
ROLES={'REVIVAL_G16':'g16','REVIVAL_DENSE_G8':'dense','REVIVAL_SATURATED_G8':'saturated','REVIVAL_ACTIVE_DATA_G8':'active_data'}
NEXT={10000:20000,20000:30000,30000:50000}
IMMUTABLE=('id','root','source','config','config_sha256','parent_checkpoint','group_size','trainer_module','revival_role','owner_state')

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def gates(cap):
    if cap not in (10000,20000,30000,50000):raise ValueError('Only reviewed native caps10/20/30/50k')
    return [2000,5000,10000]+list(range(12000,cap+1,2000))
def external_control(r):
    expected=dict(registry=str(CONTROL_ROOT/'RUNS.json'),arm='SAFETY_G8_CONTROL',root=str(CONTROL_ROOT),read_only=True)
    if r.get('external_control',expected)!=expected:raise ValueError('Exact read-only F external control required')
    if 'shared_control_file'in r and Path(r['shared_control_file']).resolve()!=CONTROL_ROOT/'RUNS.json':raise ValueError('Conflicting shared control registry')
    return expected

def validate_registry(r,root):
    if Path(root).resolve()!=ROOT_PATH or Path(r['root']).resolve()!=ROOT_PATH or r['task']!='qwen_world_safety_exploration_20261031':raise ValueError('Exact separate revival G task required')
    if r.get('continuation_max_updates',20000)not in (20000,30000,50000):raise ValueError('Explicit bounded continuation maximum required')
    if r.get('training_population')!=103288 or not 3<=len(r['runs'])<=4 or not {x['id']for x in r['runs']}<=set(ROLES) or len({x['id']for x in r['runs']})!=len(r['runs']):raise ValueError('Only three/four registered fresh revival full103288 arms')
    for run in r['runs']:
        if (run['revival_role']!=ROLES[run['id']] or run['group_size']!=(16 if run['id']=='REVIVAL_G16' else 8) or run['trainer_module']!='tools.qwen_world_safety_exploration.train'
                or Path(run['owner_state']).resolve()!=ROOT_PATH/'owners/REVIVAL_NATIVE/STATE.json'
                or run['target_updates']>r.get('continuation_max_updates',20000) or run['gates']!=gates(run['target_updates']) or not Path(run['root']).resolve().is_relative_to(ROOT_PATH/'runs')):
            raise ValueError('Wrong standard role/owner/cadence/approved cap')
        if sha(run['config'])!=run['config_sha256']:raise ValueError('Frozen scientific configuration changed')
        if any(x['host']in PROTECTED for x in run['deployment_candidates']):raise ValueError('Protected IL host cannot be a revival route')
        import yaml
        cfg=yaml.safe_load(Path(run['config']).read_text());research=cfg['research']
        if research.get('standard_role')!=run['revival_role'] or Path(research.get('campaign_root','')).resolve()!=ROOT_PATH:raise ValueError('Configuration role/campaign differs from registered G arm')
        if research.get('revival_training')!='world_legacy_algorithms_fixed50k_revival_v1' or research.get('standardized_training')!=dict(schema='world_fixed50k_warmup1000_cosine_floor01_v1',horizon_updates=50000,warmup_updates=1000,min_lr_ratio=.1):raise ValueError('Only reviewed fresh fixed50k revival science')
    external_control(r)
    external=read(CONTROL_ROOT/'RUNS.json');control=next(x for x in external['runs']if x['id']=='SAFETY_G8_CONTROL')
    if external['manifest']!=r['manifest'] or external['observation_data']!=r['observation_data'] or any(x['parent_checkpoint']!=control['parent_checkpoint']for x in r['runs']):raise ValueError('External control must use the same data and pinned IL70 parent')
    return r

def decision(cap,role,current=None,previous=None,trend=None):
    """Conservative policy in PDMS display points; incomplete evidence != futility."""
    if cap==50000:return dict(action='COMPLETE_APPROVED_MAXIMUM',next_cap=None)
    if cap not in NEXT:raise ValueError('Wrong main review gate')
    if cap==10000:return dict(action='CONTINUE_MINIMUM_LONG_RUN',next_cap=20000,reason='Healthy10k is not enough to reject a slow method; fixedcosine only18.4% into decay')
    if role=='w3_control':return dict(action='CONTINUE_MATCHED_CONTROL',next_cap=NEXT[cap])
    if not current or not previous or not trend:return dict(action='CONTINUE_INSUFFICIENT_FUTILITY_EVIDENCE',next_cap=NEXT[cap],reason='Missing or uncertain paired evidence is not a failure')
    strong=lambda x:x['PDMS_delta'] < -1.0 and x['PDMS_CI95'][1] < -.5
    no_progress=trend['PDMS_delta']<=0 and trend['PDMS_CI95'][1]<=0
    safety_help=any(v['repaired_failure']-v['new_failure']>=25 for v in current.get('safety_transitions',{}).values())
    high_help=current.get('fixed_IL70_high_advantage_PDMS',0)>=.1
    zero_help=current['repaired_zero']>current['new_zero']
    if strong(current) and strong(previous) and no_progress and not(safety_help or high_help or zero_help):
        return dict(action='RESEARCH_RESERVE_REVIEW',next_cap=None,reason='Two long gates >1PDMS behind control with upperCI<-.5, own nonimprovement upperCI<=0, no safety/high/zero benefit; retain resumableCP, not algorithm rejection')
    return dict(action='CONTINUE_LEARNING_OR_UNCERTAIN',next_cap=NEXT[cap],reason='Slow/uncertain progress or useful safety/high/zero trend remains; no max2/.02 screening')

def catastrophic(point):
    if not point:return False
    return (point['PDMS_delta'] < -5 and point['PDMS_CI95'][1] < -3
            and any(v['new_failure']-v['repaired_failure']>=243 for v in point.get('safety_transitions',{}).values()))

def alive(receipt):
    if receipt.get('host') not in (None,socket.gethostname()):raise RuntimeError('Foreign-host CPU receipt requires exact operator adoption; never inspect remote PID locally')
    try:
        fields=Path('/proc',str(receipt['pid']),'stat').read_text().split()
        return int(fields[21])==receipt['start_ticks'] and fields[2]not in ('Z','X')
    except FileNotFoundError:return False

def configure(root):
    if Path(root).resolve()!=ROOT_PATH:raise ValueError('Only explicit revival G root')
    os.environ.update(WORLD_COMPARE_ROOT=str(ROOT_PATH),WORLD_COMPARE_TASK='qwen_world_safety_exploration_20261031',CUDA_VISIBLE_DEVICES='')


def owner_class():
    from .owner import Owner,StopBeforeDispatch
    from .dispatch import launch,job_state
    from starVLA.rl.qwen_world_safety_exploration.resources import MODEL_PYTHON
    from starVLA.rl.flow_grpo.transactions import atomic_json
    class RevivalOwner(Owner):
        def __init__(self):
            super().__init__(selected_ids=[r['id']for r in read(ROOT_PATH/'RUNS.json')['runs']],owner_name='REVIVAL_NATIVE')
            validate_registry(self.registry,ROOT_PATH)
            self.bindings={r['id']:{k:r[k]for k in IMMUTABLE}for r in self.registry['runs']}
            self.campaign_done=threading.Event()
            atomic_json(self.directory/'OWNER.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),host=socket.gethostname(),source=__file__,source_sha256=sha(__file__),started_unix=time.time()))
        def current(self,arm):
            registry=validate_registry(read(ROOT_PATH/'RUNS.json'),ROOT_PATH)
            run=next(r for r in registry['runs']if r['id']==arm)
            if arm not in self.bindings:self.bindings[arm]={k:run[k]for k in IMMUTABLE}
            if any(run[k]!=v for k,v in self.bindings[arm].items()):raise ValueError('Frozen source/config/parent/owner binding changed')
            return run
        def candidate(self,run):
            # Reuse qualified UUID/headroom/owner/lease admission, excluding protected IL hosts.
            safe=dict(run,deployment_candidates=[r for r in run['deployment_candidates']if r['host']not in PROTECTED])
            return super().candidate(safe)
        def run_stage(self,run,target,resume):
            arm=run['id'];node=self.stage(arm,target)
            if node.get('complete'):return node['checkpoint']
            while True:
                run=self.current(arm)
                if self.campaign_done.is_set() and not node.get('job'):raise StopBeforeDispatch('CPU campaign ending before dispatch')
                if not node.get('job') and ((ROOT_PATH/'STOP_REQUESTED').exists() or run.get('revival_disposition') in ('RESEARCH_RESERVE_REVIEW','CATASTROPHIC_REVIEW')):raise StopBeforeDispatch('No new native stage after explicit stop/reserve')
                if not node.get('job'):
                    # Adopt an explicitly registered first invocation, never guess a live learner.
                    adopted=run.get('initial_job',run.get('job'))
                    if adopted:
                        dispatch=read(Path(adopted)/'DISPATCH.json');command=dispatch['command']
                        if '--stop-after-update' in command and int(command[command.index('--stop-after-update')+1])==target:
                            if command[command.index('--output')+1]!=run['root'] or command[command.index('--config')+1]!=run['config']:raise ValueError('Foreign registered job')
                            node.update(job=str(adopted),host=dispatch['host'],adopted=True)
                    if not node.get('job'):
                        self.update(arm,status='WAITING_RESOURCE',stage_target_updates=target)
                        chosen=self.candidate(run)
                        if chosen is None:time.sleep(30);continue
                        route,sample=chosen;attempt=int(node.get('attempt',0))+1;name=f'{arm}/formal_to{target:06}_attempt{attempt:03}'
                        latest=Path(run['root'])/'latest.json'
                        if latest.exists():
                            saved=read(latest)
                            resume=saved['checkpoint']
                            if saved['updates']>=target:raise ValueError('Unadopted native progress already at/past target; preserve all jobs for explicit review')
                        command=[MODEL_PYTHON,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=8','-m',run['trainer_module'],
                            '--config',run['config'],'--manifest',self.registry['manifest'],'--il-checkpoint',run['parent_checkpoint'],'--data',self.registry['observation_data'],
                            '--parent-selection',self.registry['parent_selection'],'--output',run['root'],'--stop-after-update',str(target),'--save-every','500']
                        if resume:
                            command+=['--resume',resume]
                            if run.get('execution_migration_receipt'):command+=['--migration-receipt',run['execution_migration_receipt']]
                        try:job=launch(route['host'],run.get('execution_source',run['source']),command,name,gpus='0,1,2,3,4,5,6,7',expected_uuids=[g['uuid']for g in sample['gpus']],share_minimum=route['minimum_free_MiB'],lease_slot=route['lease_slot'])
                        except Exception:
                            node.update(attempt=attempt,job=str(ROOT_PATH/'jobs'/name),dispatch_error=traceback.format_exc());self.update(arm,status='DISPATCH_TRANSPORT_REVIEW',job=node['job']);raise
                        node.update(attempt=attempt,job=str(job),host=route['host'],resume=resume,start_unix=time.time())
                    self.update(arm,status='STARTING',job=node['job'],host=node.get('host'))
                receipt=job_state(node['job']);latest_path=Path(run['root'])/'latest.json';updates=read(latest_path)['updates']if latest_path.exists()else 0
                self.update(arm,status='TRAINING'if receipt.get('status')=='RUNNING'else receipt.get('status','RUNNING'),updates=updates,stage_target_updates=target,job=node['job'],host=node.get('host'))
                if receipt.get('status')=='REFUSED_RESOURCE':
                    if receipt.get('returncode')!=75 or receipt.get('child_not_started')is not True:raise RuntimeError('Ambiguous resource refusal, no scientific retry')
                    node.setdefault('refused_jobs',[]).append(node.pop('job'));self.update(arm,status='WAITING_RESOURCE_CAPACITY');time.sleep(30);continue
                if 'returncode'not in receipt:time.sleep(10);continue
                if receipt['returncode']!=0:raise RuntimeError('Actual native failure retained, no automatic scientific retry: '+node['job'])
                latest=read(latest_path);checkpoint=Path(latest['checkpoint']);seal=read(checkpoint/'COMPLETE')
                if latest['updates']!=target:
                    if (ROOT_PATH/'STOP_REQUESTED').exists()or run.get('revival_disposition')=='CATASTROPHIC_REVIEW':
                        self.update(arm,status='STOPPED_AT_COOPERATIVE_BOUNDARY',updates=latest['updates'],checkpoint=str(checkpoint));raise StopBeforeDispatch('Cooperative boundary retained')
                    from .resource_boundary import cooperative_return
                    returned=cooperative_return(run,node,receipt,latest,target)
                    if returned is not None:
                        node.setdefault('cooperative_returns',[]).append(returned)
                        node.pop('job');resume=returned['checkpoint']
                        self.update(arm,status='WAITING_AUTHORIZED_MIGRATION',checkpoint=resume)
                        continue
                    raise ValueError('Native completion outside registered stage')
                if seal['updates']!=target or seal['world']!=8 or seal['next_inner']!=0:raise ValueError('Wrong formal8rank gate')
                for rank in range(8):
                    result=read(Path(run['root'])/f'result_rank{rank}.json')
                    if result['updates']!=target or result['checkpoint']!=str(checkpoint):raise ValueError('Missing full8rank completion')
                node.update(complete=True,checkpoint=str(checkpoint),end_unix=time.time());self.update(arm,status='STAGE_COMPLETE',updates=target,checkpoint=str(checkpoint));return str(checkpoint)
        def run_arm(self,initial):
            arm=initial['id'];resume=None
            try:
                while True:
                    run=self.current(arm)
                    if run.get('revival_disposition') in ('RESEARCH_RESERVE_REVIEW','CATASTROPHIC_REVIEW'):
                        self.update(arm,status='COMPLETED_AT_REGISTERED_CAP',disposition=run['revival_disposition']);return
                    for target in run['gates']:resume=self.run_stage(run,target,resume)
                    self.update(arm,status='WAITING_LONG_RUN_REVIEW',updates=run['target_updates'])
                    if run['target_updates']>=validate_registry(read(ROOT_PATH/'RUNS.json'),ROOT_PATH).get('continuation_max_updates',20000):self.update(arm,status='COMPLETED_AT_REGISTERED_CAP',updates=run['target_updates']);return
                    if self.campaign_done.wait(15):self.update(arm,status='COMPLETED_AT_REGISTERED_CAP',updates=run['target_updates']);return
                    if (ROOT_PATH/'STOP_REQUESTED').exists():raise StopBeforeDispatch('Global cooperative stop')
            except StopBeforeDispatch:self.update(arm,status='STOPPED_AT_COOPERATIVE_BOUNDARY')
            except Exception:self.update(arm,status='FAILED_REVIEW_REQUIRED',error=traceback.format_exc())
    return RevivalOwner


def run_campaign(args):
    from starVLA.rl.flow_grpo.transactions import atomic_json
    from .notify import refresh
    from .review import Review
    owner=owner_class()();folder=ROOT_PATH/'owners/REVIVAL_CAMPAIGN';folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=read(folder/'STATE.json')if(folder/'STATE.json').exists()else dict(decisions={},helpers={})
    source=Path(__file__).resolve().parents[2]
    atomic_json(folder/'OWNER.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),host=socket.gethostname(),source=__file__,source_sha256=sha(__file__),allow_continuation=args.allow_continuation,started_unix=time.time()))
    def helper(name,directory,command):
        directory.mkdir(parents=True,exist_ok=True);receipt_path=directory/'STANDARD_LAUNCH.json'
        if(directory/'STOP_OWNER').exists():return read(receipt_path)if receipt_path.exists()else None
        if receipt_path.exists():
            old=read(receipt_path)
            if alive(old):return old
            archive=directory/f"STANDARD_LAUNCH_{old['pid']}_{old['start_ticks']}.json"
            if not archive.exists():atomic_json(archive,old)
        with(directory/'standard_cpu.log').open('ab',buffering=0)as log:
            child=subprocess.Popen([os.sys.executable,'-u','-m',*command],cwd=source,env=dict(os.environ,PYTHONPATH=str(source),CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        receipt=dict(pid=child.pid,start_ticks=int(Path('/proc',str(child.pid),'stat').read_text().split()[21]),host=socket.gethostname(),command=command,source=str(source),at_unix=time.time())
        atomic_json(receipt_path,receipt);state['helpers'][name]=receipt;return receipt
    def point(arm,u):
        path=ROOT_PATH/f'analysis/PILOT_COMPARISONS/{arm}_{u}_vs_SAFETY_G8_CONTROL_{u}.json'
        if not path.exists():return None
        value=read(path)
        if value.get('status')!='COMPLETE':return None
        out=dict(value['comparison']['result'])
        out['fixed_IL70_high_advantage_PDMS']=value.get('fixed_IL70_high_advantage_PDMS',0)
        return out
    def own_trend(arm,cap,previous):
        path=ROOT_PATH/f'analysis/STANDARD_CONTINUATION/{arm}_{cap}_vs_own_{previous}.json'
        if path.exists():return read(path)['comparison']['result']
        current=ROOT_PATH/f'evaluations/{arm}_u{cap:06}/NAVTEST';base=ROOT_PATH/f'evaluations/{arm}_u{previous:06}/NAVTEST'
        if not(current/'SUMMARY.json').exists()or not(base/'SUMMARY.json').exists():return None
        review=Review.__new__(Review);review.cache={}
        comparison=review.compare(current,base,label='own_previous_long_gate')
        atomic_json(path,dict(status='COMPLETE',comparison=comparison,old_checkpoint=previous,current_checkpoint=cap,at_unix=time.time()))
        return comparison['result']
    def update_budget(run,choice,cap):
        key=f"{run['id']}:{cap}"
        path=folder/f"DECISION_{run['id']}_u{cap:06}.json"
        atomic_json(path,dict(schema='world_revival_long_run_continuation_v1',arm=run['id'],from_cap=cap,decision=choice,at_unix=time.time(),fixed_scheduler_horizon=50000,no_optimizer_RNG_or_source_change=True))
        # The qualified evaluator pins gates/target in its initial binding.
        # Cooperatively stop only its CPU owner before changing those fields;
        # all exported models, live scoring workers and native jobs stay intact.
        evdir=ROOT_PATH/'owners/RESEARCH_EVALUATION';stop=evdir/'STOP_OWNER'
        pending=state.setdefault('pending_budget_updates',{})
        intent=pending.get(key)
        if not isinstance(intent,dict):
            if stop.exists():pending[key]='EXTERNAL_EVAL_OWNER_STOP_REQUIRES_REVIEW';return
            marker=dict(reason='revival_budget_binding_adoption',decision_key=key,nonce=str(time.time_ns()),pid=os.getpid(),
                start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),host=socket.gethostname())
            intent=dict(marker=marker,status='STOP_INTENT_PERSISTED');pending[key]=intent
            # Persist ownership before creating the marker, so restart can adopt it.
            atomic_json(folder/'STATE.json',state)
        marker=intent['marker']
        if marker['host']!=socket.gethostname():raise RuntimeError('Foreign-host budget adoption intent needs exact CPU operator review')
        if stop.exists():
            if read(stop)!=marker:raise RuntimeError('External evaluator stop differs from persisted host/pid/start/key/nonce intent')
        else:atomic_json(stop,marker)
        receipt_path=evdir/'STANDARD_LAUNCH.json'
        if receipt_path.exists():
            receipt=read(receipt_path)
            if alive(receipt):
                intent['status']='WAITING_EXACT_EVAL_CPU_EXIT';atomic_json(folder/'STATE.json',state);return
        with(ROOT_PATH/'RUNS.lock').open('a')as guard:
            fcntl.flock(guard,fcntl.LOCK_EX);registry=validate_registry(read(ROOT_PATH/'RUNS.json'),ROOT_PATH);live=next(x for x in registry['runs']if x['id']==run['id'])
            if live['target_updates']==cap:
                if choice['next_cap'] is None:live['revival_disposition']=choice['action']
                else:live.update(target_updates=choice['next_cap'],gates=gates(choice['next_cap']))
                live['revival_continuation_receipt']=str(path);atomic_json(ROOT_PATH/'RUNS.json',registry)
        if read(stop)!=marker:raise RuntimeError('External CPU stop changed during exact adoption')
        stop.unlink()
        pending.pop(key,None)
        helper('evaluation',evdir,['tools.qwen_world_safety_exploration.revival_campaign','evaluate','--registry',str(ROOT_PATH/'RUNS.json'),'--max-shards','32','--max-workers','64','--interval','30'])
        state['decisions'][key]=dict(path=str(path),**choice);refresh(path,run['id']+': '+choice['action'])
    with ThreadPoolExecutor(max_workers=4)as pool:
        futures={run['id']:pool.submit(owner.run_arm,run)for run in owner.runs}
        try:
            while True:
                registry=validate_registry(read(ROOT_PATH/'RUNS.json'),ROOT_PATH)
                for newrun in registry['runs']:
                    if newrun['id']not in futures:futures[newrun['id']]=pool.submit(owner.run_arm,newrun)
                helper('evaluation',ROOT_PATH/'owners/RESEARCH_EVALUATION',['tools.qwen_world_safety_exploration.revival_campaign','evaluate','--registry',str(ROOT_PATH/'RUNS.json'),'--max-shards','32','--max-workers','64','--interval','30'])
                helper('review',ROOT_PATH/'analysis/PILOT_COMPARISONS',['tools.qwen_world_safety_exploration.revival_campaign','review','--watch','--interval','60'])
                for run in registry['runs']:
                    node=owner.state['runs'].get(run['id'],{});cap=run['target_updates'];stage=node.get('stages',{}).get(str(cap),{})
                    if not stage.get('complete')or run.get('revival_disposition')or cap>=registry.get('continuation_max_updates',20000) or f"{run['id']}:{cap}"in state['decisions']:continue
                    if not args.allow_continuation:continue
                    previous=10000 if cap==20000 else 20000 if cap==30000 else None
                    choice=decision(cap,run['revival_role'],point(run['id'],cap),point(run['id'],previous)if previous else None,own_trend(run['id'],cap,previous)if previous else None)
                    update_budget(run,choice,cap)
                if not args.allow_continuation and all(n.get('stages',{}).get('10000',{}).get('complete')or n.get('status','').startswith('FAILED')for n in owner.state['runs'].values()) and len(owner.state['runs'])==len(registry['runs']):owner.campaign_done.set()
                native_done=all(f.done()for f in futures.values())
                ev=ROOT_PATH/'owners/RESEARCH_EVALUATION/STATE.json';evstate=read(ev)if ev.exists()else{}
                state.update(status='RUNNING'if not native_done else 'TRAINING_COMPLETE_EVALUATIONS_PENDING',native=owner.state['runs'],heartbeat_unix=time.time(),host=socket.gethostname(),no_early_score_elimination=True,external_control_read_only=str(CONTROL_ROOT))
                atomic_json(folder/'STATE.json',state)
                if native_done and evstate.get('status')in ('COMPLETE_AT_REGISTERED_GATES','FINISHED_WITH_REVIEW_REQUIRED'):
                    pending=ROOT_PATH/'analysis/PILOT_COMPARISONS/STATE.json';rv=read(pending)if pending.exists()else{}
                    if rv.get('status')=='COMPLETE_AT_REGISTERED_GATES' or evstate['status']=='FINISHED_WITH_REVIEW_REQUIRED':break
                if(ROOT_PATH/'STOP_REQUESTED').exists():
                    for node in owner.state['runs'].values():
                        if node.get('job')and not node.get('status','').startswith(('FAILED','STOPPED','COMPLETED')):(Path(node['job'])/'STOP_REQUESTED').touch(exist_ok=True)
                time.sleep(30)
        except BaseException:
            state.update(status='CAMPAIGN_CPU_REVIEW_REQUIRED',error=traceback.format_exc(),heartbeat_unix=time.time());atomic_json(folder/'STATE.json',state)
            raise
        finally:
            owner.campaign_done.set()
    state.update(status='COMPLETE_AT_APPROVED_BUDGETS'if evstate.get('status')=='COMPLETE_AT_REGISTERED_GATES'else'FINISHED_WITH_REVIEW_REQUIRED',finished_unix=time.time());atomic_json(folder/'STATE.json',state);refresh(folder/'STATE.json','Revival campaign reached current approved budgets')


def run_evaluation(args):
    from .eval_owner import EvaluationOwner
    if args.registry and Path(args.registry).resolve()!=ROOT_PATH/'RUNS.json':raise ValueError('Only G registry')
    if args.max_shards!=32 or args.max_workers!=64 or args.interval<10:raise ValueError('Qualified32shards/64claims and bounded interval')
    class RevivalEvaluation(EvaluationOwner):
        def samples(self):return [(h,s)for h,s in super().samples()if h not in PROTECTED]
    RevivalEvaluation(ROOT_PATH/'RUNS.json',max_shards=32,max_workers=64,minimum=32768).run(interval=args.interval)


def run_review(args):
    from .review import Review,BASELINES
    from tools.qwen_world_research.evaluate import read_sealed
    from starVLA.rl.qwen_world_safety_exploration.contracts import fingerprint
    from starVLA.rl.flow_grpo.transactions import atomic_json
    from .notify import refresh
    class RevivalReview(Review):
        def tick(self):
            registry=validate_registry(read(ROOT_PATH/'RUNS.json'),ROOT_PATH);ready={};pending=[];published=[]
            for run in registry['runs']:
                for u in run['gates']:
                    label=f"{run['id']}_u{u:06}";ev=ROOT_PATH/'evaluations'/label/'NAVTEST';p=ev/'SUMMARY.json'
                    if not p.exists():pending.append(label);continue
                    summary=read_sealed(p)
                    if summary.get('status')!='COMPLETE' or summary.get('scenes')!=12146 or summary.get('invalid_scenes')!=0:raise ValueError('Only complete canonical full NAVTEST')
                    ready[(run['id'],u)]=(ev,summary)
                    point=self.out/f"{run['id']}_u{u:06}.json"
                    if not point.exists():
                        payload=dict(schema='world_revival_full_population_review_v1',status='COMPLETE',arm=run['id'],updates=u,evaluation=str(ev),summary_sha256=summary['sha256'],PDMS=summary['PDMS'],comparisons={n:self.compare(ev,b,label=n)for n,b in BASELINES.items()},NAVTEST_never_enters_training_selection=True)
                        payload['sha256']=fingerprint(payload);atomic_json(point,payload);published.append(str(point))
            controlrun=next(x for x in read(CONTROL_ROOT/'RUNS.json')['runs']if x['id']=='SAFETY_G8_CONTROL')
            for run in registry['runs']:
                for u in run['gates']:
                    if (run['id'],u)not in ready:continue
                    controls=[(u,'same_scenes_updates_twice_candidates'if run['group_size']==16 else 'same_updates_and_raw_exposure_count_different_scene_distribution'if run['revival_role']=='active_data'else 'same_scenes_updates_candidates')]
                    if run['group_size']==16 and 2*u in controlrun['gates']:controls.append((2*u,'same_candidates_different_scenes_updates_scheduler_clock'))
                    for cu,budget in controls:
                        base=CONTROL_ROOT/f'evaluations/SAFETY_G8_CONTROL_u{cu:06}/NAVTEST';summary=base/'SUMMARY.json';p=self.out/f"{run['id']}_{u}_vs_SAFETY_G8_CONTROL_{cu}.json"
                        if p.exists():continue
                        if not summary.exists():pending.append(f"external_F_CONTROL_u{cu:06}_for_{run['id']}_u{u:06}");continue
                        cs=read_sealed(summary)
                        if cs.get('status')!='COMPLETE'or cs.get('scenes')!=12146 or cs.get('invalid_scenes')!=0:raise ValueError('External control is partial')
                        comparison=self.compare(ready[(run['id'],u)][0],base,label='F_SAFETY_G8_CONTROL')
                        targetIL=self.compare(ready[(run['id'],u)][0],BASELINES['IL70'],label='IL70')['result'];controlIL=self.compare(base,BASELINES['IL70'],label='IL70')['result']
                        highdelta=targetIL['fixed_0p9_cohorts']['fixed_cohorts']['high']['mean_delta_PDMS']-controlIL['fixed_0p9_cohorts']['fixed_cohorts']['high']['mean_delta_PDMS']
                        payload=dict(schema='world_revival_external_control_comparison_v1',status='COMPLETE',arm=run['id'],updates=u,reference_arm='SAFETY_G8_CONTROL',reference_updates=cu,external_control_root=str(CONTROL_ROOT),external_control_summary_file_sha256=sha(summary),budget_contract=budget,comparison=comparison,fixed_IL70_high_advantage_PDMS=highdelta,F_evaluation_or_training_launched_here=False)
                        payload['sha256']=fingerprint(payload);atomic_json(p,payload);published.append(str(p))
            status=dict(status='WAITING_COMPLETE_NAVTEST'if pending else 'COMPLETE_AT_REGISTERED_GATES',ready=len(ready),pending=pending,published=published,heartbeat_unix=time.time(),external_control_read_only=True,no_training_mutations=True)
            atomic_json(self.out/'STATE.json',status)
            if published:refresh(self.out/'STATE.json','Revival full matched evidence published')
            return status
    review=RevivalReview()
    while True:
        try:state=review.tick()
        except Exception:
            atomic_json(review.out/'ERROR.json',dict(status='REVIEW_REQUIRED',error=traceback.format_exc(),at_unix=time.time()));raise
        if not args.watch or not state['pending']:return state
        if(review.out/'STOP_OWNER').exists():atomic_json(review.out/'OWNER_STOPPED.json',dict(pid=os.getpid(),at_unix=time.time(),existing_scores_untouched=True));return
        time.sleep(max(10,args.interval))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('check','self-check','run','review','evaluate'));p.add_argument('--root',default=str(ROOT_PATH));p.add_argument('--allow-continuation',action='store_true');p.add_argument('--registry');p.add_argument('--watch',action='store_true');p.add_argument('--max-shards',type=int,default=32);p.add_argument('--max-workers',type=int,default=64);p.add_argument('--interval',type=float,default=30);a=p.parse_args()
    if a.mode=='self-check':
        assert gates(20000)==[2000,5000,10000,12000,14000,16000,18000,20000]
        assert decision(10000,'inversion',dict(PDMS_delta=-.1,PDMS_CI95=[-.3,.1]))['next_cap']==20000
        weak=dict(PDMS_delta=-1.2,PDMS_CI95=[-2,-.6],new_zero=60,repaired_zero=5)
        assert decision(20000,'inversion',weak,weak,dict(PDMS_delta=0,PDMS_CI95=[-.1,.1]))['next_cap']==30000
        assert decision(20000,'inversion',weak,weak,dict(PDMS_delta=-.1,PDMS_CI95=[-.2,-.01]))['action']=='RESEARCH_RESERVE_REVIEW'
        print('PASS: no early pruning; uncertain continues; strong repeated futility only; every2k cadence');return
    configure(a.root);registry=validate_registry(read(ROOT_PATH/'RUNS.json'),ROOT_PATH)
    if a.mode=='review':return run_review(a)
    if a.mode=='evaluate':return run_evaluation(a)
    if a.mode=='check':
        cls=owner_class()
        print(json.dumps(dict(status='CPU_REGISTRY_AND_REAL_OWNER_IMPORT_COMPATIBLE_NO_DISPATCH',root=str(ROOT_PATH),owner_class=cls.__name__,arms=[r['id']for r in registry['runs']],initial_caps=[r['target_updates']for r in registry['runs']],source_may_be_pending_freeze=True)));return
    run_campaign(a)

if __name__=='__main__':main()
