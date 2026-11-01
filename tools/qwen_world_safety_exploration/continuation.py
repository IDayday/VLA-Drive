"""Evidence-gated finite continuation after complete same-budget pilot reviews.

No scientific config, checkpoint, GPU process or old result is rewritten.
Native stages continue their exact original Adam/RNG/data state and source.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from starVLA.rl.qwen_world_safety_exploration.contracts import ROOT,file_sha,fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json
from .notify import refresh

CONTROL='SAFETY_G8_CONTROL'
STAGES=(3000,5000,10000)


def read(p):return json.loads(Path(p).read_text())


def alive(receipt):
    p=Path('/proc')/str(receipt['pid'])
    try:
        s=(p/'stat').read_text().split()
        return int(s[21])==receipt['start_ticks'] and s[2]not in ('Z','X')
    except FileNotFoundError:return False


def sealed(p):
    v=read(p);b=dict(v);h=b.pop('sha256',None)
    if h!=fingerprint(b) or v.get('status')!='COMPLETE':raise ValueError('Incomplete/changed paired review: '+str(p))
    return v


def select_candidates(candidates, *, initial):
    """Predeclared single-seed screening; uncertainty never equals proof of gain."""
    eligible=[];decisions=[]
    for arm,v in candidates.items():
        delta=v['PDMS_delta'];upper=v['PDMS_CI95'][1];net=v['new_zero']-v['repaired_zero'];high=v['fixed_IL70_high_delta_PDMS']
        if initial:
            accept=upper>0 and delta>=-.10 and net<=10 and high>=-.10
        else:
            accept=upper>0 and delta>=.02 and net<=5 and high>=-.05
        if v.get('absolute_plateau_strikes',0)>=2:accept=False
        reason='PROMISING_OR_UNCERTAIN_FULL_PAIRED' if accept else 'STOP_AT_REGISTERED_CAP'
        decisions.append(dict(arm=arm,decision=reason,delta_PDMS=delta,CI95_PDMS=v['PDMS_CI95'],net_new_zero=net,
            fixed_IL70_high_delta_PDMS=high,absolute_PDMS=v.get('absolute_PDMS'),absolute_plateau_strikes=v.get('absolute_plateau_strikes',0),not_a_general_training_seed_CI=True))
        if accept:eligible.append((delta, -net, high, arm))
    eligible.sort(reverse=True)
    selected=[v[-1]for v in eligible[:2]]
    for d in decisions:
        if d['arm']not in selected and d['decision']!='STOP_AT_REGISTERED_CAP':d['decision']='LOWER_PRIORITY_WITHIN_MAX_TWO_CONTINUATIONS'
    return selected,decisions


class Continuation:
    def __init__(self):
        self.out=ROOT/'owners/CONTINUATION';self.out.mkdir(parents=True,exist_ok=True)
        self.lock=(self.out/'owner.lock').open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.path=self.out/'STATE.json'
        self.state=read(self.path)if self.path.exists()else dict(status='WAITING_COMPLETE_PILOT_REVIEWS',current_update=1000,stage_index=0,novels=None)
        self.source=Path(__file__).resolve().parents[2]
        atomic_json(self.out/'OWNER.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),source=__file__,source_sha256=file_sha(__file__)))

    def flush(self):
        self.state['heartbeat_unix']=time.time();atomic_json(self.path,self.state)

    def old_owners_done(self):
        for family in ('RESEARCH_NATIVE','PHASE2_NATIVE','RESEARCH_EVALUATION','PILOT_REVIEW'):
            p=ROOT/'owners'/family/'LAUNCH.json'
            if p.exists()and alive(read(p)):return False
        return True

    def evidence(self,registry):
        update=self.state['current_update'];root=ROOT/'analysis/PILOT_COMPARISONS'
        arms=self.state['novels']
        if arms is None:arms=[r['id']for r in registry['runs']if r['id']!=CONTROL]
        cp=root/f'{CONTROL}_u{update:06}.json'
        if not cp.exists():return None
        control=sealed(cp);high=control['comparisons']['IL70']['result']['fixed_0p9_cohorts']['fixed_cohorts']['high']['mean_delta_PDMS']
        values={};paths=[str(cp)]
        history=self.state.setdefault('absolute_score_history',{});strikes=self.state.setdefault('absolute_plateau_strikes',{})
        for arm in arms:
            p=root/f'{arm}_{update}_vs_{CONTROL}_{update}.json';own=root/f'{arm}_u{update:06}.json'
            if not p.exists()or not own.exists():return None
            comparison=sealed(p);body=comparison['comparison']['result'];ownv=sealed(own)
            values[arm]=dict(body,fixed_IL70_high_delta_PDMS=ownv['comparisons']['IL70']['result']['fixed_0p9_cohorts']['fixed_cohorts']['high']['mean_delta_PDMS']-high)
            previous=history.setdefault(arm,{})
            if str(update)not in previous:
                prior=previous.get(str(self.state.get('previous_update',0)))
                strikes[arm]=0 if prior is None or ownv['PDMS']>prior+.02 else strikes.get(arm,0)+1
                previous[str(update)]=ownv['PDMS']
            values[arm].update(absolute_PDMS=ownv['PDMS'],absolute_plateau_strikes=strikes.get(arm,0))
            paths.extend([str(p),str(own)])
        return values,paths

    def launch(self,family,args):
        directory=ROOT/'owners'/family;directory.mkdir(parents=True,exist_ok=True);p=directory/'LAUNCH.json'
        if p.exists():
            old=read(p)
            expected=str(self.out/f"DECISION_u{self.state['previous_update']:06}.json")
            if old.get('continuation_decision')==expected:return old
            if alive(old):raise RuntimeError('Different live CPU owner; no duplicate or eviction')
            archive=directory/f"LAUNCH_before_continuation_{old['pid']}.json"
            if archive.exists():raise ValueError('Owner receipt already archived')
            p.rename(archive)
        command=[os.sys.executable,'-u','-m',*args]
        env=dict(os.environ,WORLD_COMPARE_ROOT=str(ROOT),WORLD_COMPARE_TASK='qwen_world_safety_exploration_20261031',PYTHONPATH=str(self.source),CUDA_VISIBLE_DEVICES='')
        with(directory/'continuation_owner.log').open('ab',buffering=0)as log:
            child=subprocess.Popen(command,cwd=self.source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        receipt=dict(pid=child.pid,start_ticks=int(Path('/proc',str(child.pid),'stat').read_text().split()[21]),
            source=str(self.source),command=command,at_unix=time.time(),continuation_decision=str(self.out/f"DECISION_u{self.state['previous_update']:06}.json"))
        atomic_json(p,receipt)
        return receipt

    def extend(self,registry,selected,next_update,paths,decisions):
        active=[CONTROL,*selected];old=self.state['current_update']
        decision=dict(schema='world_finite_continuation_decision_v1',status='DECIDED',at_unix=time.time(),
            from_updates=old,to_updates=next_update,selected_novels=selected,control_continues=True,
            evidence=[dict(path=p,sha256=file_sha(p))for p in paths],decisions=decisions,
            scientific_config_unchanged=True,native_Adam_RNG_cursor_pending_preserved=True,GPU_signals=0)
        atomic_json(self.out/f'DECISION_u{old:06}.json',decision)
        for run in registry['runs']:
            if run['id']in active:
                run['gates'].append(next_update);run['target_updates']=next_update
                run['continuation_decision']=str(self.out/f'DECISION_u{old:06}.json')
            elif run['id']!=CONTROL:run['disposition']='COMPLETED_FINITE_PILOT_NOT_SELECTED_FOR_EXTENSION'
        plan_path=self.out/f'PLAN_u{next_update:06}.json'
        atomic_json(plan_path,dict(previous_registry_fingerprint=fingerprint(read(ROOT/'RUNS.json')),
            new_registry_fingerprint=fingerprint(registry),registry=registry))
        self.state.update(status='WAITING_CONTINUED_FULL_EVALUATIONS',previous_update=old,current_update=next_update,
            stage_index=self.state['stage_index']+1,novels=selected,active=active,
            registry_plan=str(plan_path),registry_plan_applied=False)
        grouped={}
        for run in registry['runs']:
            if run['id']in active:grouped.setdefault(Path(run['owner_state']).parent.name,[]).append(run['id'])
        self.state['launch_plan']={family:['tools.qwen_world_safety_exploration.owner','--owner-name',family,'--ids',*ids] for family,ids in grouped.items()}
        self.state['launch_plan'].update(RESEARCH_EVALUATION=['tools.qwen_world_safety_exploration.eval_owner','--max-shards','32','--max-workers','64'],
            PILOT_REVIEW=['tools.qwen_world_safety_exploration.review','--watch'])
        self.state['stage_launch_receipts']={};self.flush()
        self.ensure_launches()
        refresh(self.path,'Full paired evidence selected bounded native continuation')

    def ensure_launches(self):
        if self.state.get('registry_plan')and not self.state.get('registry_plan_applied'):
            plan=read(self.state['registry_plan']);value=read(ROOT/'RUNS.json');h=fingerprint(value)
            if fingerprint(plan['registry'])!=plan['new_registry_fingerprint']:raise ValueError('Changed bounded registry plan')
            if h==plan['previous_registry_fingerprint']:atomic_json(ROOT/'RUNS.json',plan['registry'])
            elif h!=plan['new_registry_fingerprint']:raise ValueError('Foreign concurrent registry change; retained intent, no overwrite')
            self.state['registry_plan_applied']=True;self.flush()
        for family,args in self.state.get('launch_plan',{}).items():
            if family not in self.state.setdefault('stage_launch_receipts',{}):
                self.state['stage_launch_receipts'][family]=self.launch(family,args);self.flush()

    def tick(self):
        if (ROOT/'STOP_REQUESTED').exists():
            self.state.update(status='STOPPED_BY_GLOBAL_REQUEST_NO_NEW_DISPATCH',finished_unix=time.time());self.flush();return True
        self.ensure_launches()
        registry=read(ROOT/'RUNS.json')
        if registry['task']!='qwen_world_safety_exploration_20261031':raise ValueError('Wrong task')
        evidence=self.evidence(registry)
        if evidence is None or not self.old_owners_done():self.flush();return False
        values,paths=evidence
        chosen,decisions=select_candidates(values,initial=self.state['current_update']==1000)
        if not chosen or self.state['stage_index']>=len(STAGES):
            self.state.update(status='COMPLETE_NO_PROMISING_EXTENSION' if not chosen else 'COMPLETE_FINITE_10K_RESEARCH_STAGE',
                decisions=decisions,finished_unix=time.time(),research_goal_92_not_implicitly_claimed=True)
            self.flush();refresh(self.path,'Full paired review closed finite continuation stage');return True
        self.extend(registry,chosen,STAGES[self.state['stage_index']],paths,decisions);return False

    def run(self):
        while True:
            if (self.out/'STOP_OWNER').exists():self.state['status']='STOPPED_CPU_REVIEW_ONLY';self.flush();return
            try:
                if self.tick():return
            except Exception:
                import traceback
                self.state.update(status='OWNER_REVIEW_REQUIRED',error=traceback.format_exc());self.flush();refresh(self.path,'Continuation review requires diagnosis');raise
            time.sleep(30)

if __name__=='__main__':Continuation().run()
