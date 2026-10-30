"""Finite event-driven CPU analysis of registered FULL/BATCHSTD12k probes.

No journal polling, model load, scoring, GPU dispatch, job stop or training
barrier. Existing10k exact caches are mandatory; new12k scenes are read once.
"""
from __future__ import annotations
import argparse,ctypes,fcntl,hashlib,json,os,select,struct,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path('/mnt/project/qwen-world-rl-optimization-20261030')
ANALYSIS=ROOT/'analysis'
OWNER_ROOT=ROOT/'owners/MATCHED12K_ANALYSIS'
RESULT=ANALYSIS/'MATCHED_12K_PPO_CONTRASTS.json'
LABELS=('PPO_FULL_u012000_NAVTEST','PPO_BATCHSTD_u012000_NAVTEST')
ARMS=('PPO_FULL','PPO_BATCHSTD')
NOW=lambda:datetime.now(timezone.utc).isoformat()


def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')
    os.replace(temp,path)


def refresh(status):
    # The request is administrative; evidence points to the separate scientific
    # result. The existing event watcher accepts scientific_result=False.
    atomic(ROOT/'reporting/refresh_request.json',dict(
        id=f'MATCHED12K_ANALYSIS_{status}_{NOW()}',scientific_result=False,
        reason=f'有限12k配对分析状态更新：{status}',evidence=str(RESULT),analysis_status=status,
        evidence_kind='completed_supplemental_analysis' if status=='COMPLETE' else 'supplemental_analysis_status'))


class EvaluationFailure(RuntimeError):pass


class MetadataEvents:
    """Linux directory inotify; no periodic state/journal reads."""
    HEADER=struct.Struct('iIII')
    MASK=0x8|0x80|0x100  # CLOSE_WRITE, MOVED_TO, CREATE
    def __init__(self):
        self.libc=ctypes.CDLL(None,use_errno=True)
        self.libc.inotify_init1.argtypes=[ctypes.c_int];self.libc.inotify_init1.restype=ctypes.c_int
        self.libc.inotify_add_watch.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_uint32]
        self.libc.inotify_add_watch.restype=ctypes.c_int
        self.fd=self.libc.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC)
        if self.fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
        self.paths={}
    def add(self,path):
        path=Path(path)
        if not path.is_dir() or path in self.paths.values():return
        wd=self.libc.inotify_add_watch(self.fd,os.fsencode(path),self.MASK)
        if wd<0:raise OSError(ctypes.get_errno(),'inotify_add_watch',str(path))
        self.paths[wd]=path
    def discover(self,specs):
        for p in [ROOT,ROOT/'owners/CAMPAIGN',ROOT/'owners/PPO_FULL',ROOT/'owners/PPO_BATCHSTD',
                  ROOT/'evaluation_specs',ROOT/'jobs']:
            self.add(p)
        for label in LABELS:
            family=ROOT/'jobs'/('EVAL_'+label);self.add(family)
            if family.exists():
                for attempt in family.glob('rank*_attempt*'):self.add(attempt)
        for spec in specs.values():
            if spec:self.add(Path(spec['output']))
    def wait(self,remaining):
        ready,_,_=select.select([self.fd],[],[],max(0,remaining))
        if not ready:return []
        changed=[]
        while True:
            try:raw=os.read(self.fd,262144)
            except BlockingIOError:break
            pos=0
            while pos<len(raw):
                wd,mask,cookie,length=self.HEADER.unpack_from(raw,pos);pos+=self.HEADER.size
                name=os.fsdecode(raw[pos:pos+length].split(b'\0',1)[0]);pos+=length
                if mask&0x4000:changed.append('INOTIFY_OVERFLOW');continue
                directory=self.paths.get(wd)
                if directory is None:continue
                if mask&0x40000000:
                    if name in ['EVAL_'+x for x in LABELS] or name.startswith('rank'):
                        changed.append(str(directory/name))
                    continue
                if name in ('STATE.json','OWNER.json','SUMMARY.json','RESULT.json') or name in [x+'.json' for x in LABELS]:
                    changed.append(str(directory/name))
        return sorted(set(changed))
    def close(self):os.close(self.fd)


def current_specs_and_ready(probes):
    specs={label:read(ROOT/'evaluation_specs'/(label+'.json')) for label in LABELS}
    complete=[]
    campaign=read(ROOT/'owners/CAMPAIGN/STATE.json',{})
    for arm,label in zip(ARMS,LABELS):
        node=campaign.get('runs',{}).get(arm,{}).get('stages',{}).get('12000',{}).get('NAVTEST',{})
        if node.get('evaluation_error') or node.get('status') in ('FAILED_REQUIRES_REVIEW','FAILED','FATAL','ABORTED'):
            raise EvaluationFailure(f'{label}: '+str(node.get('evaluation_error',node.get('status'))))
        spec=specs[label]
        if not spec:continue
        if spec.get('rl_checkpoint')!=probes[arm]['checkpoint']:
            raise EvaluationFailure('Registered12k checkpoint changed: '+label)
        summary_path=Path(spec['output'])/'SUMMARY.json'
        summary=read(summary_path)
        if summary:
            if (summary.get('status')!='COMPLETE' or summary.get('scenes')!=12146
                    or summary.get('invalid_scenes',0)!=0 or summary.get('RL_updates')!=12000):
                raise EvaluationFailure('12k summary is not the complete registered result: '+label)
            complete.append(label)
        # Admission75 before child startup is capacity waiting, not science failure.
        for job in spec.get('jobs',[]):
            p=Path(job['receipt'] if isinstance(job,dict) else job)
            res=read(p/'RESULT.json')
            if res and res.get('returncode') not in (None,0):
                if res.get('returncode')==75 and res.get('child_not_started') is True:continue
                raise EvaluationFailure('Evaluation child failed: '+str(p)+' '+str(res.get('status'))+' rc='+str(res.get('returncode')))
    return specs,complete


def perform_analysis(probes):
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from analyze_portfolio_10k import load_once,contrast,physical_log,METRICS,CACHE
    base_labels=('PPO_FULL_u010000_NAVTEST','PPO_BATCHSTD_u010000_NAVTEST')
    for label in base_labels:
        if not (CACHE/(label+'.npz')).exists() or not (CACHE/(label+'.json')).exists():
            raise RuntimeError('Required10k exact cache missing; refusing to reparse old scenes: '+label)
    with ThreadPoolExecutor(max_workers=2) as pool:
        values=dict(zip(LABELS,pool.map(load_once,LABELS)))
    for label in base_labels:values[label]=load_once(label)
    for label in LABELS:
        if values[label]['identity']['RL_updates']!=12000:raise ValueError('Wrong12k identity')
    pairs=[('PPO_BATCHSTD_u012000_NAVTEST','PPO_FULL_u012000_NAVTEST'),
        ('PPO_BATCHSTD_u012000_NAVTEST','PPO_BATCHSTD_u010000_NAVTEST'),
        ('PPO_FULL_u012000_NAVTEST','PPO_FULL_u010000_NAVTEST')]
    contrasts={}
    for target,baseline in pairs:
        a,b=values[target],values[baseline];out=contrast(a,b)
        # Same scene-weighted44-log resamples, simultaneously for components.
        tokens=a['identity']['tokens'];order=list(dict.fromkeys(physical_log(a['rows'][t]['log']) for t in tokens))
        if len(order)!=44:raise ValueError('Expected44 physical logs')
        loc={name:i for i,name in enumerate(order)};groups=np.asarray([loc[physical_log(a['rows'][t]['log'])] for t in tokens])
        n=np.bincount(groups,minlength=44).astype(float);sums=np.zeros((44,len(METRICS)))
        np.add.at(sums,groups,a['matrix']-b['matrix'])
        draws=np.random.default_rng(42).integers(0,44,size=(2000,44))
        boot=100*sums[draws].sum(1)/n[draws].sum(1)[:,None]
        lo,hi=np.quantile(boot,[.025,.975],axis=0)
        if not np.allclose([lo[0],hi[0]],out['CI95_PDMS'],atol=1e-10,rtol=0):
            raise ValueError('Component bootstrap must retain the source pooled-log estimator')
        for i,name in enumerate(METRICS):out['safety_metrics'][name]['CI95_delta_percentage_points']=[float(lo[i]),float(hi[i])]
        contrasts[target+'_minus_'+baseline]=out
    matched=contrasts[pairs[0][0]+'_minus_'+pairs[0][1]]
    ci=matched['CI95_PDMS'];nc=matched['safety_metrics']['no_at_fault_collisions'];ncci=nc['CI95_delta_percentage_points']
    persistence=('12k同预算仍有配对正收益，支持10k收益持续。' if ci[0]>0 else
        '12k同预算区间跨零，不能确认10k收益持续或消失。' if ci[1]>=0 else
        '12k同预算转为负差且区间排除零，10k正收益未持续至本次12k探针。')
    nc_note=('12k BATCH相对FULL的NC下降区间排除零，需root审查NC风险。' if ncci[1]<0 else
        '12k BATCH相对FULL的NC区间跨零，不能宣称NC改善或无风险。' if ncci[0]<=0 else
        '12k BATCH相对FULL的NC差为正且区间排除零，未重现10k NC点下降。')
    prior=read(ANALYSIS/'PORTFOLIO_10K_PAIRED.json',{})
    old=prior.get('contrasts',{}).get('PPO_BATCHSTD_u010000_NAVTEST_minus_PPO_FULL_u010000_NAVTEST',{})
    body=dict(schema='world_matched_12k_PPO_contrasts_v1',status='COMPLETE',utc=NOW(),contrasts=contrasts,
        conclusions=dict(persistence_zh=persistence,NC_risk_zh=nc_note,
            prior10k=dict(delta_PDMS=old.get('delta_PDMS'),CI95_PDMS=old.get('CI95_PDMS'),
                         NC_delta_percentage_points=old.get('safety_metrics',{}).get('no_at_fault_collisions',{}).get('delta_percentage_points')),
            no_automatic_training_action=True),
        evaluations={label:dict(PDMS=x['summary']['PDMS'],provenance=x['provenance']) for label,x in values.items()},
        matching_evidence=dict(scenes=12146,physical_logs=44,bootstrap=2000,bootstrap_seed=42,evaluation_seed=2101,
            exact_same_tokens_data_protocol_numerics=True,cache_hashes_and_logs_matched=True,
            inference='single-candidate ODE10 shift5',registered_checkpoint_probes=probes),
        uncertainty='固定训练seed42/评测seed2101下的44物理log簇bootstrap；不是重复训练不确定性，未作多重比较校正；观察Test已影响后续研究选择。',
        scope=dict(CPU_only=True,new_forwards=0,new_scores=0,new_seeds=0,old10k_scene_parses=0,
            new12k_population_parses_at_most_once=True,training_targets_unchanged=True,GPU_process_actions=0),
        provenance=dict(script=str(Path(__file__).resolve()),script_sha256=sha(__file__),registration=str(ROOT/'ADDITIONAL_NAVTEST_PROBES.json')))
    body['sha256']=fingerprint(body)
    return body


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--max-wait-seconds',type=float,default=21600)
    args=parser.parse_args()
    OWNER_ROOT.mkdir(parents=True,exist_ok=True)
    lock=(OWNER_ROOT/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    registered=read(ROOT/'ADDITIONAL_NAVTEST_PROBES.json')
    probes={x['arm']:x for x in registered['evaluations'] if x['arm'] in ARMS and x['updates']==12000}
    if set(probes)!=set(ARMS):raise ValueError('Two explicit registered12k probes required')
    started=time.time();deadline=started+args.max_wait_seconds
    owner=dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),source_file=str(Path(__file__).resolve()),
        source_sha256=sha(__file__),started_at=NOW(),deadline_UTC=datetime.fromtimestamp(deadline,timezone.utc).isoformat(),
        CPU_only=True,finite_labels=LABELS,registration_sha256=sha(ROOT/'ADDITIONAL_NAVTEST_PROBES.json'),
        target_owner_at_launch=read(ROOT/'owners/CAMPAIGN/OWNER.json',{}),
        event_backend='Linux directory inotify; source owner STATE atomic replacements and exact eval metadata',journal_polling=False)
    atomic(OWNER_ROOT/'OWNER.json',owner)
    notify=None
    try:
        previous=read(RESULT)
        if previous and previous.get('status')=='COMPLETE':
            atomic(OWNER_ROOT/'STATE.json',dict(status='COMPLETE_EXISTING',utc=NOW(),evidence=str(RESULT),no_reanalysis=True))
            return 0
        if previous:
            atomic(OWNER_ROOT/'STATE.json',dict(status='EXISTING_TERMINAL_RESULT_REQUIRES_REVIEW',utc=NOW(),
                evidence=str(RESULT),prior_status=previous.get('status'),no_result_overwrite=True))
            return 2
        notify=MetadataEvents();specs={};notify.discover(specs)
        last_complete=None
        while True:
            specs,complete=current_specs_and_ready(probes);notify.discover(specs)
            if complete!=last_complete:
                atomic(OWNER_ROOT/'STATE.json',dict(status='WAITING_SUMMARIES',utc=NOW(),complete_labels=complete,
                    pending_labels=[x for x in LABELS if x not in complete],evidence=str(RESULT),GPU_process_actions=0))
                last_complete=complete
            if len(complete)==2:break
            remaining=deadline-time.time()
            if remaining<=0:raise TimeoutError('Finite6h event wait expired; no partial analysis or training action')
            changed=notify.wait(remaining)
            if changed:print(json.dumps(dict(utc=NOW(),metadata_events=changed[:20])),flush=True)
        atomic(OWNER_ROOT/'STATE.json',dict(status='ANALYZING',utc=NOW(),complete_labels=complete,evidence=str(RESULT)))
        result=perform_analysis(probes);atomic(RESULT,result)
        atomic(OWNER_ROOT/'STATE.json',dict(status='COMPLETE',utc=NOW(),evidence=str(RESULT),GPU_process_actions=0))
        refresh('COMPLETE');print(json.dumps(dict(status='COMPLETE',result=str(RESULT))),flush=True)
        return 0
    except Exception as exc:
        status='FAILED_EVALUATION' if isinstance(exc,EvaluationFailure) else 'WAIT_TIMEOUT' if isinstance(exc,TimeoutError) else 'FAILED_REQUIRES_REVIEW'
        failure=dict(schema='world_matched_12k_PPO_contrasts_v1',status=status,utc=NOW(),error=traceback.format_exc(),
            no_partial_scores_imputed=True,registration_probes=probes,GPU_process_actions=0,training_targets_unchanged=True)
        atomic(RESULT,failure)
        atomic(OWNER_ROOT/'STATE.json',dict(status=status,event='matched12k_analysis_failed',utc=NOW(),evidence=str(RESULT),error=str(exc)))
        refresh(status);print(failure['error'],file=sys.stderr,flush=True)
        return 2
    finally:
        if notify is not None:notify.close()


if __name__=='__main__':raise SystemExit(main())
