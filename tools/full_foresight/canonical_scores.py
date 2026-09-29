"""Score immutable development exports in one audited CPU environment.

Does not modify trainers, predictions, caches, or the original score tables.
Identical existing score identities are reused; different runtime environments
are rescored into an independent directory before paired comparisons.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.local_interaction_mask_v2.compare_pdms import METRICS, read
from tools.local_interaction_mask_v2.score_async import digest


def read_json(path):return json.loads(Path(path).read_text())


def runtime_effect(original, canonical):
    """Same predictions/cache/population; measure only the CPU scoring difference."""
    first=read(Path(original)/'scenes.csv');second=read(Path(canonical)/'scenes.csv')
    if set(first)!=set(second):raise ValueError('Scoring populations differ')
    tokens=sorted(first)
    for token in tokens:
        if any(first[token][k]!=second[token][k] for k in ('log','proposal_sha256','metric_cache_sha256','status')):
            raise ValueError('Scoring inputs or failures changed; not a runtime-only comparison')
    if any(first[t]['status']!='ok' for t in tokens):raise ValueError('Failed scenes cannot prove scoring parity')
    result={'scenes':len(tokens),'same_prediction_and_cache':True,'metrics':{}}
    for metric in METRICS:
        a=np.asarray([float(first[t][metric]) for t in tokens]);b=np.asarray([float(second[t][metric]) for t in tokens])
        if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Invalid scoring comparison')
        delta=b-a
        result['metrics'][metric]={'changed_scenes_above_1e_12':int((abs(delta)>1e-12).sum()),
            'max_abs_difference':float(abs(delta).max()),'mean_difference':float(delta.mean())}
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('campaign-root','reference-score','devkit','metric-index','current-index','scoring-python'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--registrations',nargs='+',required=True)
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--watch-seconds',type=int,default=0);p.add_argument('--interval',type=int,default=60)
    a=p.parse_args();root=Path(a.campaign_root)
    if not 1<=a.workers<=16 or a.interval<10 or a.watch_seconds<0:raise ValueError('Bounded CPU observer required')
    if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Lock clean observer source')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    reference=read_json(Path(a.reference_score)/'summary.json')
    expected=reference['evaluator_identity']
    gate=read_json(root/'scoring_protocol_v1.json')
    if not gate['passed'] or digest(a.metric_index)!=gate['metric_index_sha256'] or expected['navsim_python_tree_sha256']!=gate['navsim_tree_sha256']:
        raise ValueError('Audited official cache/evaluator required')
    if not reference['valid'] or reference['failed'] or reference['diagnostic']:raise ValueError('Complete formal reference required')
    versions=json.loads(subprocess.check_output([a.scoring_python,'-c',
        "import importlib.metadata,json;print(json.dumps({n:importlib.metadata.version(n) for n in ('numpy','scipy','shapely')}))"],text=True))
    if versions!=expected['runtime_versions']:raise ValueError('Canonical Python differs from audited reference runtime')
    registration_paths=[Path(x) for x in a.registrations];registrations=[read_json(x) for x in registration_paths]
    milestones=registrations[0]['development_updates']
    if any(v['development_updates']!=milestones for v in registrations):raise ValueError('Unmatched milestone schedule')
    out=root/'canonical_evaluation_v1';out.mkdir(exist_ok=True)
    lock=(out/'observer.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    contract={'registration_sha256':[digest(x) for x in registration_paths],
              'reference_score_sha256':digest(Path(a.reference_score)/'summary.json'),
              'evaluator_identity':expected,'observer_source':source}
    state_path=out/'status.json'
    if state_path.exists():
        state=read_json(state_path)
        if state['contract']!=contract:raise ValueError('Canonical observer contract changed')
    else:state={'contract':contract,'events':[]}
    state.update(status='RUNNING',pid=os.getpid(),started_unix=time.time())
    def publish(event):
        state['events'].append({'time':time.time(),**event});state['updated_unix']=time.time();atomic_json(state_path,state)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    deadline=time.monotonic()+a.watch_seconds
    try:
        while True:
            if (root/'STOP_REQUESTED').exists():
                state.update(status='PAUSED',updated_unix=time.time());atomic_json(state_path,state);return
            complete=True
            for step in milestones:
                entries=[]
                for reg in registrations:
                    for run_id,spec in reg['runs'].items():
                        label=f'{run_id}_dev{step}_seed42';original=root/'scores'/label
                        if not (original/'summary.json').exists():complete=False;continue
                        raw=read_json(original/'summary.json')
                        if not raw['valid'] or raw['failed'] or raw['diagnostic']:raise ValueError('Invalid original score cannot be promoted')
                        checkpoint=raw['export_identity']['checkpoint']
                        if checkpoint['completed']!=step or checkpoint['training_source_sha']!=reg['training_source_sha'] or checkpoint['arm']!=spec['candidate']:
                            raise ValueError('Unregistered checkpoint in score table')
                        score=original
                        if raw['evaluator_identity']!=expected:
                            score=root/'scores_canonical_env_v1'/label;score.parent.mkdir(exist_ok=True)
                            if not (score/'summary.json').exists():
                                cmd=[a.scoring_python,'-m','tools.foresight.score_pdms','--devkit',a.devkit,
                                    '--metric-index',a.metric_index,'--current-index',a.current_index,
                                    '--predictions',str(root/'evaluations'/label),'--output',str(score),
                                    '--campaign-root',str(root),'--run-id',label+'_canonical_'+str(time.time_ns()),'--workers',str(a.workers)]
                                if score.exists():cmd.append('--resume')
                                publish({'event':'CPU_RESCORE','label':label,'command':cmd})
                                started=time.time()
                                with (root/'logs'/(label+'_canonical_observer.log')).open('a') as stream:
                                    subprocess.run(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
                                publish({'event':'CPU_RESCORE_DONE','label':label,'workers':a.workers,
                                         'wall_seconds':time.time()-started,'gpu_count':0})
                            actual=read_json(score/'summary.json')
                            if actual['evaluator_identity']!=expected or actual['export_identity']!=raw['export_identity'] or not actual['valid']:
                                raise ValueError('Canonical score changed environment/export or failed')
                            effect_path=out/(label+'_runtime_effect.json')
                            if not effect_path.exists():atomic_json(effect_path,runtime_effect(original,score))
                        entries.append({'arm':spec['candidate'],'training_seed':spec['seed'],'sampling_seed':42,'score_dir':str(score)})
                expected_count=sum(len(r['runs']) for r in registrations)
                if len(entries)==expected_count:
                    registry=out/f'dev{step}_registry.json';paired=out/f'dev{step}_paired'
                    if registry.exists() and read_json(registry)!=entries:raise ValueError('Canonical score mapping changed')
                    if not registry.exists():atomic_json(registry,entries)
                    if not (paired/'summary.json').exists():
                        cmd=[sys.executable,'-m','tools.full_foresight.summarize_planning','--registry',str(registry),
                            '--output',str(paired),'--split','dev','--sampling-seeds','42','--screen',
                            '--expected-arms',','.join(e['arm'] for e in entries)]
                        subprocess.run(cmd,env=env,check=True)
                        publish({'event':'PAIRED_REPORT','step':step,'report':str(paired/'summary.json')})
            state['all_seed42_milestones_complete']=complete;state['updated_unix']=time.time();atomic_json(state_path,state)
            if complete or time.monotonic()>=deadline or (root/'STOP_REQUESTED').exists():
                state['status']='SEED42_MILESTONES_COMPLETE' if complete else 'PAUSED';atomic_json(state_path,state);return
            time.sleep(min(a.interval,max(0,deadline-time.monotonic())))
    except BaseException as error:
        state.update(status='FAILED',error=repr(error));atomic_json(state_path,state);raise


if __name__=='__main__':main()
