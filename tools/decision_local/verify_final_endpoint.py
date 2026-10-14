"""Verify completed registered artifacts before the single final Navtest pass.

No score threshold or checkpoint selection. This is a finite read-only receipt
check, followed by one atomic acceptance file for the existing evaluation queue.
"""
import argparse
import math
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,check_host,digest,read,source_sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--max-seconds',type=float,required=True)
    p.add_argument('--once',action='store_true');a=p.parse_args();check_host();start=time.monotonic();r=a.artifacts
    lockpath=r/'final_navtest_v1/FINAL_METHOD_LOCK.json';lock=read(lockpath);locked_sha=digest(lockpath)
    heads=[j for j in read(r/'HEAD_TASKS.json')['jobs'] if j['scope']=='formal']
    dev=[j for j in read(r/'development_observations_v1/REGISTRY.json')['jobs'] if j['epoch']==24]
    if len(heads)!=15 or len(dev)!=10 or lock['main_endpoint_epochs']!=24:raise ValueError('Registered endpoint list changed')
    online=r/'formal_deployment_v1'
    required=[Path(j['output'])/'COMPLETE.json' for j in heads]+[Path(j['scores'])/'COMPLETE.json' for j in dev]
    required += [online/f"seed{x['seed']}_{x['kind']}.json" for x in lock['camera_only_runs']]
    required += [online/'online_DIFF_64/COMPLETE.json',r/'final_navtest_v1/features/COMPLETE.json',r/'final_navtest_v1/queries/COMPLETE.json']
    while time.monotonic()-start<a.max_seconds:
        check_host()
        missing=[str(path) for path in required if not path.exists()]
        if missing:
            if a.once:print('Pending completed artifacts:',len(missing));return
            time.sleep(10);continue
        if digest(lockpath)!=locked_sha:raise ValueError('Method lock changed while waiting')
        checkpoints={};sources={}
        for job in heads:
            root=Path(job['output']);identity=read(root/'identity.json');done=read(root/'COMPLETE.json')
            if done['epoch']!=24 or done['completed']!=math.ceil(101592/128)*24 or done['exposure']!=101592*24:
                raise ValueError('Incomplete shared training exposure')
            if not done['full_training_population'] or identity['frozen_base_optimizer_updates']!=0:raise ValueError('Different foundation/population')
            for key,expected in [('source_sha',lock['small_head_training_source_sha']),('features',lock['features_identity']),('labels',lock['labels_identity'])]:
                if identity[key]!=expected:raise ValueError('Different training source/data: '+key)
            path=root/'epoch_24.pt';checkpoints[str(path)]=digest(path);sources[job['id']]=identity['source_sha']
        for job in dev:
            score=read(Path(job['scores'])/'COMPLETE.json');meta=read(Path(job['scores'])/'identity.json')
            pred=read(Path(job['predictions'])/'identity.json')
            if meta['population']!='dev' or score['identity']!=meta['identity'] or meta['prediction_identity']!=pred['identity']:
                raise ValueError('Development protocol or output mismatch')
            if pred['checkpoint_sha256']!=checkpoints[job['checkpoint']]:raise ValueError('Development result not from locked24epoch weight')
            for variant in ('q0','q_final'):
                if not score[variant]['valid'] or score[variant]['failed'] or score[variant]['scenes']!=1696:
                    raise ValueError('Incomplete/invalid full development endpoint')
        verification=[]
        for item in lock['camera_only_runs']:
            path=online/f"seed{item['seed']}_{item['kind']}.json";proof=read(path)
            if not proof['passed'] or proof['anchor_identity']!=lock['anchor_identity'] or proof['refiner_sha256']!=checkpoints[item['checkpoint']]:
                raise ValueError('Wrong or unverified formal online head')
            if len(proof['scenes'])!=4 or any(not row['one_Qwen_forward'] or row['GT_fields_rejected']!=5 for row in proof['scenes']):
                raise ValueError('Incomplete one-base current-only checks')
            verification.append({'path':str(path),'sha256':digest(path)})
        deploy=online/'online_DIFF_64';done=read(deploy/'COMPLETE.json');identity=read(deploy/'identity.json')
        if done['scenes']!=64 or not done['python_training_file_access_guard'] or done['opened_GT_relation_or_feature_cache']:
            raise ValueError('Real current-image deployment not complete')
        job=next(j for j in dev if j['kind']=='R_FUTURE_DIFF' and j['seed']==42)
        if identity['head_checkpoint_sha256']!=checkpoints[job['checkpoint']]:raise ValueError('Wrong guarded deployment endpoint')
        errors={'q0':0.,'q_final':0.}
        for row in identity['index']:
            token=row['token']
            with np.load(deploy/'predictions'/(token+'.npz')) as x,np.load(Path(job['predictions'])/'predictions'/(token+'.npz')) as y:
                for key in errors:
                    np.testing.assert_allclose(x[key],y[key],rtol=1e-6,atol=1e-6)
                    errors[key]=max(errors[key],float(np.abs(x[key]-y[key]).max()))
        atomic_json(r/'final_navtest_v1/FINAL_READINESS.json',{'accepted':True,'method_lock_sha256':locked_sha,
            'source_sha':source_sha(ROOT),'tool_sha256':digest(__file__),'checkpoint_sha256':checkpoints,'training_sources':sources,
            'formal_heads_complete':len(heads),'full_development_endpoints':len(dev),'online_verification':verification,
            'guarded_online64_max_absolute_difference':errors,'no_PDMS_threshold_or_best_checkpoint_selection':True,
            'accepted_unix':time.time(),'scope':'Common24epoch primary arms only; any conditional predictor OOF audit remains separately reported'})
        return
    atomic_json(r/'final_navtest_v1/READINESS_PENDING.json',{'reason':'Finite wait ended; no final scoring acceptance issued'})


if __name__=='__main__':main()
