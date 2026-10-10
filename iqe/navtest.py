"""Explicit current-only Query Base checkpoint probes on official full Navtest.

This entry does not qualify/freeze a Base or write training/selector artifacts.
Navtest is a checkpoint monitoring benchmark once the user invokes this entry.
"""
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
from pathlib import Path
import argparse
import json
import os
import time
import multiprocessing
import numpy as np
from .io import atomic_json, read_json, file_hash, digest, lock
from .contracts import require, strict_record, CandidateRecord, ScoreRecord
from .scoring.navtest_schema import NavtestInput
from .scoring.reference import ReferenceBackend, COMPONENTS


def backend(config):
    m=config['metric']
    return ReferenceBackend(m['navsim_root'],m['python'],{k:m[k] for k in ('proposal_sampling','scorer')},
                            timeout=m['timeout_seconds'],eval_seed=m['eval_seed'])


def code_hash():
    root=Path(__file__).parent
    return digest({str(p.relative_to(root)):file_hash(p) for p in sorted(root.rglob('*.py'))})


def lock_dataset(config,current_root,metric_index,output,workers):
    require(1 <= workers <= 16, 'bounded manifest I/O workers')
    root=Path(current_root);identity=read_json(root/'identity.json');index=read_json(root/'index.json')
    metric=read_json(metric_index)
    require(identity['split']=='navtest' and not identity['future_in_cache'] and not identity['annotations_in_cache'], 'current-only Navtest inputs required')
    require(identity['cameras']==['CAM_F0','CAM_L0','CAM_R0'] and identity['image_size']==[1024,576] and identity['history_images']==0, 'input contract differs')
    require(digest(index)==identity['index_sha256'], 'Navtest index changed')
    pairs={(r['token'],r['log']) for r in index}
    require(len(index)==len(pairs)==12146 and len({r['log'] for r in index})==136, 'complete official Navtest population required')
    require(len(metric)==12146 and {(r['token'],r['log']) for r in metric}==pairs,'official metric population mismatch')
    for split in ('train_root','dev_root'):
        rows=read_json(Path(config['data'][split])/'index.json')
        require(not {r['token'] for r in rows} & {t for t,l in pairs} and
                not {r['log'] for r in rows} & {l for t,l in pairs}, 'Navtest overlaps training/development logs')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    definition={'current_identity':identity,'metric_index_path':str(Path(metric_index).resolve()),
                'metric_index_hash':file_hash(metric_index),'current_root':str(root.resolve()),'scene_count':12146,'logs':136}
    atomic_json(out/'INPUTS.json',definition,immutable=True)
    if (out/'DATASET.json').exists():
        data=read_json(out/'DATASET.json')
        require(data['inputs']==definition and data['dataset_hash']==digest(data['records']), 'Navtest dataset identity mismatch')
        return data
    lookup={r['token']:r for r in metric}
    def one(row):
        obs_ref=root/'current'/(row['token']+'.json');obs=read_json(obs_ref)
        require(obs['token']==row['token'] and obs['log']==row['log'] and obs['identity']==identity['identity'],'observation ID mismatch')
        require(len(obs['image_paths'])==3,'camera count')
        obs_hash=digest({'record':obs,'images':{p:file_hash(p) for p in obs['image_paths']}})
        cache=lookup[row['token']]['cache_path']
        return asdict(NavtestInput(1,row['token'],row['log'],obs_hash,str(obs_ref),cache,file_hash(cache)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records=list(pool.map(one,sorted(index,key=lambda r:r['token'])))
    result={'inputs':definition,'records':records,'dataset_hash':digest(records),'role':'navtest_checkpoint_monitoring',
            'no_GT_or_future_actor_input':True,'training_writeback':False}
    atomic_json(out/'DATASET.json',result,immutable=True)
    return result


def lock_checkpoint(config,checkpoint,dataset,output):
    from .training.resume import inspect_checkpoint
    import torch
    info=inspect_checkpoint(checkpoint);require(info['schema']=='iqe_query_base','Query checkpoint required')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    require(state['mode']=='full' and not state['learned_driving_weights_loaded'],'formal framework-trained Query Base required')
    contract=read_json(config['s0']['contract'])
    require(state['framework_contract_hash']==contract['framework_contract_hash'],'framework mismatch')
    require(not set(state['fit_scene_ids']) & {r['scene_id'] for r in dataset['records']},'checkpoint saw Navtest scenes')
    contract=deepcopy(contract);contract.update(binding_status='LOCKED_QUERY_S0',query_checkpoint=info['path'],query_checkpoint_hash=info['checksum'])
    b=backend(config);require(b.source_hash==contract['metric']['python_source_hash'],'official source changed')
    result={'purpose':'user_requested_checkpoint_navtest_monitoring','checkpoint':info,'contract':contract,
            'dataset_hash':dataset['dataset_hash'],'score_protocol_hash':b.protocol_hash,'code_hash':code_hash(),
            'precision':'inherited Query S0 fp32 master/bf16 encoder autocast; one scene per forward',
            'scorer':None,'candidates_per_scene':1,'formal_qualification':False,'training_writeback':False}
    result['identity']=digest(result)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'EVALUATION_LOCK.json',result,immutable=True)
    atomic_json(out/'NAVTEST_ACCESS.json',{'purpose':result['purpose'],'checkpoint':info,'dataset_hash':dataset['dataset_hash'],
               'benchmark_used_for_checkpoint_observation':True,'untouched_final_test_claim':False},immutable=True)
    from .config import resolve
    resolve(config,contract,out/'resolved_config.json')
    return result


def load_locked(dataset,output):
    data=read_json(dataset);record=read_json(Path(output)/'EVALUATION_LOCK.json')
    require(data['dataset_hash']==digest(data['records'])==record['dataset_hash'],'Navtest dataset changed')
    require(record['identity']==digest({k:v for k,v in record.items() if k!='identity'}),'Navtest lock changed')
    require(record['code_hash']==code_hash(),'Navtest code changed; use an explicit new evaluation version')
    return data,record


def export_shard(config,dataset,output,rank,world_size,device):
    import torch
    from .s0_adapter import S0Adapter
    from .training.randomness import initialization_rng
    from .data.candidate_bank import save_trajectory
    from .evaluation.retention import module_hash
    from .resources import qualify
    require(0<=rank<world_size<=8,'export shard budget');qualify(device)
    torch.set_num_threads(config['execution']['num_threads'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data,record=load_locked(dataset,output);out=Path(output)
    atomic_json(out/'EXPORT_LAYOUT.json',{'world_size':world_size,'batch':1,'device':device},immutable=True)
    assigned=data['records'][rank::world_size];start=time.perf_counter()
    with lock(out/f'export_rank_{rank}.lock'), initialization_rng(42):
        adapter=S0Adapter.load_and_validate_s0(record['contract'],device);before=module_hash(adapter)
        from starVLA.dataloader.foresight_dataset import current_observation
        done=0;latencies=[]
        for row in assigned:
            scene=strict_record(NavtestInput,row);metadata=out/'predictions'/(scene.scene_id+'.json')
            if metadata.exists():
                old=read_json(metadata)
                require(old['identity']==record['identity'] and old['observation_hash']==scene.observation_hash and
                        file_hash(old['candidate']['trajectory_ref'])==old['candidate']['trajectory_hash'],'export cache changed')
                done+=1;continue
            obs=read_json(scene.observation_ref)
            require(digest({'record':obs,'images':{p:file_hash(p) for p in obs['image_paths']}})==scene.observation_hash,'current images changed')
            t=time.perf_counter();observation=current_observation(obs);observation['token']=scene.scene_id
            with torch.no_grad():
                result=adapter.predict_s0(adapter.encode_scene([observation]))
            physical=result.physical.cpu();raw=result.raw.cpu()
            require(torch.isfinite(raw).all() and torch.isfinite(physical).all(),'nonfinite Base trajectory; evaluation invalid')
            latency=time.perf_counter()-t;latencies.append(latency)
            ref=out/'trajectories'/(scene.scene_id+'.npz');th=save_trajectory(ref,physical[0].numpy())
            c=CandidateRecord(1,scene.scene_id,'expert_0',record['checkpoint']['checksum'],digest(record['contract']),th,str(ref),
                record['contract']['trajectory']['raw_representation'],'ego_relative_rear_axle',physical.shape[1],record['contract']['trajectory']['dt'],
                True,record['code_hash'],record['identity'],'navtest_checkpoint_monitoring')
            atomic_json(metadata,{'identity':record['identity'],'observation_hash':scene.observation_hash,'candidate':asdict(c),
                        'raw':raw[0].tolist(),'inference_seconds':latency},immutable=True)
            done+=1
            if done%20==0 or done==1:
                atomic_json(out/f'export_rank_{rank}.json',{'status':'RUNNING','completed':done,'requested':len(assigned),'elapsed_seconds':time.perf_counter()-start,
                    'recent_mean_seconds':float(np.mean(latencies[-20:])),'peak_memory_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None})
        require(module_hash(adapter)==before,'Base changed during export')
        atomic_json(out/f'export_rank_{rank}.json',{'status':'COMPLETE','completed':done,'requested':len(assigned),'elapsed_seconds':time.perf_counter()-start,'frozen_unchanged':True})


def score_one(payload):
    # Import/module reuse only: each call reloads its own cache and creates fresh
    # official simulator/scorer. RNG resets and official pdm_score remain intact.
    from .scoring.worker import main
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(prefix='iqe-navtest-reference-') as tmp:
        req,res=Path(tmp)/'request.json',Path(tmp)/'score.json'
        atomic_json(req,payload);main(str(req),str(res))
        return read_json(res)


def payload_for(scene,candidate,b):
    require(file_hash(scene.metric_context_ref)==scene.metric_context_hash,'official cache changed')
    require(file_hash(candidate.trajectory_ref)==candidate.trajectory_hash,'trajectory changed')
    return {'candidate':asdict(candidate),'scene':asdict(scene),'scene_schema':'navtest_input_v1',
            'protocol_hash':b.protocol_hash,'navsim_root':b.root,'source_hash':b.source_hash,'config':b.config,
            'seed':int(digest({'scene_id':scene.scene_id,'seed':b.eval_seed,'repetition':0})[:8],16),
            'eval_seed':b.eval_seed,'repetition':0}


def score_all(config,dataset,output,workers,timeout):
    from .resources import qualify
    qualify('cpu');require(1<=workers<=16 and timeout>0,'explicit scoring budget')
    data,record=load_locked(dataset,output);out=Path(output);b=backend(config)
    require(b.protocol_hash==record['score_protocol_hash'],'metric configuration changed')
    rows={r['scene_id']:strict_record(NavtestInput,r) for r in data['records']};done={};pending={};start=time.monotonic()
    for path in (out/'scores').glob('*.json'):
        score=strict_record(ScoreRecord,read_json(path));scene=rows[path.stem]
        exported=read_json(out/'predictions'/(scene.scene_id+'.json'));candidate=strict_record(CandidateRecord,exported['candidate'])
        require(exported['identity']==record['identity'],'wrong cached export identity')
        payload_for(scene,candidate,b)
        require(score.candidate_key==candidate.key and score.metric_context_hash==scene.metric_context_hash and score.score_protocol_hash==b.protocol_hash,'stale Navtest score')
        done[scene.scene_id]=asdict(score)
    if (out/'NAVTEST_RESULT.json').exists():
        previous=read_json(out/'NAVTEST_RESULT.json')
        require(previous['dataset_hash']==data['dataset_hash'] and previous['code_hash']==record['code_hash'] and
                previous['checkpoint']==record['checkpoint'] and len(done)==len(rows), 'completed Navtest identity changed')
        require(previous['status']=='COMPLETE','previous Navtest evaluation invalid')
        return previous
    # Prove persistent imports equal independent subprocess reference on four
    # actual scenes, including reversed/repeated calls, before enabling full work.
    proof=out/'REFERENCE_EQUIVALENCE.json'
    while not proof.exists():
        ready=sorted((out/'predictions').glob('*.json'))[:4]
        if len(ready)<4:
            require(time.monotonic()-start<timeout,'timed out waiting for GPU export');time.sleep(2);continue
        payloads=[];expected=[]
        for path in ready:
            scene=rows[path.stem];c=strict_record(CandidateRecord,read_json(path)['candidate'])
            payloads.append(payload_for(scene,c,b));expected.append(asdict(b.score(c,scene)))
        require(all(s['score_valid'] for s in expected),'reference invalid')
        for i in [0,1,2,3,3,1,0,2,0]:
            require(score_one(payloads[i])==expected[i],'persistent import scoring differs from independent reference')
        atomic_json(proof,{'passed':True,'protocol_hash':b.protocol_hash,'code_hash':record['code_hash'],
                    'scenes':[p.stem for p in ready],'independent_calls':4,'persistent_calls':9,'identical_named_components_and_total':True},immutable=True)
    evidence=read_json(proof);require(evidence['passed'] and evidence['code_hash']==record['code_hash'] and evidence['protocol_hash']==b.protocol_hash,'stale scoring equivalence')
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        while len(done)<len(rows):
            require(time.monotonic()-start<timeout,'Navtest scoring time budget exhausted; partial scores retained')
            in_flight=set(pending.values())
            for sid,scene in rows.items():
                if len(pending)>=2*workers:break
                if sid in done or sid in in_flight:continue
                path=out/'predictions'/(sid+'.json')
                if not path.exists():continue
                exported=read_json(path);require(exported['identity']==record['identity'],'wrong export identity')
                c=strict_record(CandidateRecord,exported['candidate'])
                pending[pool.submit(score_one,payload_for(scene,c,b))]=sid
            if pending:
                finished,_=wait(pending,timeout=2,return_when=FIRST_COMPLETED)
                for future in finished:
                    sid=pending.pop(future);score=strict_record(ScoreRecord,future.result())
                    atomic_json(out/'scores'/(sid+'.json'),asdict(score),immutable=True);done[sid]=asdict(score)
            else:
                require(time.monotonic()-start<timeout,'timed out waiting for remaining exports');time.sleep(2)
            atomic_json(out/'SCORING_PROGRESS.json',{'status':'RUNNING','completed':len(done),'requested':len(rows),
                        'invalid':sum(not s['score_valid'] for s in done.values()),'seconds':time.monotonic()-start,'workers':workers})
    valid=[(sid,s) for sid,s in done.items() if s['score_valid']]
    report={'status':'COMPLETE' if len(valid)==len(rows) else 'INVALID','checkpoint':record['checkpoint'],
            'scenes':len(rows),'valid_scores':len(valid),'logs':len({s.source_log_id for s in rows.values()}),
            'PDMS_01':float(np.mean([s['total_score_01'] for sid,s in valid])) if valid else None,
            'zero_fraction':float(np.mean([s['total_score_01']<=1e-8 for sid,s in valid])) if valid else None,
            'components':{c:float(np.mean([s['named_components'][c] for sid,s in valid if s['component_valid_masks'][c]])) for c in COMPONENTS},
            'per_log':{log:{'scenes':sum(rows[sid].source_log_id==log for sid,s in valid),
                'PDMS_01':float(np.mean([s['total_score_01'] for sid,s in valid if rows[sid].source_log_id==log]))} for log in sorted({rows[sid].source_log_id for sid,s in valid})},
            'dataset_hash':data['dataset_hash'],'score_protocol_hash':b.protocol_hash,'code_hash':record['code_hash'],
            'evaluation_purpose':'user_requested_checkpoint_monitoring','formal_qualification':False,'training_writeback':False,
            'wall_seconds':time.monotonic()-start}
    from .evaluation.oracle import paired_group_bootstrap
    # Constant zero comparator gives an absolute log-bootstrap interval for the mean.
    if valid:
        values=np.array([s['total_score_01'] for sid,s in valid]);groups=[rows[sid].source_log_id for sid,s in valid]
        report['log_bootstrap']=paired_group_bootstrap(values,groups,config['execution']['bootstrap_repetitions'],config['execution']['bootstrap_seed'])
    atomic_json(out/'NAVTEST_RESULT.json',report,immutable=True)
    require(report['status']=='COMPLETE','invalid official scores; do not report valid full benchmark')
    return report


def main(argv=None):
    p=argparse.ArgumentParser(__doc__);p.add_argument('command',choices=['lock-data','lock-checkpoint','export','score'])
    p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--dataset')
    p.add_argument('--current-root');p.add_argument('--metric-index');p.add_argument('--checkpoint')
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--timeout',type=int,default=14400)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args(argv)
    from .config import load_config
    c=load_config(a.config)
    try:
        if a.command=='lock-data':
            require(a.current_root and a.metric_index,'real current input and official metric index required')
            result=lock_dataset(c,a.current_root,a.metric_index,a.output,a.workers)
            result={'status':'COMPLETE','scenes':len(result['records']),'dataset_hash':result['dataset_hash']}
        elif a.command=='lock-checkpoint':
            require(a.dataset and a.checkpoint,'dataset and checkpoint required')
            result=lock_checkpoint(c,a.checkpoint,read_json(a.dataset),a.output)
            result={'status':'COMPLETE','identity':result['identity'],'checkpoint':result['checkpoint']}
        elif a.command=='export':
            require(a.dataset,'dataset required');export_shard(c,a.dataset,a.output,a.rank,a.world_size,a.device);result={'status':'COMPLETE','rank':a.rank}
        else:
            require(a.dataset,'dataset required')
            with lock(Path(a.output)/'scoring.lock'):
                result=score_all(c,a.dataset,a.output,a.workers,a.timeout)
        print(json.dumps(result,allow_nan=False));return 0
    except Exception as error:
        atomic_json(Path(a.output)/(a.command+f'_rank_{a.rank}.FAILED.json'),{'status':'FAILED','error':repr(error)})
        print(json.dumps({'status':'FAILED','error':repr(error)}));return 2


if __name__=='__main__':
    raise SystemExit(main())
