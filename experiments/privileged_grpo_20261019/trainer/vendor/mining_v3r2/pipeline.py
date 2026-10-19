"""Prepare, infer, score and assemble a resumable single-PPU navtrain pass.

Historical predictions are allowed only with --historical-smoke and never
produce training-ready pairs. All mutations stay under this task directory.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from common import ROOT, read, write, digest, file_hash, raw_scene, poses_valid, decode


def configuration():
    return read(ROOT/'CONFIG.json')


def prepare():
    cfg = configuration()
    tokens = read(cfg['train_tokens'])
    assert len(tokens) == len(set(tokens)) == cfg['population_expected']
    logs = {r['token']:r['log'] for r in read(cfg['log_index'])}
    assert set(tokens) == set(logs)
    metric = {}
    with open(cfg['metric_index']) as f:
        for row in csv.DictReader(f):
            p = Path(row['file_name']); token = p.parent.name
            assert token not in metric, f'Duplicate cache {token}'
            metric[token] = str(p)
    population = []
    for token in tokens:
        assert len(token)==16 and all(c in '0123456789abcdef' for c in token)
        if token not in metric: raise ValueError(f'Missing metric index: {token}')
        p = Path(metric[token])
        assert p.parents[2].name == logs[token], 'Metric log mismatch'
        population.append(dict(token=token, log=logs[token], metric=str(p), raw_meta=str(Path(cfg['raw_meta_root'])/(token+'.pkl'))))
    from score_trace import protocol
    score_protocol = protocol(cfg)
    source = Path(cfg['source_root'])
    paths = sorted((source/'starVLA').rglob('*.py'))
    paths += [Path(cfg['run_root'])/'config.yaml',Path(cfg['run_root'])/'config.json']
    paths += sorted(Path(cfg['tokenizer_root']).glob('*.json'))
    paths += sorted(Path(cfg['tokenizer_root']).glob('*.jinja'))
    paths += [Path(cfg['tokenizer_root'])/'merges.txt']
    identity = dict(config_hash=digest(cfg), population_hash=digest(population),
                    checkpoint=cfg['checkpoint'], checkpoint_sha256=file_hash(cfg['checkpoint']),
                    source_hashes={str(p):file_hash(p) for p in paths},
                    pipeline_hashes={p.name:file_hash(p) for p in sorted(ROOT.glob('*.py'))},
                    shell_hashes={p.name:file_hash(p) for p in sorted(ROOT.glob('*.sh'))},
                    score_protocol=score_protocol,
                    inputs={p:file_hash(p) for p in (cfg['train_tokens'],cfg['metric_index'],cfg['log_index'])},
                    base_seen='training split listed in original run config; exact historical exposure not audited',
                    prediction_seed_policy='sha256(global_seed:token), one sample per scene, batch1',
                    strict_load_required=True, metadata_coverage=len(population))
    assert identity['checkpoint_sha256'] == cfg['checkpoint_sha256']
    identity['identity'] = digest(identity)
    write(ROOT/'POPULATION.json', population)
    write(ROOT/'IDENTITY.json', identity)
    print(json.dumps({'status':'PREPARED','scenes':len(population),'identity':identity['identity']}),flush=True)


def verify():
    cfg, identity = configuration(), read(ROOT/'IDENTITY.json')
    assert digest(cfg)==identity['config_hash'], 'Config changed: use a new run directory'
    assert digest(read(ROOT/'POPULATION.json'))==identity['population_hash']
    for name, expected in identity['pipeline_hashes'].items():
        assert file_hash(ROOT/name)==expected, 'Pipeline changed after lock: '+name
    for name, expected in identity['shell_hashes'].items():
        assert file_hash(ROOT/name)==expected, 'Launcher changed after lock: '+name
    for p, expected in identity['source_hashes'].items():
        assert file_hash(p)==expected, 'Original source/tokenizer changed: '+p
    for p, expected in identity['score_protocol']['source_hashes'].items():
        assert file_hash(p)==expected, 'Scoring source changed: '+p
    return cfg, identity


def load_images(scene):
    from PIL import Image
    result=[]
    for path in scene['image_paths']:
        p=Path(path)
        if not p.is_file():
            # Documented complete camera mirror; preserve log/camera/image suffix.
            prefix='/mnt/data/navsim/trainval_sensor_blobs/trainval/'
            if not str(p).startswith(prefix): raise FileNotFoundError(p)
            p=Path('/mnt/data/navsim/trainval_all/trainval_sensor_blobs/trainval')/str(p)[len(prefix):]
        with Image.open(p) as image:
            im=image.convert('RGB');width,height=im.size;ratio=1024/576
            if width/height>ratio:
                new=int(height*ratio);left=(width-new)//2;im=im.crop((left,0,left+new,height))
            elif width/height<ratio:
                new=int(width/ratio);top=(height-new)//2;im=im.crop((0,top,width,top+new))
            result.append(im.resize((1024,576),Image.Resampling.LANCZOS))
    return result


def infer(limit, output_name, rank=0, world_size=1, barrier=False):
    cfg, identity=verify()
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()<1:
        raise RuntimeError('No accelerator exposed to this process; refusing CPU fallback for full VLM inference')
    assert file_hash(cfg['checkpoint'])==identity['checkpoint_sha256']
    sys.path.insert(0,cfg['source_root'])
    from omegaconf import OmegaConf
    from starVLA.model.framework.QwenOFT import Qwenvl_OFT
    import starVLA.model.framework.QwenOFT as qwen_module
    # Resolve via importlib because package attributes can shadow module names.
    import importlib
    qwen_module=importlib.import_module('starVLA.model.framework.QwenOFT')
    assert Path(qwen_module.__file__).resolve().is_relative_to(Path(cfg['source_root']).resolve())
    model_cfg=OmegaConf.load(Path(cfg['run_root'])/'config.yaml')
    model_cfg.framework.qwenvl.base_vlm=cfg['tokenizer_root']
    model_cfg.framework.qwenvl.attn_implementation='sdpa'
    model=Qwenvl_OFT(model_cfg,infer_not_load_wan=1)
    state=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state,strict=True)
    model._inference_qwen_forward_mode='optimized'
    model=model.to('cuda:0').eval()
    output=ROOT/output_name;output.mkdir(exist_ok=True)
    assert 0 <= rank < world_size <= 4
    receipt=output/f'MODEL_LOAD.rank{rank}.json'
    load=dict(identity=identity['identity'],checkpoint_sha256=identity['checkpoint_sha256'],strict_load=True,
              accelerator=torch.cuda.get_device_name(0),dtype_policy='original constructor dtype; checkpoint values loaded strictly',
              qwen_forward_mode='optimized',attention='sdpa',smooth=False,batch=1, rank=rank, world_size=world_size)
    if receipt.exists():assert read(receipt)==load
    else:write(receipt,load)
    rows=read(ROOT/'POPULATION.json')
    if limit:rows=rows[:limit]
    rows=rows[rank::world_size]
    write(output/f'READY.rank{rank}.json', dict(ready_unix=time.time(), rank=rank), replace=True)
    if barrier:
        deadline=time.time()+600
        while not (output/'GO.json').exists():
            if time.time()>deadline:raise TimeoutError('Inference start barrier timed out')
            time.sleep(.1)
    started=time.time();done=0;generated=0;reused=0
    for row in rows:
        token=row['token'];out=output/(token+'.json')
        if out.exists():
            prior=read(out);assert prior['identity']==identity['identity'] and prior['token']==token;poses_valid(prior['trajectory']);done+=1;reused+=1;continue
        scene=raw_scene(row);seed=int(hashlib.sha256(f"{cfg['seed']}:{token}".encode()).hexdigest()[:8],16)
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            prediction=model.predict_action_infer_1d([dict(image=load_images(scene),state=scene['state'],lang=scene['instruction'],token=token)])
        raw=np.asarray(prediction['normalized_actions'])[0]
        physical=decode(raw)
        record=dict(identity=identity['identity'],token=token,log=row['log'],seed=seed,raw=raw.tolist(),
                    trajectory=physical.tolist(),checkpoint_sha256=identity['checkpoint_sha256'],
                    raw_meta_sha256=file_hash(row['raw_meta']))
        write(out,record);done+=1;generated+=1
        write(output/f'STATUS.rank{rank}.json',dict(stage='inference',rank=rank,world_size=world_size,completed=done,generated=generated,reused=reused,total=len(rows),elapsed_s=time.time()-started,updated_unix=time.time()),replace=True)
        if done%10==0:print(f'inference {done}/{len(rows)}',flush=True)
    write(output/f'COMPLETE.rank{rank}.json',dict(identity=identity['identity'],rank=rank,world_size=world_size,completed=done,generated=generated,reused=reused,total=len(rows),started_unix=started,finished_unix=time.time(),elapsed_s=time.time()-started),replace=True)


def score_row(job):
    row, cfg, identity, prediction_dir, output, historical=job
    token=row['token'];output=Path(output);start=time.time()
    try:
        scene=raw_scene(row)
        if historical:
            pred_path=Path(cfg['historical_predictions'])/(token+'.npy')
            pred=poses_valid(np.load(pred_path,allow_pickle=False))
        else:
            pred_path=Path(prediction_dir)/(token+'.json');record=read(pred_path)
            assert record['identity']==identity['identity'] and record['token']==token
            assert record['raw_meta_sha256']==file_hash(row['raw_meta'])
            pred=poses_valid(record['trajectory'])
        from score_trace import score_one
        base=score_one(pred,row['metric'],cfg)
        assert np.allclose(base['initial_global_rear_axle'][:2],scene['origin'][:2],rtol=0,atol=1e-3), 'Raw GT and metric cache origin mismatch'
        from common import wrap
        assert abs(wrap(base['initial_global_rear_axle'][2]-scene['origin'][2]))<1e-5
        gt=score_one(scene['gt'],row['metric'],cfg)
        from guidance import build_guidance
        guidance=build_guidance(pred,scene['gt'],base,gt,scene['navigation'],cfg)
        result=dict(schema='failure_gt_pair_v3',identity=identity['identity'],token=token,log=row['log'],
                    prediction_provenance='historical_unverified_replay' if historical else 'fresh_strict_checkpoint_inference',
                    training_ready=bool(guidance['eligibility']['accepted'] and not historical),
                    checkpoint_sha256=identity['checkpoint_sha256'],score_protocol_sha256=identity['score_protocol']['sha256'],
                    source_paths=row,prediction_path=str(pred_path),prediction_sha256=file_hash(pred_path),
                    raw_meta_sha256=file_hash(row['raw_meta']),metric_cache_sha256=file_hash(row['metric']),
                    input_trajectory=pred.tolist(),gt_trajectory=scene['gt'].tolist(),
                    base_score=base,gt_score=gt,guidance=guidance,elapsed_s=time.time()-start)
        write(output/'records'/(token+'.json'),result)
        return dict(token=token,status='ok',accepted=guidance['eligibility']['accepted'],training_ready=result['training_ready'])
    except Exception as e:
        error=dict(token=token,status='failed',error=repr(e),traceback=traceback.format_exc())
        write(output/'errors'/(token+'.json'),error,replace=True)
        return error


def score(limit, workers, historical, prediction_name, output_name):
    cfg,identity=verify();rows=read(ROOT/'POPULATION.json')
    if historical:
        assert 0<limit<=64, 'Historical predictions limited to diagnostic smoke'
    if limit:rows=rows[:limit]
    output=ROOT/output_name;output.mkdir(exist_ok=True)
    if not historical:
        model=read(ROOT/prediction_name/'MODEL_LOAD.json')
        assert model['identity']==identity['identity'] and model['strict_load']
    request=dict(identity=identity['identity'],historical_smoke=historical,tokens=[r['token'] for r in rows])
    if (output/'REQUEST.json').exists():assert read(output/'REQUEST.json')==request
    else:write(output/'REQUEST.json',request)
    pending=[];done=0;failed=0;start=time.time()
    # Do not serialize the full model/source hash manifest for every scene.
    worker_identity={k:identity[k] for k in ('identity','checkpoint_sha256')}
    worker_identity['score_protocol']={'sha256':identity['score_protocol']['sha256']}
    for row in rows:
        p=output/'records'/(row['token']+'.json')
        if p.exists():
            existing=read(p);assert existing['identity']==identity['identity']
            assert file_hash(existing['prediction_path'])==existing['prediction_sha256'], 'Prediction changed after scoring'
            assert file_hash(row['raw_meta'])==existing['raw_meta_sha256'], 'Raw labels changed after scoring'
            assert file_hash(row['metric'])==existing['metric_cache_sha256'], 'Metric context changed after scoring'
            done+=1
        else:pending.append((row,cfg,worker_identity,str(ROOT/prediction_name),str(output),historical))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(score_row,pending,chunksize=1):
            done+=result['status']=='ok';failed+=result['status']!='ok'
            write(output/'STATUS.json',dict(stage='scoring',completed=done,failed=failed,total=len(rows),updated_unix=time.time(),elapsed_s=time.time()-start),replace=True)
            print(json.dumps({k:result[k] for k in ('token','status')}),flush=True)
            if failed>=3 and done==0:raise RuntimeError('Three consecutive initial scoring failures; stop for diagnosis')
    summarize(output,rows,historical,identity)


def summarize(output,rows,historical,identity):
    from collections import Counter
    records=[];errors=[];reasons=Counter();metrics={k:[] for k in ('PDMS','NC','DAC','TTC','EP','CF','DDC')}
    pairs=[];examples=[]
    for row in rows:
        p=output/'records'/(row['token']+'.json')
        if not p.exists():errors.append(row['token']);continue
        r=read(p);records.append(r['token']);e=r['guidance']['eligibility'];reasons[e['reason']]+=1
        for k in metrics:metrics[k].append(r['base_score']['metrics'][k])
        if e['accepted']:
            pair=dict(token=r['token'],log=r['log'],record=str(p),record_sha256=file_hash(p),training_ready=r['training_ready'])
            pairs.append(pair)
            if len(examples)<8:examples.append(dict(token=r['token'],training_ready=r['training_ready'],guidance=r['guidance']['guidance_text']))
    summary=dict(identity=identity['identity'],historical_smoke=historical,status='COMPLETE' if not errors else 'INCOMPLETE',requested=len(rows),
                 scored=len(records),coverage=len(records)/len(rows),failed_tokens=errors,pair_eligibility_counts=dict(reasons),
                 accepted_pairs=len(pairs),training_ready_pairs=sum(p['training_ready'] for p in pairs),
                 mean_base_metrics={k:float(np.mean(v)) if v else None for k,v in metrics.items()},
                 denominator_note='Means use valid scored scenes only; failed and missing scenes explicitly listed.',
                 interpretation='Historical prediction identity not replay-verified; not a fresh checkpoint evaluation.' if historical else 'Current training-set evaluation; no unseen-test claim.')
    write(output/'SUMMARY.json',summary,replace=True)
    write(output/'PAIRS.json',pairs,replace=True)
    write(output/'EXAMPLES.json',examples,replace=True)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','infer','score'])
    parser.add_argument('--limit',type=int,default=0);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--historical-smoke',action='store_true')
    parser.add_argument('--prediction-name',default='predictions');parser.add_argument('--output-name',default='evaluation')
    parser.add_argument('--rank',type=int,default=0);parser.add_argument('--world-size',type=int,default=1)
    parser.add_argument('--barrier',action='store_true')
    args=parser.parse_args()
    for name in (args.prediction_name,args.output_name):
        if Path(name).name!=name or name in ('.','..'):raise ValueError('Output must be a direct task subdirectory')
    if args.stage=='prepare':prepare()
    elif args.stage=='infer':infer(args.limit,args.prediction_name,args.rank,args.world_size,args.barrier)
    else:score(args.limit,args.workers,args.historical_smoke,args.prediction_name,args.output_name)


if __name__=='__main__':main()
