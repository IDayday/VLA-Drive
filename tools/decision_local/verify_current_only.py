"""Real-image pinned-S0 online/cache equivalence and GT rejection checks."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('dl_helpers',Path(__file__).with_name('common.py'))
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)


def load_runtime(anchor):
    legacy=Path(anchor['source_root'])
    if helpers.source_sha(legacy)!=anchor['training_source_sha']:raise ValueError('Pinned S source changed')
    sys.path.insert(0,str(legacy))
    import starVLA
    if Path(starVLA.__file__).parent.resolve()!=(legacy/'starVLA').resolve():raise ValueError('Legacy source shadowed')
    import starVLA.model.modules as modules
    modules.__path__.append(str(ROOT/'starVLA/model/modules'))
    spec=importlib.util.spec_from_file_location('decision_local_deployment',ROOT/'starVLA/model/framework/frozen_vla_consequence_refiner.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('anchor','cache','labels','refiner','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--predictor',type=Path)
    p.add_argument('--samples',type=int,default=4);a=p.parse_args();resources=helpers.check_host()
    anchor=helpers.read(a.anchor);runtime=load_runtime(anchor)
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from tools.foresight.checkpoints import load_student
    from starVLA.dataloader.foresight_dataset import ForesightCurrentDataset,decode_ego
    from starVLA.model.modules.decision_local.consequence_head import ConsequenceQueryHead
    from starVLA.model.modules.decision_local.residual_refiner import ResidualRefiner
    from starVLA.model.modules.decision_local.consequence_contract import ModelInputs,PredictionQueries,predicted_relations
    from starVLA.model.modules.decision_local.trajectory_queries import QueryConfig
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    train=helpers.read(Path(anchor['run'])/'identity.json')
    model=load_student(anchor['run'],anchor['tag'],train,precision='fp32',strip=True).requires_grad_(False).eval()
    rr=torch.load(a.refiner,map_location='cpu',weights_only=False);kind=rr['identity']['kind']
    if kind not in ('E_NONE','R_CURRENT_PRED','R_FUTURE_ABS','R_FUTURE_DIFF'):raise ValueError('Camera-only group required')
    predictor=None;records=[rr]
    if kind.startswith('R_'):
        if a.predictor is None or rr['identity']['predictor_sha256']!=helpers.digest(a.predictor):
            raise ValueError('Refiner was trained with a different predictor')
        pp=torch.load(a.predictor,map_location='cpu',weights_only=False)
        if pp['identity']['kind']!=('P_DIFF' if kind=='R_FUTURE_DIFF' else 'P_ABS'):
            raise ValueError('Matched predictor kind required')
        predictor=ConsequenceQueryHead(dim=pp['identity']['dim']).cuda();predictor.load_state_dict(pp['model'],strict=True)
        records.append(pp)
    elif a.predictor is not None:raise ValueError('E_NONE has no predictor')
    refiner=ResidualRefiner(dim=rr['identity']['dim']).cuda();refiner.load_state_dict(rr['model'],strict=True)
    labelid=helpers.read(a.labels/'identity.json');cacheid=helpers.read(a.cache/'identity.json')
    for record in records:
        if record['identity']['anchor_identity']!=anchor['identity'] or record['identity']['query_config']!=labelid['query_config']:
            raise ValueError('Different frozen foundation or query contract')
    policy=runtime.FrozenVLAConsequenceRefiner(model,refiner,predictor,mode=kind,query_config=QueryConfig(**labelid['query_config'])).eval()
    data=ForesightCurrentDataset(anchor['data'][cacheid['split']]['current_root']);lookup={r['token']:i for i,r in enumerate(data.index)}
    rows=[]
    for receipt in sorted((a.cache/'records').glob('*.json')):
        meta=helpers.read(receipt);f=load_file(str(a.cache/'chunks'/(receipt.stem+'.safetensors')));l=load_file(str(a.labels/'chunks'/(receipt.stem+'.safetensors')))
        for i,token in enumerate(meta['tokens']):
            if len(rows)>=a.samples:break
            observation={k:v for k,v in data[lookup[token]].items() if k in policy.camera_fields};observation['token']=token
            torch.cuda.synchronize();started=time.monotonic()
            online=policy.predict(observation,sampling_seed=42)
            torch.cuda.synchronize();elapsed=time.monotonic()-started
            current=ModelInputs(f['H_A'][i:i+1].cuda().float(),f['W'][i:i+1].cuda().float(),
                f['q0_encoded'][i:i+1,0].cuda(),f['q0'][i:i+1,0].cuda(),f['state'][i:i+1].cuda(),l['body'][i:i+1].cuda())
            queries=PredictionQueries(l['poses'][i:i+1,0].cuda(),torch.arange(1,current.q0.shape[1]+1,device='cuda')*.5,
                l['in_grid'][i:i+1,0].cuda(),l['kinematic_valid'][i:i+1,0].cuda())
            with torch.no_grad():
                relations=predicted_relations(predictor(current,queries)) if predictor is not None else None
                if kind=='R_CURRENT_PRED':relations=torch.cat((relations[...,:3],relations[...,1:3]),-1)
                final=decode_ego(refiner.camera_only(current,queries,relations))
            for name in ('H_A','W','q0_encoded','state'):
                torch.testing.assert_close(getattr(current,name),getattr(online['inputs'],name),rtol=0,atol=0)
            torch.testing.assert_close(queries.poses,online['queries'].poses,rtol=0,atol=0)
            torch.testing.assert_close(queries.in_grid,online['queries'].in_grid,rtol=0,atol=0)
            torch.testing.assert_close(queries.kinematic_valid,online['queries'].kinematic_valid,rtol=0,atol=0)
            torch.testing.assert_close(final,online['q_final'],rtol=1e-6,atol=1e-6)
            rejected=0
            for key in ('ego','gt_map','future_valid','relations','occupancy'):
                try:policy.predict(dict(observation,**{key:'POISON'}))
                except ValueError:rejected+=1
                else:raise AssertionError('Forbidden target input accepted')
            rows.append({'token':token,'online_cache_max_abs_m':float((final-online['q_final']).abs().max()),
                'GT_fields_rejected':rejected,'seconds':elapsed,'one_Qwen_forward':online['base_forward_count']==1})
        if len(rows)>=a.samples:break
    helpers.atomic_json(a.output,{'passed':True,'scenes':rows,'anchor_identity':anchor['identity'],'resources':resources,
        'source_sha':helpers.source_sha(ROOT),'precision':'FP32 anchor, FP16 feature roundtrip, FP32 small heads; TF32off',
        'kind':kind,'predictor_sha256':helpers.digest(a.predictor) if a.predictor else None,'refiner_sha256':helpers.digest(a.refiner),
        'limitations':'smoke heads only unless endpoint receipts show full formal training; label/cache files used solely in verification comparator, never policy.predict'})


if __name__=='__main__':main()
