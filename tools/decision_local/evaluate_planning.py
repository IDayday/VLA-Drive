"""Atomic trajectory export with explicit oracle/camera-only entry separation.

Scoring is delegated to the existing locked canonical score_navsim adapter.
This exports the exact frozen-cache path previously checked against real images.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,check_host,digest,identity,read,source_sha
from tools.decision_local.train_small_model import build_model,load_predictor
from starVLA.model.modules.decision_local.consequence_contract import ModelInputs,PredictionQueries,predicted_relations,OracleRelations
from starVLA.dataloader.foresight_dataset import decode_ego


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('features','queries','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--predictor',type=Path);p.add_argument('--oracle-labels',type=Path)
    p.add_argument('--mode',choices=('camera_only','oracle_probe'),required=True)
    p.add_argument('--batch',type=int,default=1);p.add_argument('--max-seconds',type=float,required=True)
    a=p.parse_args();resources=check_host();started=time.monotonic()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    fmeta,qmeta=read(a.features/'identity.json'),read(a.queries/'identity.json')
    if qmeta['features']!=fmeta['identity'] or qmeta['contains_GT']:raise ValueError('Current-only query bank required')
    for path in (a.features,a.queries):
        if not (path/'COMPLETE.json').exists():raise ValueError('Complete immutable input population required')
    s=torch.load(a.checkpoint,map_location='cpu',weights_only=False);kind=s['identity']['kind']
    if s['identity'].get('anchor_identity')!=fmeta['anchor_identity']:raise ValueError('Wrong frozen base')
    if s['identity'].get('query_config')!=qmeta['query_config']:raise ValueError('Different registered queries')
    if fmeta['split'] in ('dev','navtest') and a.batch!=1:raise ValueError('Canonical head inference matches online batch1')
    if kind.startswith('P_'):raise ValueError('A relation head is not an execution policy')
    camera=kind in ('E_NONE','R_CURRENT_PRED','R_FUTURE_ABS','R_FUTURE_DIFF')
    if (a.mode=='camera_only')!=camera or (camera and a.oracle_labels is not None):raise ValueError('GT information cannot enter camera-only export')
    if fmeta['split']=='navtest' and not (a.output.parent/'FINAL_METHOD_LOCK.json').exists():raise ValueError('Final method/endpoint must be locked before new Navtest')
    if not camera and a.oracle_labels is None:raise ValueError('Oracle probe requires explicit GT relation labels')
    if a.oracle_labels and read(a.oracle_labels/'identity.json')['anchor_cache_identity']!=fmeta['identity']:
        raise ValueError('Oracle label population mismatch')
    model=build_model(kind,s['identity']['seed'],s['identity']['dim']).cuda().eval();model.load_state_dict(s['model'],strict=True)
    predictor=None
    if kind.startswith('R_'):
        if not a.predictor:raise ValueError('Matching frozen predictor required')
        if digest(a.predictor)!=s['identity']['predictor_sha256']:raise ValueError('Refiner predictor differs from training')
        predictor,_=load_predictor(a.predictor,'P_DIFF' if kind=='R_FUTURE_DIFF' else 'P_ABS')
    contract={'schema':'decision_local_single_trajectory_export_v1','kind':kind,'mode':a.mode,'features':fmeta['identity'],
        'queries':qmeta['identity'],'checkpoint_sha256':digest(a.checkpoint),'predictor_sha256':digest(a.predictor) if a.predictor else None,
        'oracle_labels':read(a.oracle_labels/'identity.json')['identity'] if a.oracle_labels else None,
        'source_sha':source_sha(ROOT),'writer_sha256':digest(__file__),'sampling_seed':42,'FP32_head':True,'TF32':False,
        'base_identity':fmeta['anchor_identity'],'base_sampler':'original independent scene-bound10-step single proposal',
        'head_inference_batch':a.batch,'main_trajectory':'q_final; never choose higher-scoring q0/query'}
    ident=identity(contract);a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'identity.json').exists() and read(a.output/'identity.json')['identity']!=ident:raise ValueError('Foreign export')
    atomic_json(a.output/'identity.json',{'identity':ident,**contract})
    for k in ('predictions','records'):(a.output/k).mkdir(exist_ok=True)
    done=0
    with torch.no_grad():
        for receipt in sorted((a.features/'records').glob('*.json')):
            if time.monotonic()-started>=a.max_seconds:raise TimeoutError('Bounded inference time exhausted')
            fm=read(receipt);qm=read(a.queries/'records'/receipt.name)
            if fm['tokens']!=qm['tokens']:raise ValueError('Mismatched input/query scene order')
            fp=a.features/'chunks'/(receipt.stem+'.safetensors');qp=a.queries/'chunks'/(receipt.stem+'.safetensors')
            if digest(fp)!=fm['sha256'] or digest(qp)!=qm['sha256']:raise ValueError('Input chunk changed')
            f,q=load_file(str(fp)),load_file(str(qp));target=None
            if a.oracle_labels:
                lp=a.oracle_labels/'chunks'/(receipt.stem+'.safetensors');lm=read(a.oracle_labels/'records'/receipt.name)
                if lm['tokens']!=fm['tokens'] or lm['sha256']!=digest(lp):raise ValueError('Oracle labels changed or reordered')
                target=load_file(str(lp))
            for start in range(0,len(fm['tokens']),a.batch):
                stop=min(start+a.batch,len(fm['tokens']));sl=slice(start,stop)
                current=ModelInputs(f['H_A'][sl].cuda().float(),f['W'][sl].cuda().float(),f['q0_encoded'][sl,0].cuda(),
                    f['q0'][sl,0].cuda(),f['state'][sl].cuda(),q['body'][sl].cuda())
                T=current.q0.shape[1]
                query=PredictionQueries(q['poses'][sl,0].cuda(),torch.arange(1,T+1,device='cuda')*.5,q['in_grid'][sl,0].cuda(),q['kinematic_valid'][sl,0].cuda())
                if not camera:
                    source={'E_CURRENT_GT':0,'E_CV_GT':1,'E_FUTURE_GT':2,'E_MASK_ONLY':2}[kind]
                    values=torch.cat((target['relations'][sl,0,0],target['relations'][sl,0,source,...,1:]),-1).cuda()
                    support=torch.cat((target['common_valid'][sl,0],target['common_valid'][sl,0,...,1:]),-1).cuda()
                    kin=query.kinematic_valid.clone();kin[:,0]=True;support&=kin[...,None,None]
                    if kind=='E_MASK_ONLY':values.zero_()
                    output=model.oracle_probe(current,query,OracleRelations(values,support,{'E_CURRENT_GT':'CURRENT','E_CV_GT':'CV','E_FUTURE_GT':'FUTURE','E_MASK_ONLY':'MASK_ONLY'}[kind]))
                else:
                    rel=predicted_relations(predictor(current,query)) if predictor is not None else None
                    if kind=='R_CURRENT_PRED':rel=torch.cat((rel[...,:3],rel[...,1:3]),-1)
                    output=model.camera_only(current,query,rel)
                final=decode_ego(output).cpu().numpy();q0=current.q0.cpu().numpy()
                if not np.isfinite(final).all():raise FloatingPointError('Invalid predicted trajectory; no scene dropped')
                for j,token in enumerate(fm['tokens'][start:stop]):
                    dest=a.output/'predictions'/(token+'.npz');temp=dest.with_suffix('.tmp')
                    with temp.open('wb') as stream:np.savez_compressed(stream,q0=q0[j],q_final=final[j])
                    temp.replace(dest);atomic_json(a.output/'records'/(token+'.json'),{'identity':ident,'token':token,'sha256':digest(dest)})
                    done+=1
    if done!=len(fmeta['index']):raise ValueError('Incomplete prediction population')
    atomic_json(a.output/'COMPLETE.json',{'identity':ident,'scenes':done,'seconds':time.monotonic()-started,
        'GPU_hours':(time.monotonic()-started)/3600.,'resources':resources,'cache_based':True,'online_latency_included':False})


if __name__=='__main__':main()
