"""Read-only relation accuracy grouped by current-camera frustum support.

Frustum membership, label validity and kinematic/query support remain distinct.
This is neither an occlusion test nor an execution-time GT mask.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,digest,read,source_sha
from starVLA.dataloader.decision_local import FrozenConsequenceData,inputs_queries_targets
from starVLA.model.modules.decision_local.consequence_contract import CHANNELS,CONTINUOUS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('features','labels','frustum','predictions','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    if a.output.exists():raise FileExistsError('Immutable support analysis')
    data=FrozenConsequenceData(a.features,a.labels,reader='mmap_tensors')
    fm=read(a.frustum/'COMPLETE.json')
    if fm['features']!=data.feature_identity['identity'] or digest(a.frustum/'frustum.npz')!=fm['array_sha256']:
        raise ValueError('Different current-query population')
    with np.load(a.frustum/'frustum.npz') as f:
        support={str(t):v for t,v in zip(f['tokens'],f['fraction'])}
    predictions={}
    if a.predictions.is_file():
        with np.load(a.predictions) as f:
            for token,value in zip(f['tokens'],f['relations']):predictions[str(token)]=value
        pred_id=digest(a.predictions)
    else:
        pm=read(a.predictions/'identity.json');done=read(a.predictions/'COMPLETE.json')
        if pm['identity']!=done['identity']:raise ValueError('Complete G diagnostic required')
        for receipt in sorted((a.predictions/'records').glob('*.json')):
            r=read(receipt);path=a.predictions/'chunks'/(receipt.stem+'.safetensors')
            if digest(path)!=r['sha256']:raise ValueError('G field export changed')
            arr=load_file(str(path))['relations'].numpy()
            for token,value in zip(r['tokens'],arr):
                if token in predictions:raise ValueError('Duplicate prediction scene')
                predictions[token]=value
        pred_id=pm['identity']
    tokens={r['token'] for r in data.index}
    if set(support)!=tokens or set(predictions)!=tokens:raise ValueError('Complete matched population required')
    totals={}
    for start in range(0,len(data.index),32):
        ids=list(range(start,min(start+32,len(data.index))))
        batch=data.batch(ids,'cpu',evaluation=True);_,_,target=inputs_queries_targets(batch)
        y=target.relations.numpy();valid=target.valid.numpy()
        pred=np.stack([predictions[data.index[i]['token']] for i in ids])
        frustum=np.stack([support[data.index[i]['token']] for i in ids])
        groups={'all_queries':np.ones_like(frustum,bool),'full_body_in_frustum':frustum==1,
                'partial_body_in_frustum':(frustum>0)&(frustum<1),'no_body_in_frustum':frustum==0}
        for group,selected in groups.items():
            entries=totals.setdefault(group,{})
            for c,name in enumerate(CHANNELS):
                entry=entries.setdefault(name,dict(queries=0,valid_queries=0,absolute_error_sum=0.,
                    pair_valid_times=0,pair_error_sum=0.,TP=0,FN=0,FP=0,TN=0))
                m=selected&valid[...,c]
                entry['queries']+=int(selected.sum());entry['valid_queries']+=int(m.sum())
                entry['absolute_error_sum']+=float(np.where(m,np.abs(pred[...,c]-y[...,c]),0.).sum(dtype=np.float64))
                if c in CONTINUOUS:
                    pair=m[:,1:]&m[:,:1]
                    error=np.abs((pred[:,1:,:,c]-pred[:,:1,:,c])-(y[:,1:,:,c]-y[:,:1,:,c]))
                    entry['pair_valid_times']+=int(pair.sum());entry['pair_error_sum']+=float(np.where(pair,error,0.).sum(dtype=np.float64))
                else:
                    pp=pred[...,c]>=.5;yy=y[...,c]>.5
                    for key,mask in [('TP',pp&yy),('FN',~pp&yy),('FP',pp&~yy),('TN',~pp&~yy)]:
                        entry[key]+=int((mask&m).sum())
    for entries in totals.values():
        for name,e in entries.items():
            e['label_coverage']=e['valid_queries']/e['queries'] if e['queries'] else None
            e['MAE']=e['absolute_error_sum']/e['valid_queries'] if e['valid_queries'] else None
            e['pair_MAE']=e['pair_error_sum']/e['pair_valid_times'] if e['pair_valid_times'] else None
            if name.endswith('overlap'):
                e['precision']=e['TP']/(e['TP']+e['FP']) if e['TP']+e['FP'] else None
                e['recall']=e['TP']/(e['TP']+e['FN']) if e['TP']+e['FN'] else None
    atomic_json(a.output,{'source_sha':source_sha(ROOT),'writer_sha256':digest(__file__),'scenes':len(tokens),
        'features':data.feature_identity['identity'],'labels':data.label_identity['identity'],
        'frustum_sha256':digest(a.frustum/'COMPLETE.json'),'predictions':pred_id,'groups':totals,
        'pair_group_rule':'both query endpoints must have the same indicated support category',
        'limitation':'Ground-plane geometric frustum proxy only; not actual occlusion visibility. No policy inputs or label masks changed.'})


if __name__=='__main__':main()
