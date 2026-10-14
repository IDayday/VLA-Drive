"""Compare predicted future relations to current/CV on identical GT support.

These are relation errors, not planning scores. Current/CV GT are privileged
references. Query positions and physical times are identical for every source.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,digest,read,source_sha
from starVLA.dataloader.decision_local import FrozenConsequenceData


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('features','labels','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--prediction',action='append',required=True,help='Name=completed evaluate_relations directory')
    a=p.parse_args();torch.set_num_threads(1)
    if a.output.exists():raise FileExistsError('Preserve existing relation audit')
    data=FrozenConsequenceData(a.features,a.labels,reader='mmap_tensors');tokens=[x['token'] for x in data.index]
    models={};identities={}
    for item in a.prediction:
        name,path=item.split('=',1);path=Path(path);done=read(path/'COMPLETE.json')
        if done['features']!=data.feature_identity['identity'] or done['labels']!=data.label_identity['identity']:
            raise ValueError('Identical query/label population required')
        with np.load(path/'predictions.npz') as values:
            if list(values['tokens'])!=tokens:raise ValueError('Prediction scene order changed')
            pred=values['relations'].copy()
        if pred.shape[0]!=len(tokens) or pred.shape[-1]!=5 or not np.isfinite(pred).all():raise ValueError('Invalid relation field')
        if name in models:raise ValueError('Duplicate model')
        models[name]=pred;identities[name]={'checkpoint_sha256':done['checkpoint_sha256'],'array_sha256':digest(path/'predictions.npz')}
    totals={};per_scene=[]
    for start in range(0,len(tokens),32):
        stop=min(start+32,len(tokens));batch=data.batch(list(range(start,stop)),'cpu',evaluation=True)
        rel=batch['relations'].numpy();truth=rel[:,2,...,1:];cv=rel[:,1,...,1:]
        mask=batch['common_valid'][...,1:].numpy();kin=batch['kinematic_valid'].numpy().copy();kin[:,0]=True
        mask&=kin[...,None,None]
        sources={'CURRENT_GT':rel[:,0,...,1:],'CV_GT':cv}
        for name,pred in models.items():
            sources[name+'_current_hold']=pred[start:stop,...,1:3]
            sources[name+'_future']=pred[start:stop,...,3:5]
        B,K,T,C=truth.shape
        q0=np.zeros((1,K,1,1),bool);q0[:,0]=True
        different=np.stack((np.abs(truth[...,0]-cv[...,0])>=.5,(truth[...,1]>.5)!=(cv[...,1]>.5)),-1)
        groups={'all_queries':mask,'q0':mask&q0,'differs_from_CV':mask&different,'q0_differs_from_CV':mask&different&q0}
        for name,pred in sources.items():
            for group,valid in groups.items():
                entry=totals.setdefault(name,{}).setdefault(group,{})
                for t in [None,*range(T)]:
                    key='all_times' if t is None else f'{(t+1)*.5:.1f}s'
                    dst=entry.setdefault(key,dict(clearance_valid=0,clearance_absolute_error_sum=0.,overlap_valid=0,
                        overlap_score_squared_error_sum=0.,TP=0,FN=0,FP=0,TN=0))
                    m=valid if t is None else valid[:,:,t:t+1]
                    v=pred if t is None else pred[:,:,t:t+1]
                    y=truth if t is None else truth[:,:,t:t+1]
                    dst['clearance_valid']+=int(m[...,0].sum())
                    dst['clearance_absolute_error_sum']+=float(np.where(m[...,0],np.abs(v[...,0]-y[...,0]),0).sum(dtype=np.float64))
                    dst['overlap_valid']+=int(m[...,1].sum())
                    dst['overlap_score_squared_error_sum']+=float(np.where(m[...,1],(v[...,1]-y[...,1])**2,0).sum(dtype=np.float64))
                    positive=v[...,1]>=.5;target=y[...,1]>.5
                    for field,selected in [('TP',positive&target),('FN',~positive&target),('FP',positive&~target),('TN',~positive&~target)]:
                        dst[field]+=int((selected&m[...,1]).sum())
                m=valid
                error=np.where(m[...,0],np.abs(pred[...,0]-truth[...,0]),0).sum((1,2))
                counts=m[...,0].sum((1,2))
                for i in range(B):per_scene.append({'token':tokens[start+i],'log':data.index[start+i]['log'],
                    'source':name,'group':group,'clearance_error_sum':float(error[i]),'clearance_valid':int(counts[i])})
    for source in totals.values():
        for group in source.values():
            for entry in group.values():
                entry['clearance_MAE_m']=entry['clearance_absolute_error_sum']/entry['clearance_valid'] if entry['clearance_valid'] else None
                entry['overlap_score_squared_error']=entry['overlap_score_squared_error_sum']/entry['overlap_valid'] if entry['overlap_valid'] else None
                entry['overlap_precision']=entry['TP']/(entry['TP']+entry['FP']) if entry['TP']+entry['FP'] else None
                entry['overlap_recall']=entry['TP']/(entry['TP']+entry['FN']) if entry['TP']+entry['FN'] else None
    a.output.mkdir(parents=True)
    import json
    with (a.output/'scenes.jsonl').open('w') as stream:
        for row in per_scene:stream.write(json.dumps(row)+'\n')
    atomic_json(a.output/'summary.json',{'source_sha':source_sha(ROOT),'tool_sha256':digest(__file__),
        'scenes':len(tokens),'features':data.feature_identity['identity'],'labels':data.label_identity['identity'],
        'models':identities,'results':totals,'support':'Three-source GT intersection, same kinematic eligibility; all original scenes retained',
        'interpretation':'Privileged current/CV references versus camera-only readout; differences are relation-error diagnostics, never a planning oracle upper bound or calibrated collision probability',
        'different_from_CV_rule':'Per-channel: distance changes >=0.5m; binary overlap changes state; fixed physical threshold, no PDMS selection',
        'statistics':'Descriptive pooled valid-query ratios; per-scene sums retained for log-cluster inference, no scene-times-query pseudo-replication'})


if __name__=='__main__':main()
