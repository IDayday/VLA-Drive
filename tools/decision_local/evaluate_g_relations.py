"""Compare already-exported G fields on the same S0 queries and GT support."""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,digest,read,source_sha
from tools.decision_local.evaluate_relations import scene_metrics
from starVLA.dataloader.decision_local import FrozenConsequenceData,inputs_queries_targets
from starVLA.model.modules.decision_local.consequence_contract import CHANNELS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('features','labels','queries','predictions','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);data=FrozenConsequenceData(a.features,a.labels)
    meta=read(a.predictions/'identity.json');done=read(a.predictions/'COMPLETE.json');qm=read(a.queries/'identity.json')
    if done['identity']!=meta['identity'] or meta['query_identity']!=qm['identity'] or qm['features']!=data.feature_identity['identity']:
        raise ValueError('Mismatched immutable query/feature population')
    lookup={r['token']:i for i,r in enumerate(data.index)};results={};seen=set();rows=[]
    for receipt in sorted((a.predictions/'records').glob('*.json')):
        r=read(receipt);path=a.predictions/'chunks'/(receipt.stem+'.safetensors')
        if r['identity']!=meta['identity'] or digest(path)!=r['sha256']:raise ValueError('Changed G predictions')
        if seen&set(r['tokens']):raise ValueError('Duplicate diagnostic scene')
        seen.update(r['tokens']);ids=[lookup[t] for t in r['tokens']]
        batch=data.batch(ids,'cpu',evaluation=True);_,queries,targets=inputs_queries_targets(batch)
        pred=load_file(str(path))['relations']
        metrics=scene_metrics(pred,targets.relations,targets.valid,queries.poses)
        for k,v in metrics.items():results.setdefault(k,[]).append(v)
        for i,token in enumerate(r['tokens']):rows.append({'token':token,'log':data.index[ids[i]]['log'],**{k:float(v[i]) for k,v in metrics.items()}})
    if len(seen)!=done['scenes']:raise ValueError('Incomplete diagnostic export')
    values={k:np.concatenate(v) for k,v in results.items()};summary={}
    for c,name in enumerate(CHANNELS):
        count=values[name+'_valid_points'];error=values[name+'_abs_error_sum'];valid=count>0
        entry={'coverage':float(values[name+'_coverage'].mean()),'supported_scenes':int(valid.sum()),
            'pooled_MAE':float(error.sum()/count.sum()) if count.sum() else None,
            'supported_scene_MAE':float((error[valid]/count[valid]).mean()) if valid.any() else None}
        if name+'_pair_valid_times' in values:
            den=values[name+'_pair_valid_times'].sum();direction=values[name+'_direction_valid_pairs'].sum()
            entry.update(pair_MAE=float(values[name+'_pair_abs_error_sum'].sum()/den) if den else None,
                relative_direction_accuracy=float(values[name+'_direction_correct_pairs'].sum()/direction) if direction else None,
                direction_valid_pairs=int(direction))
        else:
            tp,fn,fp,tn=[float(values[name+'_'+k].sum()) for k in ('TP','FN','FP','TN')]
            entry.update(TP=tp,FN=fn,FP=fp,TN=tn,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn))
        summary[name]=entry
    quadrants={}
    for group in ('near_sensitive','near_insensitive','far_sensitive','far_insensitive'):
        q={'pairs':int(values[group+'_pairs'].sum())}
        for name in (CHANNELS[c] for c in (0,1,3)):
            den=values[group+'_'+name+'_valid_times'].sum()
            q[name+'_pair_MAE']=float(values[group+'_'+name+'_pair_abs_error_sum'].sum()/den) if den else None
        quadrants[group]=q
    a.output.mkdir(parents=True,exist_ok=False)
    import json
    with (a.output/'scenes.jsonl').open('w') as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    atomic_json(a.output/'summary.json',{'source_sha':source_sha(ROOT),'tool_sha256':digest(__file__),'model_export':meta,
        'labels':data.label_identity['identity'],'scenes':len(seen),'full_dev':len(seen)==1696,
        'metrics':summary,'locality_sensitivity':quadrants,
        'limitations':['G clearance uses fixed0.5 occupancy threshold plus EDT, not an original learned G distance head',
            'Fixed query locality is relative to the shared S0 plan; each G future field retains its own normal q0 conditioning',
            'Common-query evidence does not by itself isolate the causal effect of the historical G sampling rule']})


if __name__=='__main__':main()
