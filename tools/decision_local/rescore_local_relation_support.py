"""Reclassify saved LocalEvidence query support on CPU; no model or label generation."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,check_host,digest,read,source_sha
from tools.decision_local.evaluate_relations import scene_metrics,sensitivity_population,LOCAL_SENSITIVITY,LEGACY_SENSITIVITY
from starVLA.dataloader.natural_local_dataset import NaturalLocalData
from starVLA.dataloader.recovery_plan_dataset import repair_inputs_targets


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('saved-evaluation','features','labels','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--max-seconds',type=float,required=True)
    args=parser.parse_args()
    if args.batch<=0 or args.max_seconds<=0:raise ValueError('Positive finite budget and batch required')
    resources=check_host();started=time.monotonic();torch.set_num_threads(2)
    old=read(args.saved_evaluation/'COMPLETE.json')
    if old.get('local_evidence_protocol') is not True:
        raise ValueError('Only the versioned local evidence protocol may be rescored')
    if old.get('sensitivity_protocol',LEGACY_SENSITIVITY)!=LEGACY_SENSITIVITY:
        raise ValueError('This adapter preserves and corrects the original grouping only')
    data=NaturalLocalData(args.features,args.labels)
    if old['features']!=data.feature_identity['identity'] or old['labels']!=data.label_identity['identity']:
        raise ValueError('Saved predictions and exact feature/label identity differ')
    records=[json.loads(line) for line in (args.saved_evaluation/'metrics.jsonl').read_text().splitlines()]
    normal=[r for r in records if r.get('condition','normal')=='normal']
    legacy={r['token']:r for r in normal}
    tokens=[r['token'] for r in data.index]
    if len(normal)!=len(legacy) or len(legacy)!=len(tokens) or set(legacy)!=set(tokens) or old['scenes']!=len(tokens):
        raise ValueError('Complete matched scene population required')
    with np.load(args.saved_evaluation/'predictions.npz',allow_pickle=False) as saved:
        if saved['tokens'].tolist()!=tokens:raise ValueError('Saved prediction scene order differs')
        predictions=saved['relations'].copy()
    if predictions.dtype!=np.float32 or predictions.shape[0]!=len(tokens):
        raise ValueError('Retain saved FP32 predictions exactly')
    args.output.mkdir(parents=True,exist_ok=False)
    new_rows=[];old_rows=[];max_legacy_difference=0.
    with (args.output/'metrics.jsonl').open('x') as output:
        for start in range(0,len(tokens),args.batch):
            if time.monotonic()-started>=args.max_seconds:raise TimeoutError('Partial support rescore is not complete')
            ids=list(range(start,min(len(tokens),start+args.batch)))
            batch=data.batch(ids,'cpu',evaluation=True,distribution=old['distribution'])
            _,_,queries,targets=repair_inputs_targets(batch)
            pred=torch.from_numpy(predictions[ids])
            reference=scene_metrics(pred,targets.relations,targets.relation_validity,queries.poses)
            revised=scene_metrics(pred,targets.relations,targets.relation_validity,queries.poses,local_evidence=True)
            for key,value in reference.items():
                # Only group membership changes; point/pair/classification errors
                # and ambiguity thresholds are identical on these saved tensors.
                if not key.startswith(('near_','far_')):
                    np.testing.assert_array_equal(value,revised[key])
            for j,i in enumerate(ids):
                row=data.index[i];previous=legacy[row['token']]
                if previous['log']!=row['log']:raise ValueError('Saved log identity changed')
                for key,values in reference.items():
                    wanted=previous[key];actual=float(values[j])
                    if key.endswith(('_pairs','_valid_times','_valid_points','_TP','_FP','_TN','_FN')):
                        if actual!=wanted:raise ValueError('Saved label/mask count does not replay: '+key)
                    # CPU versus original CUDA reductions, without rerunning the
                    # predictor. Record discrepancies rather than claim bit parity.
                    np.testing.assert_allclose(actual,wanted,rtol=2e-6,atol=2e-5)
                    max_legacy_difference=max(max_legacy_difference,abs(actual-wanted))
                result={**row,'condition':'normal','sensitivity_protocol':LOCAL_SENSITIVITY,
                        **{key:float(value[j]) for key,value in revised.items()}}
                output.write(json.dumps(result,allow_nan=False)+'\n')
                new_rows.append({key:value for key,value in result.items()
                    if key in ('token','log') or key.endswith('_pairs')})
                old_rows.append({**row,**{key:float(value[j]) for key,value in reference.items()
                    if key.endswith('_pairs')}})
    atomic_json(args.output/'COMPLETE.json',{
        'schema':'local_relation_saved_prediction_rescore_v1','source_sha':source_sha(ROOT),
        'tool_sha256':digest(__file__),'resources':resources,
        'scenes':len(tokens),'logs':len({r['log'] for r in data.index}),
        'scope':data.label_identity['scope'],'distribution':old['distribution'],
        'features':data.feature_identity['identity'],'labels':data.label_identity['identity'],
        'original_evaluation_sha256':digest(args.saved_evaluation/'COMPLETE.json'),
        'original_metrics_sha256':digest(args.saved_evaluation/'metrics.jsonl'),
        'predictions_sha256':digest(args.saved_evaluation/'predictions.npz'),
        'checkpoint_sha256':old['checkpoint_sha256'],
        'sensitivity_protocol':LOCAL_SENSITIVITY,
        'legacy_group_population':sensitivity_population(old_rows),
        'local_group_population':sensitivity_population(new_rows),
        'all_ungrouped_metrics_unchanged':True,
        'legacy_GPU_to_CPU_reduction_max_absolute_difference':max_legacy_difference,
        'metrics_sha256':digest(args.output/'metrics.jsonl'),
        'wall_seconds':time.monotonic()-started,'GPU_hours':0.,
        'model_executed':False,'labels_regenerated':False,
        'interpretation':'Known sensitivity survives partial support. Insensitivity requires every defining sampled fact on both queries; unknown pairs retain their separately valid local error metrics. This is not strict 4-second certification or a new trained-model result.'})


if __name__=='__main__':main()
