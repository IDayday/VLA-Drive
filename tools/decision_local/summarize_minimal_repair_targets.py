"""Audit immutable target coverage and changes without using planning scores."""
import argparse,json,sys
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import read,atomic_json,digest
from starVLA.dataloader.foresight_dataset import decode_ego
from starVLA.model.modules.decision_local.minimal_repair_contract import CaseStatus


def summary(values):
    x=np.asarray(values,float)
    return {'n':len(x),'mean':float(x.mean()),'p50':float(np.median(x)),
            'p90':float(np.quantile(x,.9)),'max':float(x.max())} if len(x) else {'n':0}


def main():
    p=argparse.ArgumentParser();p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    meta=read(a.labels/'identity.json');states=Counter();reasons=Counter();byplan=defaultdict(Counter)
    repair_causes=Counter();checks=Counter();metrics=defaultdict(list);support=Counter();eligible_scenes=0
    cpu=wall=0.;exceptions=[]
    for row in meta['index']:
        record=read(a.labels/'records'/(row['token']+'.json'));path=a.labels/'scenes'/(row['token']+'.npz')
        if record['identity']!=meta['identity'] or digest(path)!=record['sha256']:raise ValueError('Corrupted target identity')
        cpu+=record['cpu_seconds'];wall+=record['wall_seconds']
        with np.load(path,allow_pickle=False) as data:
            eligible_scenes+=int(data['target_eligible'].any())
            gt=decode_ego(torch.from_numpy(data['ego'])).numpy()
            for c in record['cases']:
                s,i=c['anchor_index'],c['plan_index'];status=CaseStatus(c['status']).name
                states[status]+=1;reasons[c['reason']]+=1;byplan[str(i)][status]+=1
                q=data['plan_in_physical'][s,i];ok=bool(data['target_eligible'][s,i])
                if ok!=(status in ('KEEP','REPAIR')) or ok!=c['target_valid']:raise AssertionError('Fabricated target eligibility')
                observed=c['input_check'];support['input_'+str(observed['complete_annotations'])]+=1
                for k in ('road_violation','annotated_overlap','kinematic_ok','out_of_range'):
                    checks[k]+=int(observed[k])
                if not ok:continue
                target=decode_ego(torch.from_numpy(data['repair_target'][s,i])).numpy()
                xy=np.linalg.norm(target[:,:2]-q[:,:2],axis=-1)
                metrics[status+'_maximum_change_m'].append(float(xy.max()))
                metrics[status+'_target_GT_ADE_m'].append(float(np.linalg.norm(target[:,:2]-gt[:,:2],axis=-1).mean()))
                metrics[status+'_input_GT_ADE_m'].append(float(np.linalg.norm(q[:,:2]-gt[:,:2],axis=-1).mean()))
                metrics[status+'_endpoint_norm_change_m'].append(float(np.linalg.norm(target[-1,:2])-np.linalg.norm(q[-1,:2])))
                if status=='KEEP' and not np.allclose(target,q,rtol=0,atol=1e-5):raise AssertionError('KEEP not identical after encoding round trip')
                if status=='REPAIR':
                    if not c['target_check']['feasible']:raise AssertionError('Unvalidated repair')
                    for k in ('road_violation','annotated_overlap'):
                        repair_causes[k]+=int(observed[k])
                    repair_causes['kinematic_violation']+=int(not observed['kinematic_ok'])
                    checks['repair_new_full_stop']+=int(np.linalg.norm(target[-1,:2])<1<=np.linalg.norm(q[-1,:2]))
                valid=data['common_valid'][s,i]
                delta=np.abs(data['relations'][s,i,2]-data['relations'][s,i,1])
                for ch,name in enumerate(('road','clearance','overlap')):
                    mask=valid[...,ch];support[name+'_valid_points']+=int(mask.sum())
                    if mask.any():metrics['eligible_CV_FUT_'+name+'_absolute_difference'].extend(delta[...,ch][mask].tolist())
                support['eligible_CV_FUT_overlap_changed_plans']+=int(((delta[...,2]>.5)&valid[...,2]).any())
                support['eligible_CV_FUT_clearance_changed_ge_05m_plans']+=int(((delta[...,1]>=.5)&valid[...,1]).any())
    count=sum(states.values())
    result={'label_identity':meta['identity'],'scenes':len(meta['index']),'plans':count,'states':dict(states),
        'target_eligible_fraction':(states['KEEP']+states['REPAIR'])/max(count,1),
        'scenes_with_any_eligible_plan':eligible_scenes,'by_plan':{k:dict(v) for k,v in byplan.items()},
        'reasons':dict(reasons),'input_checks':dict(checks),'repair_causes':dict(repair_causes),
        'metrics':{k:summary(v) for k,v in metrics.items()},'relation_support':dict(support),
        'CPU_process_seconds':cpu,'worker_wall_seconds':wall,
        'label_scope':'annotated planar geometry with unknown support retained; not full free-space/safety GT',
        'planning_scores_used_for_target_selection':False,'decoder_roundtrip_atol_m':1e-5}
    atomic_json(a.output,result);print(json.dumps({k:result[k] for k in ('scenes','plans','states','target_eligible_fraction','repair_causes','relation_support')}))


if __name__=='__main__':main()
