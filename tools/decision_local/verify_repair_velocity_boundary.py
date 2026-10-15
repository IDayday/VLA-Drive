"""Real frozen scenes: preserve legacy queries and verify corrected boundaries.

This is a read-only comparison with the preserved superseded label cache. It
does not use that cache as accepted training data or change any saved arrays.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import read,digest,atomic_json,check_host
from starVLA.dataloader.foresight_dataset import decode_ego
from starVLA.model.modules.decision_local.recovery_plans import PlanCurve,RecoveryConfig,build_plan_pool
from starVLA.model.modules.decision_local.trajectory_queries import QueryConfig,trajectory_queries


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--legacy-labels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=64);a=p.parse_args()
    resources=check_host();meta=read(a.legacy_labels/'identity.json');start=time.monotonic();cpu=time.process_time()
    if meta['split']!='train' or not 1<=a.limit<=256:raise ValueError('Bounded train-only verification')
    old_r=RecoveryConfig(**meta['recovery']);old_q=QueryConfig(**meta['query_config'])
    if old_r.velocity_boundary!='body_aligned_speed_v1' or old_q.velocity_boundary!='body_aligned_speed_v1':
        raise ValueError('Expected explicitly preserved old convention')
    new_r=RecoveryConfig(velocity_boundary='measured_vector_v2',version='actual_plan_pool_v2_current_velocity')
    new_q=QueryConfig(velocity_boundary='measured_vector_v2',kinematics_revision='heading_transition_v3_current_velocity')
    old_errors=[];new_errors=[];point_changes=[];heading_errors=[];plans=0
    for row in meta['index'][:a.limit]:
        receipt=read(a.legacy_labels/'records'/(row['token']+'.json'));path=a.legacy_labels/'scenes'/(row['token']+'.npz')
        if receipt['identity']!=meta['identity'] or digest(path)!=receipt['sha256']:raise ValueError('Changed preserved cache')
        with safe_open(receipt['source_feature_chunk'],framework='pt',device='cpu') as f:
            state=f.get_slice('state')[receipt['source_feature_offset']]
        velocity=decode_ego(state[None])[0,:2].numpy()/.5
        with np.load(path,allow_pickle=False) as f:data={k:f[k] for k in ('plan_in_physical','poses','kinematic_valid')}
        for anchor in range(len(data['plan_in_physical'])):
            q0=data['plan_in_physical'][anchor,0]
            old=build_plan_pool(q0,velocity,row['token'],old_r)
            np.testing.assert_array_equal(old['plans'],data['plan_in_physical'][anchor])
            for i,plan in enumerate(old['plans']):
                query=trajectory_queries(plan,velocity,old_q)
                np.testing.assert_array_equal(query['poses'],data['poses'][anchor,i])
                np.testing.assert_array_equal(query['kinematic_valid'],data['kinematic_valid'][anchor,i])
            new=build_plan_pool(q0,velocity,row['token'],new_r)
            np.testing.assert_array_equal(new['plans'][0],q0)
            old_curve=PlanCurve(q0,velocity);new_curve=PlanCurve(q0,velocity,velocity_boundary='measured_vector_v2')
            old_errors.append(float(np.linalg.norm(old_curve.spline(0,1)-velocity)))
            new_errors.append(float(np.linalg.norm(new_curve.spline(0,1)-velocity)))
            point_changes.append(float(np.linalg.norm(new['plans'][...,:2]-old['plans'][...,:2],axis=-1).max()))
            for plan in new['plans']:
                queries=trajectory_queries(plan,velocity,new_q)
                for stat in queries['kinematics']:
                    assert stat['initial_velocity_error_mps']==0
                    heading_errors.append(stat['initial_body_heading_error_rad'])
            plans+=len(new['plans'])
    result={'training_scenes':min(a.limit,len(meta['index'])),'plans':plans,'legacy_label_identity':meta['identity'],
            'legacy_pool_query_and_masks_exact':True,'native_q0_exact':True,
            'old_velocity_error_mps':{'p95':float(np.quantile(old_errors,.95)),'max':max(old_errors)},
            'new_velocity_error_max_mps':max(new_errors),'new_initial_query_heading_error_max_rad':max(heading_errors),
            'new_plan_point_change_max_m':max(point_changes),'new_recovery':new_r.to_dict(),'new_query':new_q.to_dict(),
            'CPU_process_seconds':time.process_time()-cpu,'elapsed_seconds':time.monotonic()-start,
            'source_code_sha256':digest(__file__),'resources':resources,
            'interpretation':'Corrected inputs/labels require a new scientific identity and common retraining even when positions differ little'}
    atomic_json(a.output,result);print(json.dumps(result))


if __name__=='__main__':main()
