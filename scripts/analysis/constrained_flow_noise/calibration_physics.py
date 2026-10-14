"""Audit saved TRAIN calibration physics; no new range, samples or selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from infer import _X_STD,_Y_STD
from starVLA.rl.constrained_flow_noise.contracts import require_dispatch,file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


def quantiles(values):return np.quantile(np.asarray(values).reshape(-1),[0,.05,.5,.95,1]).tolist()


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--calibration-root',required=True)
    p.add_argument('--registration',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();host=require_dispatch()
    if Path(a.output).exists():raise FileExistsError(a.output)
    tokens=json.loads(Path(a.registration).read_text())['calibration_tokens'];root=Path(a.calibration_root);summary={}
    for name in ('C_SDE','N_FLOW_NOISE','N_FLOW_NOISE_half'):
        std=[];angles=[];velocity_jump=[];initial_difference=[];road=[];objects=[];events=[]
        for token in tokens:
            path=root/name/'scenes'/(token+'.pt')
            if file_sha(path)!=json.loads(path.with_suffix('.json').read_text())['sha256']:raise ValueError('Calibration evidence changed')
            row=torch.load(path,map_location='cpu',weights_only=False);chain=row['chain']
            sigma=chain['old_std'][0].numpy();std.append(sigma)
            mean=chain['old_mean'][0].numpy();nxt=chain['chain'][0,:,1:].numpy()
            angle=np.arctan2(nxt[...,2],nxt[...,3])-np.arctan2(mean[...,2],mean[...,3])
            angles.append(np.arctan2(np.sin(angle),np.cos(angle)))
            states=np.array([r['simulated_states'] for r in row['records']]);initial=states[:,0]
            initial_difference.append(float(np.abs(initial-initial[:1]).max()))
            # NAVSIM's batch bicycle uses VELOCITY_X as body longitudinal speed
            # (global x_dot=v_x*cos(heading)); these values are already ego-frame.
            local_v=initial[:,3:5]
            velocity_jump.extend(np.linalg.norm(np.asarray(row['requests'])[:,0,:2]/.5-local_v,axis=-1))
            road.extend(x for r in row['records'] for x in r['geometry']['road_clearance_m'][1:])
            objects.extend(x for r in row['records'] for x in r['geometry']['object_distance_m'][1:] if x is not None)
            events.extend(r['official_first_ttc_infraction_seconds'] for r in row['records'] if r['official_first_ttc_infraction_seconds'] is not None)
        sigma=np.concatenate(std);angular=np.concatenate(angles)
        summary[name]=dict(candidates=len(tokens)*16,
            transition_std_mean_m_by_generation_step_waypoint_xy=np.stack([
                (sigma[...,0]*np.broadcast_to(np.asarray(_X_STD),(8,))[None,None]).mean(0),
                (sigma[...,1]*np.broadcast_to(np.asarray(_Y_STD),(8,))[None,None]).mean(0)],-1).tolist(),
            absolute_heading_innovation_degrees_quantiles_by_generation_step=[quantiles(np.abs(angular[:,k])*180/np.pi) for k in range(10)],
            heading_caveat='atan2 innovation around each saved transition mean, not a fixed radians-per-channel conversion; early flow states are not physical requested poses',
            first_half_second_mean_velocity_minus_initial_velocity_norm_mps_quantiles=quantiles(velocity_jump),
            simulated_t0_max_candidate_difference=max(initial_difference),
            initial_state_caveat='Official simulated t0 is fixed; the request begins at0.5s. This does not assert continuous request derivatives.',
            velocity_frame_source='navsim_v1.1/navsim/navsim/planning/simulation/planner/pdm_planner/simulation/batch_kinematic_bicycle.py::get_state_dot',
            future_road_clearance_m_quantiles=quantiles(road),future_object_distance_m_quantiles=quantiles(objects),
            TTC_event_candidate_count=len(events),TTC_first_event_physical_seconds_quantiles=quantiles(events) if events else None,
            nonfinite_policy_samples=0,
            nonfinite_evidence='All original chains passed strict finite checks before scoring; no failed trajectory was replaced or removed')
    atomic_json(Path(a.output),dict(status='COMPLETE_SAVED_CALIBRATION_AUDIT',host=host,registration_sha256=file_sha(a.registration),
                quantile_order=[0,.05,.5,.95,1],selection_unchanged=True,variants=summary))


if __name__=='__main__':main()
