"""Read-only triage of all strict KEEP / canonical-zero natural dev cases.

Canonical simulation/scoring are unmodified. Intermediate executed states are
inspected afterwards; neither scores nor teacher labels are changed.
"""
import argparse,csv,gzip,json,lzma,pickle,sys,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,read,digest,check_host,source_sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('canonical','geometry'))
    for name in ('cross-rows','metric-index','proposals','devkit','labels','anchor','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();check_host();start=time.monotonic();cpu=time.process_time()
    with gzip.open(a.cross_rows,'rt') as f:
        selected=[r for r in csv.DictReader(f) if r['distribution']=='NATURAL' and r['strict_status']=='KEEP' and float(r['score'])==0]
    a.output.mkdir(parents=True,exist_ok=True);lookup={r['token']:r for r in read(a.metric_index)};results=[]
    if a.mode=='canonical':
        from tools.local_interaction_mask_v2.score_async import initialize
        initialize(str(a.devkit))
        from navsim.common.dataclasses import Trajectory
        from navsim.evaluate.pdm_score import pdm_score,transform_trajectory,get_trajectory_as_array
        from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
        from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
        sampling=TrajectorySampling(num_poses=40,interval_length=.1)
        for row in selected:
            token=row['token'];path=a.proposals/(token+'.npz')
            with np.load(path) as f:q0=f['trajectory'].astype(np.float64)
            with np.load(a.labels/'scenes'/(token+'.npz')) as f:teacher_input=f['plan_in_physical'][0,0].astype(np.float64)
            if not np.array_equal(q0,teacher_input):raise ValueError('Scorer/checker did not use the same input plan')
            with lzma.open(lookup[token]['cache_path'],'rb') as f:cache=pickle.load(f)
            scorer=PDMScorer(sampling);simulator=PDMSimulator(sampling)
            proposal=Trajectory(q0,TrajectorySampling(num_poses=8,interval_length=.5))
            score=asdict(pdm_score(cache,proposal,sampling,simulator,scorer))
            for key,value in score.items():
                if key in row and abs(float(row[key])-float(value))>1e-12:raise ValueError('Canonical rerun changed')
            linear=get_trajectory_as_array(transform_trajectory(proposal,cache.ego_state),sampling,cache.ego_state.time_point)[:,:3]
            simulated=scorer._states[1,:,:3].copy()
            origin=cache.ego_state.rear_axle;c,s=np.cos(origin.heading),np.sin(origin.heading)
            def relative(poses):
                xy=(poses[:,:2]-[origin.x,origin.y])@np.array([[c,-s],[s,c]])
                yaw=np.arctan2(np.sin(poses[:,2]-origin.heading),np.cos(poses[:,2]-origin.heading))
                return np.column_stack((xy,yaw))
            linear,simulated=relative(linear),relative(simulated)
            np.savez_compressed(a.output/(token+'.npz'),linear=linear,simulated=simulated,times=np.arange(41)*.1)
            results.append(dict(token=token,log=row['log'],score=score,proposal_equal_to_teacher_input=True,
                simulated_linear_max_xy_m=float(np.linalg.norm(simulated[:,:2]-linear[:,:2],axis=-1).max()),
                metric_cache_sha256=digest(lookup[token]['cache_path'])))
    else:
        from starVLA.dataloader.local_evidence_scene import load_local_evidence_scene
        from starVLA.model.modules.decision_local.local_evidence import extract_local_evidence
        anchor=read(a.anchor)
        for row in selected:
            scene=load_local_evidence_scene(anchor,'dev',row);metrics={}
            with np.load(a.output/(row['token']+'.npz')) as f:
                for kind in ('linear','simulated'):
                    facts=extract_local_evidence(scene,f[kind],f['times'])
                    metrics[kind]=dict(known_road_violation=bool(facts.positive_known[:,0].any()),
                        road_margin_min_m=float(facts.values[:,0].min()),known_overlap=bool(facts.positive_known[:,2].any()),
                        annotated_distance_min_m=float(facts.values[:,1].min()),events=list(facts.events))
            results.append(dict(token=row['token'],log=row['log'],same_checker_geometry_at_canonical_states=metrics))
    atomic_json(a.output/(a.mode+'.json'),dict(source_sha=source_sha(ROOT),mode=a.mode,selected_scenes=len(selected),
        selection='All natural strict KEEP with canonical score zero; fixed before examining mechanism',results=results,
        CPU_process_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-start,
        official_scores_and_teacher_targets_modified=False))
    print(json.dumps(results))


if __name__=='__main__':main()
