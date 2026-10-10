"""Bounded real stage cost probe, two scenes and no training/full candidate bank."""
import argparse
from dataclasses import asdict
from pathlib import Path
import time
import torch
from omegaconf import OmegaConf
from iqe.config import load_config
from iqe.pipeline import Pipeline,stack_features
from iqe.io import atomic_json,atomic_torch,read_json,file_hash,digest
from iqe.contracts import CandidateRecord
from iqe.query_base import bind_source,QueryActionBridge
from iqe.data.sources import load_target
from iqe.data.candidate_bank import save_trajectory
from iqe.resources import qualify


def main():
    a=argparse.ArgumentParser(__doc__);a.add_argument('--config',required=True);args=a.parse_args()
    qualify('cpu');torch.set_num_threads(2)
    p=Pipeline(load_config(args.config),mode='smoke',max_samples=32,device='cpu');p.use_locked_base()
    c=p.contract;bind_source(c['source_root'],c['source_commit'])
    state=torch.load(c['query_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
    bridge=QueryActionBridge(OmegaConf.create(c['source_config']),c['query_architecture'])
    bridge.expert.load_state_dict({k.removeprefix('action_model.expert.'):v for k,v in state['model'].items() if k.startswith('action_model.expert.')},strict=True)
    bridge.expert.eval();del state
    scenes=p.scenes(['stage_val'])[:2];costs={}
    t=time.perf_counter();features=[p.cached(s) for s in scenes];costs['feature_read_seconds']=time.perf_counter()-t
    t=time.perf_counter()
    with torch.no_grad():outputs=[bridge.expert(f) for f in features]
    costs['candidate_generation_seconds']=time.perf_counter()-t
    scores=[]
    for kind in ['candidate','GT']:
        times=[]
        for s,out in zip(scenes,outputs):
            tau=out.physical[0] if kind=='candidate' else p.trajectory_contract.physical(load_target(s,p.trajectory_contract))
            ref=p.root/'cost_probe'/(kind+'_'+s.scene_id+'.npz');h=save_trajectory(ref,tau.numpy())
            candidate=CandidateRecord(1,s.scene_id,'expert_0' if kind=='candidate' else 'ground_truth',
                c['query_checkpoint_hash'] if kind=='candidate' else s.target_id,digest(c),h,str(ref),p.trajectory_contract.raw_representation,
                'ego_relative_rear_axle',p.trajectory_contract.horizon,p.trajectory_contract.dt,True,c['iqe_code_hash'],p.config['config_hash'],'bounded_cost_probe')
            t=time.perf_counter();score=p.backend().score(candidate,s);times.append(time.perf_counter()-t)
            assert score.score_valid
            scores.append(asdict(score))
        costs[kind+'_official_seconds']=times
    checkpoint=p.round_root(1)/'expert/step_000002.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    probe=p.root/'cost_probe/checkpoint_write.pt';t=time.perf_counter();atomic_torch(probe,state);checksum=file_hash(probe)
    costs['checkpoint_write_and_checksum_seconds']=time.perf_counter()-t;costs['checkpoint_bytes']=probe.stat().st_size
    arrays=p.bank_arrays(1,'stage_val');network=p.selector_module(1)
    network.load_state_dict(torch.load(p.round_root(1)/'scorer/best.pt',weights_only=True),strict=True)
    t=time.perf_counter();p.prediction_arrays(network,arrays);costs['scorer_validation_prediction_seconds']=time.perf_counter()-t
    costs['scorer_validation_scenes']=len(arrays['scenes'])
    result={'status':'PASS','device':'CPU','threads':2,'scenes':2,'costs':costs,'scores':scores,
            'checkpoint_probe_checksum':checksum,'full_bank_started':False,'image_to_trajectory':False,
            'candidate_path':'actual Query Base head on actual frozen image FeatureBundles'}
    atomic_json(p.root/'COST_PROBE.json',result,immutable=True)
    print(costs)


if __name__=='__main__':main()
