"""Official singleton, ordering and distractor audit on saved real TRAIN requests."""
import argparse
from pathlib import Path
import numpy as np
import torch
from starVLA.rl.d44_native_rl.config import defaults
from starVLA.rl.d44_native_rl.data import DataStream
from starVLA.rl.simwam_recipe.reward_readouts import RewardService
from starVLA.rl.safe_flow_improvement import v1_reward
from starVLA.rl.qwen_world_compare.contracts import require_dispatch
from starVLA.rl.flow_grpo.transactions import atomic_json

TIMINGS={'rollout_seconds','scoring_seconds','reference_seconds','scoring_cpu_seconds','scene_scoring_cpu_seconds','scene_scoring_wall_seconds','geometry_seconds','score_wait_seconds','collection_execution'}
def exact(a,b,path='root'):
    if isinstance(a,dict):
        keys=set(a)-TIMINGS-{'readouts'};assert keys==set(b)-TIMINGS-{'readouts'},path
        for key in keys:exact(a[key],b[key],path+'/'+str(key))
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(v,w)in enumerate(zip(a,b)):exact(v,w,path+'/'+str(i))
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b,err_msg=path)
    elif torch.is_tensor(a):assert torch.equal(a,b),path
    elif isinstance(a,float)and np.isnan(a):assert np.isnan(b),path
    else:assert a==b,path


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--behavior',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    require_dispatch();torch.set_num_threads(2)
    stream=DataStream(defaults('D0'),64);saved=torch.load(a.behavior,map_location='cpu',weights_only=False)
    service=RewardService(Path.cwd(),workers=1,cache_roots=stream.data.manifest['canonical_metric_cache_roots']);calls=0;checks=[]
    try:
        for r in saved['scenes'][:2]:
            assert r['token']in stream.tokens
            job=(r['token'],r['log'],np.array([v['requested_trajectory']for v in r['records']]))
            batched=service.score(job);calls+=len(batched)
            serial=service.pool.submit(v1_reward.score_scene,job).result();calls+=len(serial)
            exact(serial,batched)
            reverse=service.score((job[0],job[1],job[2][::-1].copy()));calls+=len(reverse)
            exact(batched,reverse[::-1])
            proposal=job[2][[0,1,0,2]].copy();proposal[1,:,0]*=3;proposal[3,:,1]+=12
            distractor=service.score((job[0],job[1],proposal));calls+=len(distractor)
            exact(batched[0],distractor[0]);exact(distractor[0],distractor[2])
            checks.append(dict(token=r['token'],singleton_all_native_fields=True,reorder=True,other_candidates_changed=True,duplicate=True))
        atomic_json(a.output,dict(status='PASS',checks=checks,scorer_candidate_calls=calls,protocol=service.protocol,
            source_behavior=a.behavior,new_actor_candidates=0,training_updates=0))
    finally:service.close()


if __name__=='__main__':main()
