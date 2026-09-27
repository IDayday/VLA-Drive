"""Real online BEV → joint graph → original DiT connectivity, no optimizer update."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from tools.structured_world.runtime import load_baseline,load_dataset,load_world_batch,seed_all
from tools.structured_world_v1p1.budget import start,record
from starVLA.model.modules.joint_world.policy import JointTrajectoryPolicy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['world-checkpoint','graph-checkpoint','bev-checkpoint','bev-config','observations','output','ledger','run-id']:
        p.add_argument('--'+key,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    world=torch.load(a.world_checkpoint,map_location='cpu',weights_only=False)
    graph=torch.load(a.graph_checkpoint,map_location='cpu',weights_only=False)
    bev=torch.load(a.bev_checkpoint,map_location='cpu',weights_only=False)
    identity={'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'arguments':vars(a),
              'checkpoint_hashes':{k:hashlib.sha256(Path(getattr(a,k)).read_bytes()).hexdigest() for k in ['world_checkpoint','graph_checkpoint','bev_checkpoint']}}
    start(a.ledger,a.run_id,0,identity)
    try:
        args=world['identity']['arguments'];seed_all(42);agent=load_baseline(args['checkpoint'],args['vlm']);agent.model.requires_grad_(False)
        policy=JointTrajectoryPolicy(agent.model,world['identity']['config'],graph['identity']['config'],json.loads(Path(a.bev_config).read_text())).cuda().eval().requires_grad_(False)
        for name in ['reader','heads']:getattr(policy.world,name).load_state_dict(world[name],strict=True)
        policy.graph.load_state_dict(graph['model'],strict=True)
        for prefix,module in [('encoder',policy.bev_encoder),('fusion',policy.bev_fusion),('interaction',policy.interaction_head)]:
            module.load_state_dict({k[len(prefix)+1:]:v for k,v in bev['model'].items() if k.startswith(prefix+'.')},strict=True)
        raw=load_dataset(agent,args['manifest'],args['data_root'],1)[0]
        example={k:raw[k] for k in ['image','lang','state','token']}
        inputs,_=load_world_batch([example],a.observations,load_targets=False)
        with torch.no_grad():
            seed_all(51);native=agent.model.predict_action_infer_1d([example])['normalized_actions']
            start_time=time.perf_counter();seed_all(51);zero=policy.predict_action([example],inputs);torch.cuda.synchronize();latency=time.perf_counter()-start_time
            # Both real BEV/provider path and joint model execute at zero gates.
            assert np.array_equal(native,zero['normalized_actions'])
            policy.bev_fusion.gate.fill_(.1);policy.adapter.gate.fill_(.1)
            seed_all(51);active=policy.predict_action([example],inputs)
            dirty=dict(example,WorldTargets={'future':float('nan')},future_file='/absent/file',action=np.full((8,4),1e9))
            seed_all(51);poisoned=policy.predict_action([dirty],inputs)
            hook=policy.bev_encoder.register_forward_pre_hook(lambda m,args:(torch.zeros_like(args[0]),*args[1:]))
            try:seed_all(51);blank=policy.predict_action([example],inputs)
            finally:hook.remove()
        result={'identity':identity,'token':example['token'],'optimizer_updates':0,'gate0_native_exact':bool(np.array_equal(native,zero['normalized_actions'])),
                'poisoned_action_exact':bool(np.array_equal(active['normalized_actions'],poisoned['normalized_actions'])),
                'poisoned_joint_exact':bool(torch.equal(active['joint_trajectories_xy'],poisoned['joint_trajectories_xy'])),
                'blank_bev_joint_max_delta_m':float((active['joint_trajectories_xy']-blank['joint_trajectories_xy']).abs().max()),
                'blank_bev_action_max_delta_normalized':float(np.abs(active['normalized_actions']-blank['normalized_actions']).max()),
                'full_pipeline_latency_seconds':latency,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
                'provider_frozen':not any(p.requires_grad for p in policy.bev_provider.parameters()),
                'limitation':'Single scene connectivity with manually opened gates; not trained planning benefit. Targets not loaded.'}
        assert result['poisoned_action_exact'] and result['poisoned_joint_exact']
        assert result['blank_bev_joint_max_delta_m']>0 and result['blank_bev_action_max_delta_normalized']>0
        result['status']='PASS';(out/'BEV_PIPELINE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
        record(a.ledger,a.run_id,0,'complete')
    except BaseException:record(a.ledger,a.run_id,0,'failed');raise


if __name__=='__main__':main()
