"""Bounded real-chain probe of CUDA graphs; never grants training acceptance."""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch
import torch
from starVLA.rl.flow_grpo.config import resolve_config
from starVLA.rl.flow_grpo.loading import load_policy,file_sha
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo import rollout as module


from starVLA.rl.flow_grpo.velocity_graph import NoGradVelocityGraph

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',required=True);p.add_argument('--bank',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();cfg,sft=resolve_config(a.config);configure_numerics()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    result={'status':'RUNNING','scope':'real F G16 K10 no-grad graph, strict equality; not optimizer acceptance',
            'script_sha256':file_sha(__file__),'kernel_sha256':file_sha('starVLA/rl/flow_grpo/velocity_graph.py'),'checkpoint_sha256':cfg['checkpoint_contract']['sha256'],
            'torch':torch.__version__,'device_type':'cuda','device':torch.cuda.get_device_name(),'scenes':[]}
    policy=load_policy(cfg,sft).cuda().bfloat16().eval();policy._inference_qwen_forward_mode='optimized'
    original=module.velocity
    graph=None
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for r in range(4):
            scene=torch.load(Path(a.bank)/f'rollout_rank{r}_v0.pt',map_location='cpu',weights_only=False)[0]
            for k in ('chain','times','old_elementwise_logprob','dimension_mask','transition_mask'):
                setattr(scene,k,getattr(scene,k).cuda())
            cond=policy.encode_policy_condition(scene.observation)
            if graph is None:
                rng=torch.cuda.get_rng_state().clone()
                graph=NoGradVelocityGraph(policy.action_model,scene.chain[:,0,0],torch.zeros(1,device='cuda',dtype=torch.long),cond)
                result['capture_rng_unchanged']=torch.equal(rng,torch.cuda.get_rng_state())
            row={'tokens':scene.observation.tokens,'measurements':[],'checks':[]}
            first=None
            for mode in ('eager','graph','graph','eager'):
                torch.cuda.synchronize();start=time.monotonic()
                with patch.object(module,'velocity',graph if mode=='graph' else original):
                    stats=module.evaluate_transitions(policy,scene.observation,scene)
                torch.cuda.synchronize();row['measurements'].append({'mode':mode,'seconds':time.monotonic()-start})
                if first is None:first=stats
                row['checks'].append({k:{'equal':torch.equal(v,first[k]),'max_abs':float((v-first[k]).abs().max())} for k,v in stats.items()})
            with patch.object(module,'velocity',graph):
                generated=module.sample_chain(policy,scene.observation,scene.spec,0,scene.noise_seed,scene.provenance)
            row['chain_equal']=torch.equal(generated.chain,scene.chain)
            row['old_equal']=torch.equal(generated.old_elementwise_logprob,scene.old_elementwise_logprob)
            result['scenes'].append(row);(out/'probe.json').write_text(json.dumps(result,indent=2))
            print(json.dumps(row),flush=True)
        # A graph must read live parameter values, not a stale autocast snapshot.
        param=policy.action_model.action_decoder.layer2.bias
        before=param.clone();param.add_(.01)
        eager=original(policy,scene.chain[:,0,0],torch.zeros(1,device='cuda',dtype=torch.long),cond)
        graphed=graph(policy,scene.chain[:,0,0],torch.zeros(1,device='cuda',dtype=torch.long),cond)
        result['live_weight_equal']=torch.equal(eager,graphed);param.copy_(before)
    result['status']='PASS' if result['live_weight_equal'] and result['capture_rng_unchanged'] and all(r['chain_equal'] and r['old_equal'] and all(v['equal'] for c in r['checks'] for v in c.values()) for r in result['scenes']) else 'FAIL'
    (out/'probe.json').write_text(json.dumps(result,indent=2))
    if result['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
