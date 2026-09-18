"""Fixed real F/G16/K10 bank: exact outputs and synchronized cache timings."""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch
import torch
from starVLA.rl.flow_grpo.audit import capture_source_environment
from starVLA.rl.flow_grpo.config import resolve_config
from starVLA.rl.flow_grpo.loading import load_policy, file_sha
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.inference_cache import inference_velocity_snapshot
from starVLA.rl.flow_grpo import rollout as module


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',required=True);p.add_argument('--bank',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();cfg,sft=resolve_config(a.config);configure_numerics()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);capture_source_environment(out,cfg)
    # Fixed first scene on each of the first four ranks = global positions0..3.
    bank=[torch.load(Path(a.bank)/f'rollout_rank{r}_v0.pt',map_location='cpu',weights_only=False)[0] for r in range(4)]
    policy=load_policy(cfg,sft).cuda().bfloat16().eval();policy._inference_qwen_forward_mode='optimized'
    identity={n:(id(v),v.data_ptr(),v.dtype) for n,v in policy.named_parameters()}
    result={'scope':'no-grad native velocity/transition/full-chain, exact tolerance0; no optimizer acceptance',
        'checkpoint_sha256':cfg['checkpoint_contract']['sha256'],'script_sha256':file_sha(__file__),
        'torch_version':torch.__version__,'gpu':torch.cuda.get_device_name(),'scenes':[],'status':'RUNNING'}
    original=module.velocity
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for scene in bank:
            scene.chain=scene.chain.cuda();scene.times=scene.times.cuda()
            scene.old_elementwise_logprob=scene.old_elementwise_logprob.cuda()
            scene.dimension_mask=scene.dimension_mask.cuda();scene.transition_mask=scene.transition_mask.cuda()
            # Warmup is fixed and excluded from timing, never selected by speed.
            module.evaluate_transitions(policy,scene.observation,scene)
            measurements=[];first=None;row={'tokens':scene.observation.tokens,'measurements':measurements,'checks':[]}
            for mode in ['original','snapshot','snapshot','original','original','snapshot']:
                torch.cuda.synchronize();start=time.monotonic()
                if mode=='snapshot':
                    with inference_velocity_snapshot(policy.action_model) as fn:
                        def cached(policy_,x,bucket,condition,checkpoint=False):
                            assert policy_ is policy
                            return fn(x,bucket,condition)
                        with patch.object(module,'velocity',cached):
                            stats=module.evaluate_transitions(policy,scene.observation,scene)
                else:stats=module.evaluate_transitions(policy,scene.observation,scene)
                torch.cuda.synchronize();seconds=time.monotonic()-start
                measurements.append({'mode':mode,'seconds_including_snapshot_and_qwen':seconds})
                if first is None:first={k:v.detach().clone() for k,v in stats.items()}
                checks={k:{'equal':torch.equal(v,first[k]),'max_abs':float((v-first[k]).abs().max()),
                           'finite':bool(torch.isfinite(v).all())} for k,v in stats.items()}
                row['checks'].append({'mode':mode,'tensors':checks})
            with inference_velocity_snapshot(policy.action_model) as fn:
                def cached(policy_,x,bucket,condition,checkpoint=False):return fn(x,bucket,condition)
                with patch.object(module,'velocity',cached):
                    current=module.sample_chain(policy,scene.observation,scene.spec,0,scene.noise_seed,scene.provenance)
            row['sampled_chain_equals_saved']=torch.equal(current.chain,scene.chain)
            row['sampled_old_logprob_equals_saved']=torch.equal(current.old_elementwise_logprob,scene.old_elementwise_logprob)
            result['scenes'].append(row);(out/'probe.json').write_text(json.dumps(result,indent=2))
            print(row['tokens'],measurements,'chain',row['sampled_chain_equals_saved'],flush=True)
        if identity!={n:(id(v),v.data_ptr(),v.dtype) for n,v in policy.named_parameters()}:
            raise RuntimeError('parameter object/storage/dtype changed')
        if any(p.grad is not None for p in policy.parameters()):raise RuntimeError('no-grad probe created parameter gradients')
    result['parameter_objects_restored']=True
    result['status']='PASS' if all(r['sampled_chain_equals_saved'] and r['sampled_old_logprob_equals_saved']
        and all(t['equal'] and t['finite'] for c in r['checks'] for t in c['tensors'].values()) for r in result['scenes']) else 'FAIL'
    (out/'probe.json').write_text(json.dumps(result,indent=2))
    if result['status']!='PASS':raise SystemExit(1)
    (out/'COMPLETE').write_text(json.dumps({'sha256':file_sha(out/'probe.json')}))


if __name__=='__main__':main()
