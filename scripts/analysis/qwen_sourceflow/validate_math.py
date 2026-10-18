"""Small source oracle: arithmetic, external-KV runner, padding and gradients."""
import argparse
import copy
import json
from pathlib import Path
import sys
import torch
from starVLA.rl.qwen_sourceflow.contracts import UPSTREAM, file_sha
from starVLA.model.modules.qwen_sourceflow.vendor.action_dit import ActionDiT
from starVLA.model.modules.qwen_sourceflow.cached_action_runner import CachedActionRunner
from starVLA.model.modules.qwen_sourceflow.condition_provider import QwenConditionProvider, depth_mapping
from starVLA.model.modules.qwen_sourceflow.source_flow_policy import SourceFlowPolicy
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42)
    sys.path.insert(0,str(UPSTREAM/'src'))
    from simwam.models.wan22.mot import MoT as OriginalMoT
    from simwam.models.wan22.action_dit import ActionDiT as OriginalAction
    from simwam.models.wan22.wan_video_dit import WanVideoDiT
    from simwam.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler as OriginalScheduler
    cfg=dict(hidden_dim=32,action_dim=3,ffn_dim=64,text_dim=48,freq_dim=16,eps=1e-6,
             num_heads=2,attn_head_dim=12,num_layers=3,use_gradient_checkpointing=False)
    action=ActionDiT(**cfg).double();original=OriginalAction(**cfg).double()
    original.load_state_dict(action.state_dict(),strict=True)
    # A real tiny source video class is needed by the original constructor only.
    # Its forward is not used: both oracles consume identical nonzero external KV.
    video=WanVideoDiT(hidden_dim=32,in_dim=4,ffn_dim=64,out_dim=4,text_dim=48,
        freq_dim=16,eps=1e-6,num_heads=2,attn_head_dim=12,num_layers=3,
        patch_size=(1,2,2),has_image_input=False).double()
    oracle=OriginalMoT({'video':video,'action':original},mot_checkpoint_mixed_attn=False)
    runner=CachedActionRunner(action)
    tests=[];errors=[]
    for b in [1,2]:
        x=torch.randn(b,8,3,dtype=torch.float64);time=torch.tensor([.7]*b,dtype=torch.float64)*1000
        cond=dict(context=torch.randn(b,5,48,dtype=torch.float64),context_mask=torch.ones(b,5,dtype=torch.bool),
            visual_mask=torch.ones(b,9,dtype=torch.bool),visual_kv=[dict(k=torch.randn(b,9,24,dtype=torch.float64),v=torch.randn(b,9,24,dtype=torch.float64)) for _ in range(3)])
        y=runner(x,time,cond)
        pre=original.pre_dit(x,time,cond['context'],cond['context_mask'])
        z=oracle.forward_action_with_video_cache(pre['tokens'],pre['freqs'],pre['t_mod'],
            dict(context=pre['context'],mask=pre['context_mask']),cond['visual_kv'],torch.ones(17,17,dtype=torch.bool),9)
        ref=original.post_dit(z,pre)
        torch.testing.assert_close(y,ref,rtol=0,atol=0)
        g=torch.autograd.grad(y.square().mean(),tuple(action.parameters()))
        rg=torch.autograd.grad(ref.square().mean(),tuple(original.parameters()))
        for u,v in zip(g,rg):torch.testing.assert_close(u,v,rtol=0,atol=0)
        tests.append(f'B{b}_source_external_KV_output_and_gradient_exact')
        cond['visual_mask'][0,6:]=False;cond['context_mask'][0,3:]=False
        padded=runner(x,time,cond)
        for i in range(b):
            vi=cond['visual_mask'][i];ti=cond['context_mask'][i]
            single=dict(context=cond['context'][i:i+1,ti],context_mask=torch.ones(1,int(ti.sum()),dtype=torch.bool),
                visual_mask=torch.ones(1,int(vi.sum()),dtype=torch.bool),
                visual_kv=[{k:v[i:i+1,vi] for k,v in item.items()} for item in cond['visual_kv']])
            reference=runner(x[i:i+1],time[i:i+1],single)
            torch.testing.assert_close(padded[i:i+1],reference,rtol=1e-10,atol=1e-10)
        tests.append(f'B{b}_padding_matches_unpadded_individual')
    provider=QwenConditionProvider(action,16,3).double()
    actor_ids={p.data_ptr() for p in action.parameters()}
    assert not any(p.data_ptr() in actor_ids for p in provider.parameters())
    assert depth_mapping(28,30)==[1+l*27//29 for l in range(30)]
    tests+=['provider_no_parameter_alias','fixed_depth_mapping']
    data=torch.randn(2,8,3);noise=torch.randn_like(data);timesteps=torch.tensor([123.4,987.6])
    scheduler=SourceFlowPolicy(action).scheduler;source=OriginalScheduler(1000,5)
    for name,args in [('training_weight',(timesteps,)),('add_noise',(data,noise,timesteps)),
                      ('training_target',(data,noise,timesteps))]:
        torch.testing.assert_close(getattr(scheduler,name)(*args),getattr(source,name)(*args),rtol=0,atol=0)
        tests.append('source_'+name+'_exact')
    t,d=scheduler.build_inference_schedule(10,'cpu',torch.float32)
    ot,od=source.build_inference_schedule(10,'cpu',torch.float32)
    assert torch.equal(t,ot) and torch.equal(d,od)
    target=data;velocity=noise-target;result=noise.clone()
    for delta in d:result=scheduler.step(velocity,delta,result)
    torch.testing.assert_close(result,target,rtol=1e-5,atol=1e-6)
    tests+=['shift5_grid_exact','noise_minus_data_Euler_sign']
    physical=torch.randn(2,8,3)*torch.tensor([20.,5.,1.])
    torch.testing.assert_close(ActionCodec.denorm_odo(ActionCodec.norm_odo(physical)),physical,atol=1e-5,rtol=1e-5)
    tests.append('source_absolute_codec_roundtrip_no_clipping')
    result=dict(status='PASS',tests=tests,source_commit='1433ee7e06a62bd8b89da74455df0d82bda7b726',
                qualification_only=True,driving_result=False,grid=t.tolist(),deltas=d.tolist())
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
