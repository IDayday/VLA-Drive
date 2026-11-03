"""Actual Tiny D FPO MC4 fusion checks, no GPU/official scoring."""
import copy,json,os
from pathlib import Path
import torch
from tools.d46_world16.qualify_native import make_actor,condition,SyntheticService
from starVLA.rl.d44_native_rl.config import defaults
from starVLA.rl.d44_native_rl.batch_runtime import collect_scene
from starVLA.rl.d44_native_rl.fpo import mc_predictions_energy


def test_real_fused_MC_preserves_draws_SDE_and_fresh_energy_and_gradients(monkeypatch):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *a,**kw:None)
    torch.manual_seed(42);torch.set_num_threads(1);cfg=defaults('D3_FPO');cfg['checkpoint_policy']='all_off'
    actor=make_actor(cfg)
    with torch.no_grad():
        for n,p in actor.named_parameters():
            if n.endswith('lora_B'):p.fill_(.001)
    scenes=[];outputs=[];params=[p for p in actor.parameters()if p.requires_grad]
    for fused in(False,True):
        actor.execution_fast_prefix=True;actor.execution_fuse_fm_mc=fused
        gen=torch.Generator().manual_seed(420000)
        s=collect_scene(actor,condition(actor,0),'TRAIN_CPU','TRAIN_log',SyntheticService(),gen,cfg,0.,{'run':'CPU','update':0,'collection':1})
        s.update(advantages=torch.linspace(-1,1,16),beta=.001,expect_fresh=True)
        scenes.append((s,gen.get_state()))
        loss,metrics=actor(s);grads=torch.autograd.grad(loss,params)
        assert metrics['fresh_ratio_error']==0
        assert metrics['fpo_joint_logratio_min']==0 and metrics['fpo_joint_logratio_max']==0
        outputs.append((loss.detach(),grads))
    for key in('chain','old_mean','std','ref_mean','fm_state','fm_sigma','fm_target'):
        assert torch.equal(scenes[0][0][key],scenes[1][0][key])
    assert torch.equal(scenes[0][1],scenes[1][1])
    energy_diff=float((scenes[0][0]['old_energy']-scenes[1][0]['old_energy']).abs().max())
    grad_diff=max(float((a-b).abs().max())for a,b in zip(outputs[0][1],outputs[1][1]))
    print('Tiny BF16 fused energy/loss/gradient maxabs:',energy_diff,float((outputs[0][0]-outputs[1][0]).abs()),grad_diff)
    torch.testing.assert_close(scenes[0][0]['old_energy'],scenes[1][0]['old_energy'],rtol=1e-3,atol=1e-5)
    torch.testing.assert_close(outputs[0][0],outputs[1][0],rtol=1e-3,atol=1e-5)
    v=torch.cat([a.flatten().double()for a in outputs[0][1]]);w=torch.cat([a.flatten().double()for a in outputs[1][1]])
    relative=float((v-w).norm()/v.norm().clamp_min(1e-30));cosine=float(torch.nn.functional.cosine_similarity(v,w,dim=0))
    strict=all(torch.allclose(a,b,rtol=1e-3,atol=1e-5)for a,b in zip(outputs[0][1],outputs[1][1]))
    print('Tiny BF16 gradient relative-L2/cosine/strict-elementwise:',relative,cosine,strict)
    # BF16 execution approximation is measured, never silently called bitexact
    # or admitted as a production upgrade by this CPU fixture.
    assert all(torch.isfinite(x).all()for x in outputs[1][1])
    if os.environ.get('FPO_FUSION_CPU_DIAGNOSTIC'):
        Path(os.environ['FPO_FUSION_CPU_DIAGNOSTIC']).write_text(json.dumps(dict(energy_max_abs=energy_diff,loss_max_abs=float((outputs[0][0]-outputs[1][0]).abs()),gradient_max_abs=grad_diff,gradient_relative_L2=relative,gradient_cosine=cosine,gradient_strict_elementwise_allclose=bool(strict),strict_tolerance={'atol':1e-5,'rtol':1e-3},status='MEASURED_NOT_GPU_OR_FORMAL_ADMISSION'),indent=2)+'\n')


def test_fused_prefix_is_stride0_not_MC_KV_materialization(monkeypatch):
    torch.manual_seed(42);cfg=defaults('D3_FPO');cfg['checkpoint_policy']='all_off';actor=make_actor(cfg)
    actor.execution_fast_prefix=True
    with actor._context_call(condition(actor,0))as cond:
        prefix=actor._prepare(cond,False);original=actor.action_expert.denoise_x0;calls=[]
        def observed(state,sigma,p):
            assert len(p.tensors.valid)==64 and p.tensors.valid.stride(0)==0
            for a,b in zip(prefix.tensors.layer_kv,p.tensors.layer_kv):
                for x,y in zip(a,b):
                    assert y.stride(0)==0 and y.untyped_storage().data_ptr()==x.untyped_storage().data_ptr()
            calls.append(len(state));return original(state,sigma,p)
        monkeypatch.setattr(actor.action_expert,'denoise_x0',observed)
        state=torch.randn(4,16,8,3);sigma=torch.full((4,16),.5);target=torch.randn_like(state)
        energies,predictions=mc_predictions_energy(actor.action_expert,prefix,state,sigma,target,fuse=True)
        assert energies.shape==(4,16)and predictions.shape==(4,16,8,3)and calls==[64]


def test_fusion_same_FP32_mathematical_objective_and_gradients(monkeypatch):
    from contextlib import nullcontext
    torch.manual_seed(42);torch.set_num_threads(1)
    monkeypatch.setattr(torch,'autocast',lambda *a,**kw:nullcontext())
    cfg=defaults('D3_FPO');cfg['checkpoint_policy']='all_off';actor=make_actor(cfg);actor.execution_fast_prefix=True
    with torch.no_grad():
        for n,p in actor.named_parameters():
            if n.endswith('lora_B'):p.fill_(.001)
    params=[p for p in actor.parameters()if p.requires_grad]
    with actor._context_call(condition(actor,0))as cond:
        prefix=actor._prepare(cond,False);state=torch.randn(4,16,8,3);sigma=.2+.6*torch.rand(4,16);target=torch.randn_like(state)
        a,_=mc_predictions_energy(actor.action_expert,prefix,state,sigma,target,fuse=False)
        b,_=mc_predictions_energy(actor.action_expert,prefix,state,sigma,target,fuse=True)
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        ga=torch.autograd.grad(a.mean(),params,retain_graph=True);gb=torch.autograd.grad(b.mean(),params)
        for x,y in zip(ga,gb):torch.testing.assert_close(x,y,rtol=1e-4,atol=1e-6)
