"""Behavior tests with the production DiT class at reduced CPU width.

These are not real pretrained S0/image/NAVSIM integration evidence.
"""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from torch import nn

from starVLA.model.modules.action_effect.effect_flow_adapter import EffectFlowAdapter, EffectFlowAdapterConfig
from starVLA.model.modules.action_effect.effect_distillation import (
    PairedFlowSample, sample_original_pair, corrected_residual_target,
    masked_mean, velocity_loss, effect_consistency_loss)
from starVLA.model.modules.action_effect.preservation import preservation_loss
from starVLA.model.framework.effect_internalized_flow import EffectInternalizedFlow


def tiny_head():
    from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead
    cfg = SimpleNamespace(hidden_size=16, DiTConfig=dict(input_embedding_dim=16,
        num_layers=2, num_attention_heads=2, attention_head_dim=8),
        diffusion_model_cfg=dict(output_dim=16, cross_attention_dim=16,
            dropout=.2, final_dropout=True, interleave_self_attention=True),
        action_dim=4, action_horizon=8, num_inference_timesteps=10,
        noise_beta_alpha=1.5, noise_beta_beta=1., noise_s=.999,
        num_timestep_buckets=1000, repeated_diffusion_steps=8,
        add_pos_embed=True, max_seq_len=8)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1207)
        return FlowmatchingActionHead(SimpleNamespace(framework=SimpleNamespace(
            action_model=cfg, qwenvl=SimpleNamespace(vl_hidden_dim=12))))


def model(arm='B1-trajectory', dropout=0.):
    cfg = EffectFlowAdapterConfig(condition_dim=12, hidden_dim=16, heads=2,
                                  layers=1, dropout=dropout)
    return EffectInternalizedFlow(tiny_head(), EffectFlowAdapter(cfg), arm=arm)


def inputs(batch=2):
    g = torch.Generator().manual_seed(88)
    cond = torch.randn(batch, 8, 12, generator=g)
    minus = torch.randn(batch, 8, 4, generator=g)
    plus = minus+.2
    pair = sample_original_pair(minus, plus, noise_generator=torch.Generator().manual_seed(9),
                                time_generator=torch.Generator().manual_seed(10))
    return cond, pair


def state_copy(module):
    return {k:v.detach().clone() for k,v in module.state_dict().items()}


def effect_context(cond):
    return {'memory': cond, 'ego': torch.zeros(len(cond),1,4),
            'mask': torch.ones(cond.shape[:2],dtype=torch.bool)}


def test_zero_and_disabled_exact_original_full_euler10_not_post_refiner():
    m = model().eval()
    cond, pair = inputs()
    before = state_copy(m.base)
    reference = m.base.action_head.predict_action(cond, initial_noise=pair.noise)
    calls = []
    h = m.adapter.register_forward_hook(lambda _, args, out: calls.append(args[1].detach().clone()))
    actual = m.predict_action(cond, initial_noise=pair.noise)
    h.remove()
    assert len(calls) == 10
    assert torch.equal(calls[0], pair.noise)
    assert torch.equal(actual, reference)
    m.enabled = False
    hook = m.adapter.register_forward_pre_hook(lambda *_: pytest.fail('Disabled adapter executed'))
    assert torch.equal(m.predict_action(cond, initial_noise=pair.noise), reference)
    hook.remove()
    assert all(torch.equal(before[k],v) for k,v in m.base.state_dict().items())


def test_residual_target_includes_base_error_and_two_arms_use_different_paths():
    cond, pair = inputs()
    imperfect = pair.minus-pair.noise+.37
    target = corrected_residual_target(pair, imperfect)
    assert torch.allclose(target, pair.plus-pair.minus-.37, atol=2e-7)
    assert not torch.allclose(target, pair.plus-pair.minus)
    b0, b1 = model('B0'), model('B1-trajectory')
    seen = {}
    for name, m in [('B0',b0), ('B1',b1)]:
        h=m.adapter.register_forward_pre_hook(lambda _, args, n=name: seen.update({n:args[1].detach()}))
        result=m(cond,pair,torch.ones(2,dtype=torch.bool));h.remove()
        expected = (1-pair.times[:,None,None])*pair.noise+pair.times[:,None,None]*(pair.plus if name=='B0' else pair.minus)
        assert torch.equal(seen[name],expected)
        assert result['losses']['velocity'].ndim == 0
    assert not torch.equal(seen['B0'],seen['B1'])
    assert set(b0.state_dict()) == set(b1.state_dict())
    assert sum(p.numel() for p in b0.trainable_parameters()) == sum(p.numel() for p in b1.trainable_parameters())


def test_real_backward_updates_adapter_only_freezes_modes_and_input():
    m=model();m.train()
    cond,pair=inputs();cond.requires_grad_()
    before=state_copy(m.base);old=state_copy(m.adapter)
    optimizer=torch.optim.AdamW(m.trainable_parameters(),lr=.01)
    assert not any(x.training for x in m.base.modules())
    assert all(not p.requires_grad for p in m.base.parameters())
    for _ in range(2):
        optimizer.zero_grad();loss=m(cond,pair,torch.ones(2,dtype=torch.bool))['losses']['velocity']
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None for p in m.base.parameters())
        assert cond.grad is None
        optimizer.step()
    assert any(not torch.equal(old[k],v) for k,v in m.adapter.state_dict().items())
    assert all(torch.equal(before[k],v) for k,v in m.base.state_dict().items())
    assert m.adapter.output.weight.grad.isfinite().all()
    assert m.adapter.context[1].weight.grad.abs().sum() > 0
    with torch.no_grad():
        assert not torch.equal(m.predict_action(cond,initial_noise=pair.noise),
                               m.base.action_head.predict_action(cond,initial_noise=pair.noise))


def test_adapter_init_reproducible_preserves_external_rng_and_storage_independent():
    cfg=EffectFlowAdapterConfig(condition_dim=12,hidden_dim=16,heads=2,layers=1)
    before=torch.get_rng_state().clone()
    a,b=EffectFlowAdapter(cfg),EffectFlowAdapter(cfg)
    assert torch.equal(before,torch.get_rng_state())
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert all(p.data_ptr()!=q.data_ptr() for p,q in zip(a.parameters(),b.parameters()))
    c=EffectFlowAdapter(replace(cfg,init_seed=43))
    assert not torch.equal(a.position,c.position)
    assert torch.count_nonzero(a.output.weight)==0 and torch.count_nonzero(a.output.bias)==0


def test_original_beta_negative_tail_and_shared_pair_noise_resume():
    _,pair=inputs(8)
    t=pair.times.clone();t[0]=-.0005
    changed=replace(pair,times=t).validate()
    assert torch.equal(changed.target_velocity(),pair.plus-pair.noise)
    ng,tg=torch.Generator().manual_seed(3),torch.Generator().manual_seed(4)
    ns,ts=ng.get_state(),tg.get_state()
    x=sample_original_pair(pair.minus,pair.plus,noise_generator=ng,time_generator=tg)
    ng.set_state(ns);tg.set_state(ts)
    y=sample_original_pair(pair.minus,pair.plus,noise_generator=ng,time_generator=tg)
    assert torch.equal(x.noise,y.noise) and torch.equal(x.times,y.times)
    with pytest.raises(ValueError):sample_original_pair(pair.minus,pair.plus,noise_generator=ng,time_generator=ng)
    with pytest.raises(ValueError):replace(pair,times=torch.full((8,),-.1)).validate()


def test_zero_valid_and_microbatch_ddp_global_denominators():
    pred=torch.tensor([1.,2.,3.,4.],requires_grad=True)
    target=torch.zeros(4);valid=torch.tensor([True,False,True,True])
    full=masked_mean((pred-target).square(),valid)
    fullgrad=torch.autograd.grad(full,pred,retain_graph=True)[0]
    # DDP averages the two rank gradients. Each has a distinct valid count.
    rank_losses=[masked_mean((pred[s]-target[s]).square(),valid[s],global_denominator=3,world_size=2)
                 for s in (slice(0,2),slice(2,4))]
    ddpgrad=torch.autograd.grad(sum(rank_losses)/2,pred,retain_graph=True)[0]
    assert torch.equal(fullgrad,ddpgrad)
    micro=sum(masked_mean((pred[s]-target[s]).square(),valid[s],global_denominator=3)
              for s in (slice(0,1),slice(1,4)))
    assert torch.equal(torch.autograd.grad(micro,pred)[0],fullgrad)
    empty=masked_mean(pred,torch.zeros(4,dtype=torch.bool),global_denominator=0)
    assert empty.requires_grad and empty.item()==0
    with pytest.raises(FloatingPointError):masked_mean(pred*float('nan'),torch.zeros(4,dtype=torch.bool))
    with pytest.raises(ValueError):masked_mean(pred,valid,global_denominator=float('nan'))


class SyntheticEffectTeacher(nn.Module):
    """Analytic gradient fixture, never a real feedback/teacher qualification."""
    def __init__(self):
        super().__init__();self.scale=nn.Parameter(torch.tensor(1.))
    def forward(self,condition,minus,delta):
        return delta.mean((1,2))[:,None]*self.scale


def test_effect_arm_requires_real_supervision_frozen_teacher_and_nonzero_extra_gradient():
    cond,pair=inputs();m=model('B1-effect')
    with pytest.raises(ValueError,match='verified G1'):m(cond,pair,torch.ones(2,dtype=torch.bool))
    teacher=SyntheticEffectTeacher()
    kwargs=dict(teacher=teacher,true_delta=torch.full((2,1),.8),
                effect_valid=torch.ones(2,1,dtype=torch.bool),effect_scales=[1.],
                effect_context=effect_context(cond))
    with pytest.raises(ValueError,match='frozen eval'):m(cond,pair,torch.ones(2,dtype=torch.bool),**kwargs)
    teacher.requires_grad_(False).eval()
    result=m(cond,pair,torch.ones(2,dtype=torch.bool),**kwargs)
    assert set(result['losses'])=={'velocity','effect_consistency','effect_direction'}
    effect=result['losses']['effect_consistency']+.2*result['losses']['effect_direction']
    effect.backward()
    assert m.adapter.output.weight.grad.abs().sum()>0 and teacher.scale.grad is None
    with pytest.raises(ValueError,match='silently consume'):
        model('B1-trajectory')(cond,pair,torch.ones(2,dtype=torch.bool),**kwargs)
    with pytest.raises(ValueError,match='global weight'):
        m(cond,pair,torch.ones(2,dtype=torch.bool),world_size=2,
          global_denominator=128,effect_global_denominator=4,**kwargs)


def test_effect_global_gap_weight_is_not_cancelled_or_squared():
    cond,pair=inputs();teacher=SyntheticEffectTeacher().requires_grad_(False).eval()
    predicted=(pair.minus-pair.noise).clone().requires_grad_()
    result=effect_consistency_loss(teacher,effect_context(cond),pair,predicted,torch.tensor([[.95],[.02]]),
             torch.ones(2,1,dtype=torch.bool),[1.])
    grad=torch.autograd.grad(result['effect_direction'],predicted)[0]
    assert torch.allclose(grad[0].abs().sum()/grad[1].abs().sum(),torch.tensor(.95/.02),rtol=1e-5)


def test_preservation_shared_state_noise_zero_then_nonzero():
    cond,pair=inputs();m=model()
    assert preservation_loss(m,cond,pair,torch.ones(2,dtype=torch.bool)).item()==0
    with torch.no_grad():m.adapter.output.bias.fill_(.2)
    value=preservation_loss(m,cond,pair,torch.ones(2,dtype=torch.bool))
    assert torch.allclose(value,torch.tensor(.04),atol=1e-6)


def test_next_update_exact_with_model_optimizer_rng_and_two_stream_restore():
    m=model(dropout=.1);opt=torch.optim.AdamW(m.trainable_parameters(),lr=.001)
    cond,pair=inputs();ng,tg=torch.Generator().manual_seed(4),torch.Generator().manual_seed(5)
    def update(m,opt,ng,tg):
        draw=sample_original_pair(pair.minus,pair.plus,noise_generator=ng,time_generator=tg)
        opt.zero_grad();loss=m(cond,draw,torch.ones(2,dtype=torch.bool))['losses']['velocity']
        loss.backward();opt.step()
        return draw.noise.clone(),draw.times.clone(),loss.detach().clone(),state_copy(m)
    update(m,opt,ng,tg)
    checkpoint=deepcopy({'model':m.state_dict(),'optimizer':opt.state_dict(),
                         'noise':ng.get_state(),'time':tg.get_state(),'rng':torch.get_rng_state()})
    reference=update(m,opt,ng,tg)
    new=model(dropout=.1);newopt=torch.optim.AdamW(new.trainable_parameters(),lr=.001)
    new.load_state_dict(checkpoint['model'],strict=True);newopt.load_state_dict(checkpoint['optimizer'])
    ng.set_state(checkpoint['noise']);tg.set_state(checkpoint['time']);torch.set_rng_state(checkpoint['rng'])
    resumed=update(new,newopt,ng,tg)
    assert all(torch.equal(a,b) for a,b in zip(reference[:3],resumed[:3]))
    assert all(torch.equal(v,resumed[3][k]) for k,v in reference[3].items())


def test_effect_context_whitelist_and_deployment_reject_labels():
    cond,pair=inputs();m=model('B1-effect')
    teacher=SyntheticEffectTeacher().requires_grad_(False).eval()
    poisoned={**effect_context(cond),'feedback':torch.ones(2,1)}
    with pytest.raises(ValueError,match='never GT or feedback'):
        effect_consistency_loss(teacher,poisoned,pair,pair.minus-pair.noise,
                                torch.zeros(2,1),torch.ones(2,1,dtype=torch.bool),[1.])
    with pytest.raises(TypeError):
        m.predict_action(cond,initial_noise=pair.noise,feedback=torch.ones(2,1))
    with pytest.raises(TypeError):
        m.predict_action(cond,initial_noise=pair.noise,teacher=teacher)


def test_export_strict_new_process_has_no_simulator_or_teacher_dependency(tmp_path):
    import os, subprocess, sys
    m=model().eval();cond,pair=inputs()
    with torch.no_grad():m.adapter.output.bias.fill_(.03)
    expected=m.predict_action(cond,initial_noise=pair.noise)
    artifact=tmp_path/'bundle.pt';result=tmp_path/'out.pt'
    torch.save({'state':m.state_dict(),'condition':cond,'noise':pair.noise},artifact)
    code="""
import builtins,runpy,sys,torch
original=builtins.__import__
def restricted(name,*args,**kwargs):
    if name.startswith('navsim'):raise RuntimeError('Deployment imported NAVSIM')
    return original(name,*args,**kwargs)
builtins.__import__=restricted
torch.set_num_threads(1)
factory=runpy.run_path(sys.argv[1])['model']
m=factory().eval()
p=torch.load(sys.argv[2],weights_only=True,map_location='cpu')
m.load_state_dict(p['state'],strict=True)
torch.save(m.predict_action(p['condition'],initial_noise=p['noise']),sys.argv[3])
"""
    completed=subprocess.run([sys.executable,'-c',code,__file__,str(artifact),str(result)],
        capture_output=True,text=True,timeout=60,
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1',
             'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TORCHINDUCTOR_COMPILE_THREADS':'1'})
    assert completed.returncode==0,completed.stderr
    assert torch.equal(expected,torch.load(result,weights_only=True))


def test_analytic_euler10_endpoint_requires_corrected_not_naive_delta_target():
    # Analytic constant-field counterexample; distinct from pretrained evidence.
    _,pair=inputs(1)
    base=torch.full_like(pair.minus,.37)
    corrected=corrected_residual_target(pair,base)
    good,bad=pair.noise.clone(),pair.noise.clone()
    for _ in range(10):
        good=good+.1*(base+corrected)
        bad=bad+.1*(base+pair.plus-pair.minus)
    assert torch.allclose(good,pair.plus,atol=1e-6)
    assert not torch.allclose(bad,pair.plus,atol=1e-3)
