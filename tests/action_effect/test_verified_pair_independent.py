"""Independent CPU review of verified-pair IEG; no real S0/NAVSIM claims.

Run against the explicitly named candidate worktree, not the reviewer's imports:
ACTION_EFFECT_REVIEW_SOURCE=/absolute/candidate python -m pytest THIS_FILE
"""
from dataclasses import replace
from pathlib import Path
from copy import deepcopy
import os
import sys

candidate=Path(os.environ.get('ACTION_EFFECT_REVIEW_SOURCE',Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0,str(candidate));sys.path.insert(1,str(candidate/'tests/action_effect'))

import torch
import pytest
from torch import nn
from starVLA.model.modules.action_effect.effect_flow_adapter import EffectFlowAdapter,EffectFlowAdapterConfig
from starVLA.model.modules.action_effect.effect_distillation import verified_pair_effect_loss, corrected_residual_target
from starVLA.model.framework.effect_internalized_flow import EffectInternalizedFlow
from tools.action_effect.train_effect_flow import FlowStep,FlowLossRecipe,RawActionEffectTeacher
from tools.action_effect.training import run_training
from test_flow_internalization import tiny_head,inputs

torch.set_num_threads(1)


def adapter():
    return EffectFlowAdapter(EffectFlowAdapterConfig(condition_dim=12,hidden_dim=16,heads=2,layers=1,effect_dim=3,init_seed=81))


def model(arm='B1-effect'):
    return EffectInternalizedFlow(tiny_head(),adapter(),arm=arm,effect_protocol='verified_pair_v1')


def state_equal(first,second):
    return all(torch.equal(v,second.state_dict()[k]) for k,v in first.state_dict().items())


def test_all_arms_allocate_same_random_state_and_legacy_velocity_initialization():
    models=[model(a) for a in ('B0','B1-trajectory','B1-effect')]
    assert all(state_equal(models[0].adapter,x.adapter) for x in models)
    assert len({sum(p.numel() for p in x.trainable_parameters()) for x in models})==1
    legacy=EffectFlowAdapter(replace(models[0].adapter.config,effect_dim=0))
    assert all(torch.equal(v,models[0].adapter.state_dict()[k]) for k,v in legacy.state_dict().items())
    assert all(p.untyped_storage().data_ptr()!=q.untyped_storage().data_ptr()
               for p,q in zip(models[0].adapter.parameters(),models[1].adapter.parameters()))


def test_new_module_init_restores_cpu_rng_and_does_not_call_global_cuda_seed(monkeypatch):
    before=torch.get_rng_state().clone()
    def forbidden(*args,**kwargs):raise AssertionError('global all-device manual_seed touched')
    monkeypatch.setattr(torch,'manual_seed',forbidden)
    first,second=adapter(),adapter()
    assert torch.equal(before,torch.get_rng_state()) and state_equal(first,second)


def targets(batch=2):
    return dict(true_delta=torch.tensor([[.95,.02,-.1]]).repeat(batch,1),effect_valid=torch.ones(batch,3,dtype=torch.bool),
                effect_scales=(1.,20.,4.),teacher_embedding=torch.ones(batch,16)*.07)


def test_labels_and_teacher_features_change_losses_but_not_forward_inputs_or_velocity():
    m=model();cond,pair=inputs();valid=torch.tensor([True,False]);args=targets()
    first=m(cond,pair,valid,**args)
    changed={**args,'true_delta':args['true_delta']*-3,'teacher_embedding':args['teacher_embedding']+8}
    second=m(cond,pair,valid,**changed)
    assert torch.equal(first['prediction'],second['prediction'])
    assert torch.equal(first['student_effect'],second['student_effect'])
    assert torch.equal(first['student_embedding'],second['student_embedding'])
    assert not torch.equal(first['losses']['effect_representation'],second['losses']['effect_representation'])
    with pytest.raises(ValueError,match='never an endpoint'):
        m(cond,pair,valid,teacher=nn.Linear(1,1),**args)
    with pytest.raises(ValueError,match='cannot silently'):
        model('B0')(cond,pair,valid,**args)
    imperfect=pair.minus-pair.noise+.37
    expected=pair.plus-pair.noise-imperfect
    torch.testing.assert_close(corrected_residual_target(pair,imperfect),expected,rtol=0,atol=0)


def test_effect_auxiliary_gradient_reaches_shared_body_without_teacher_target_gradient():
    m=model();cond,pair=inputs();cond.requires_grad_();args=targets()
    args['true_delta'].requires_grad_();args['teacher_embedding'].requires_grad_()
    prior={k:v.clone() for k,v in m.base.state_dict().items()}
    out=m(cond,pair,torch.tensor([True,False]),**args)
    loss=sum(out['losses'][n] for n in ('effect_consistency','effect_direction','effect_representation'))
    loss.backward()
    assert m.adapter.context[1].weight.grad.abs().sum()>0
    assert m.adapter.effect_head.weight.grad.abs().sum()>0 and m.adapter.effect_projection.weight.grad.abs().sum()>0
    assert m.adapter.output.weight.grad is None or torch.count_nonzero(m.adapter.output.weight.grad)==0
    assert args['true_delta'].grad is None and args['teacher_embedding'].grad is None and cond.grad is None
    assert all(p.grad is None for p in m.base.parameters()) and all(torch.equal(v,prior[k]) for k,v in m.base.state_dict().items())


def test_direction_keeps_small_gap_mass_and_exact_expected_gradient_ratio():
    p=torch.tensor([[.05],[.05]],requires_grad=True);h=torch.zeros(2,2,requires_grad=True)
    common=dict(true_delta=torch.tensor([[.95],[.02]]),teacher_embedding=torch.zeros_like(h),
                effect_valid=torch.ones(2,1,dtype=torch.bool),repair_valid=torch.ones(2,dtype=torch.bool),effect_scales=(1.,))
    loss=verified_pair_effect_loss(p,h,**common)['effect_direction']
    gradient,=torch.autograd.grad(loss,p)
    assert float(gradient[0]/gradient[1])==pytest.approx(.95/.02,rel=1e-6)
    one=verified_pair_effect_loss(p[1:],h[1:],**{k:(v[1:] if isinstance(v,torch.Tensor) else v) for k,v in common.items()})
    expected=torch.nn.functional.softplus(-p[1,0]/.1)
    torch.testing.assert_close(one['effect_direction'],expected,rtol=0,atol=1e-7)


def test_global_denominators_microbatch_and_simulated_ddp_gradient_match():
    p=torch.tensor([[.1,.2,-.1],[.3,.1,.2],[-.2,.4,.3]],requires_grad=True)
    h=torch.randn(3,4,requires_grad=True,generator=torch.Generator().manual_seed(9))
    target=torch.tensor([[.95,.02,-.1],[.02,.5,.1],[.1,.3,.2]])
    valid=torch.tensor([[1,1,1],[1,0,0],[1,1,1]],dtype=torch.bool);repair=torch.tensor([True,True,False])
    scales=torch.tensor([1.,20.,4.]);z=torch.zeros_like(h)
    base=dict(true_delta=target,teacher_embedding=z,effect_valid=valid,repair_valid=repair,effect_scales=scales)
    full=verified_pair_effect_loss(p,h,**base);loss=sum(full.values());g=torch.autograd.grad(loss,(p,h),retain_graph=True)
    weight=(target/scales).abs()*valid*repair[:,None]
    den=dict(global_denominator=int((valid&repair[:,None]).sum()),direction_global_denominator=float(weight.sum()),
             representation_global_denominator=int(repair.sum())*h.shape[1])
    pieces=[]
    for sl in (slice(0,1),slice(1,3)):
        kw={k:(v[sl] if isinstance(v,torch.Tensor) and k!='effect_scales' else v) for k,v in base.items()}
        pieces.append(sum(verified_pair_effect_loss(p[sl],h[sl],**kw,**den).values()))
    micro=torch.autograd.grad(sum(pieces),(p,h),retain_graph=True)
    for a,b in zip(g,micro):torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)
    # DDP averages local gradients: the function's world_size compensates this.
    ddp=[]
    for sl in (slice(0,1),slice(1,3)):
        kw={k:(v[sl] if isinstance(v,torch.Tensor) and k!='effect_scales' else v) for k,v in base.items()}
        ddp.append(sum(verified_pair_effect_loss(p[sl],h[sl],**kw,**den,world_size=2).values()))
    distributed=torch.autograd.grad(sum(ddp)/2,(p,h))
    for a,b in zip(g,distributed):torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)
    empty=verified_pair_effect_loss(p,h,**{**base,'repair_valid':torch.zeros(3,dtype=torch.bool)})
    assert all(torch.isfinite(x) and x.item()==0 for x in empty.values())


def step_fixture():
    cond,pair=inputs(3);physical=torch.zeros(3,8,3);physical[...,0]=torch.arange(1,9)*2.5
    rows=[];records={}
    for i in range(3):
        repair=i!=1;targetphysical=physical[i].clone();targetphysical[:,1]+=.15 if repair else 0
        rows.append({'scene_id':str(i),'is_repair':repair,'is_preserve':not repair,
            'target_raw':pair.plus[i] if repair else pair.minus[i], 'target_physical':targetphysical,
            'effect_delta':torch.tensor([.95,.02,-.1]) if repair else torch.zeros(3),'effect_valid':torch.tensor([1,1,1],dtype=torch.bool)})
        records[str(i)]={'scene_id':str(i),'role':'fit','condition':cond[i], 'raw':pair.minus[i],
            'physical':physical[i], 'memory':cond[i], 'ego':torch.zeros(1,4),
            'target_raw':torch.full((8,4),float('nan'))} # irrelevant GT cache must not be read.
    class Teacher(nn.Module):
        def __init__(self):super().__init__();self.scalar=nn.Parameter(torch.tensor(1.),requires_grad=False);self.eval();self.calls=[]
        def forward(self,*args):raise AssertionError('Legacy eps+v teacher query forbidden')
        def verified_pair(self,context,base,modified):
            assert not torch.is_grad_enabled() and not base.requires_grad and not modified.requires_grad
            expected=[r for r in rows if r['is_repair']]
            # FlowStep in this test consumes the fixed order [0,1,2].
            assert torch.equal(base,torch.stack([records[r['scene_id']]['physical'] for r in expected]))
            assert torch.equal(modified,torch.stack([r['target_physical'] for r in expected]))
            self.calls.append((base.clone(),modified.clone()))
            return {'mean_delta':torch.stack([r['effect_delta'] for r in expected]),
                    'effect_embedding':torch.ones(len(base),16)*.07}
    def batch_context(records,device):
        memory=torch.stack([r['memory'] for r in records]).to(device)
        return {'memory':memory,'ego':torch.stack([r['ego'] for r in records]).to(device),
                'mask':torch.ones(memory.shape[:2],dtype=torch.bool,device=device)}
    return rows,records,Teacher,batch_context


def test_step_teacher_sees_only_native_pair_once_and_noise_repeat_microbatch_is_identical():
    rows,records,Teacher,bc=step_fixture();states=[];losses=[];models=[]
    for micro in (3,1):
        m=model();teacher=Teacher();ng=torch.Generator().manual_seed(31);tg=torch.Generator().manual_seed(32)
        step=FlowStep(rows,records.__getitem__,recipe=FlowLossRecipe(microbatch_scenes=micro),device='cpu',
            noise_generator=ng,time_generator=tg,context_batcher=bc,teacher=teacher,effect_scales=(1.,20.,4.))
        losses.append(step(m,[0,1,2],0));models.append(m);states.append((ng.get_state(),tg.get_state()))
        assert len(teacher.calls)==1 and teacher.calls[0][0].shape[0]==2
        assert losses[-1]['FM_noise_presentations']==24 and losses[-1]['verified_pair_teacher_calls']==1
        assert all(p.grad is None for p in teacher.parameters())
    assert all(torch.equal(a,b) for a,b in zip(*states))
    for k in ('velocity','preserve','effect_consistency','effect_direction','effect_representation'):
        assert losses[0][k]==pytest.approx(losses[1][k],rel=1e-5,abs=1e-7)
    for p,q in zip(models[0].adapter.parameters(),models[1].adapter.parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=3e-7)


def test_export_strips_heads_and_euler10_never_calls_teacher_or_auxiliary_heads():
    m=model().eval();cond,pair=inputs()
    with torch.no_grad():m.adapter.output.weight.fill_(.01)
    calls=[]
    def forbidden(*args):raise AssertionError('training-only effect head ran in Euler10')
    hs=[m.adapter.effect_projection.register_forward_hook(forbidden),m.adapter.effect_head.register_forward_hook(forbidden),
        m.adapter.register_forward_hook(lambda _,args,out:calls.append(1))]
    out=m.predict_action(cond,initial_noise=pair.noise)
    for h in hs:h.remove()
    assert len(calls)==10
    export=m.adapter.velocity_export();assert export['config']['effect_dim']==0
    assert not any(k.startswith(('effect_head.','effect_projection.')) for k in export['state_dict'])
    restored=EffectFlowAdapter(EffectFlowAdapterConfig(**export['config']));restored.load_state_dict(export['state_dict'],strict=True)
    pure=EffectInternalizedFlow(tiny_head(),restored,arm='B1-trajectory').eval()
    assert torch.equal(out,pure.predict_action(cond,initial_noise=pair.noise))


def test_repeated_fp32_direction_mass_roundoff_is_not_invalid_global_denominator():
    # Original scene sum times repeat8 is mathematically the same denominator as
    # the repeated population sum, but FP32 reduction orders differ by two ulps.
    delta=torch.tensor([[.2569085359573364,.0517691969871521,.3522474765777588,
        .027882516384124756,.6525750756263733,.5727039575576782,.664831280708313,
        .37794923782348633,.7857222557067871,.21219444274902344,.9239256978034973]])
    valid=torch.tensor([[True]*10+[False]]);weight=delta.abs()*valid
    denominator=float(weight.sum())*8
    expanded_delta=delta.repeat(8,1);expanded_valid=valid.repeat(8,1)
    assert float(weight.repeat(8,1).sum())>denominator
    prediction=torch.zeros_like(expanded_delta,requires_grad=True)
    kwargs=dict(true_delta=expanded_delta,teacher_embedding=torch.zeros(8,16),
        effect_valid=expanded_valid,repair_valid=torch.ones(8,dtype=torch.bool),effect_scales=[1.]*11,
        global_denominator=int(valid.sum())*8,representation_global_denominator=128)
    result=verified_pair_effect_loss(prediction,torch.zeros(8,16),
        direction_global_denominator=denominator,**kwargs)
    expected=(torch.nn.functional.softplus(-prediction/.1)*weight.repeat(8,1)).sum()/denominator
    torch.testing.assert_close(result['effect_direction'],expected,rtol=0,atol=1e-7)
    # The numerical-tolerance fix must not silently accept a genuinely too-small
    # denominator or replace it with clamp_min(1) / a local mean.
    with pytest.raises(ValueError,match='direction.*denominator'):
        verified_pair_effect_loss(prediction,torch.zeros(8,16),
            direction_global_denominator=denominator*.5,**kwargs)


def test_verified_pair_shared_loop_resume_retains_model_optimizer_rng_and_both_streams(tmp_path):
    rows,records,_,bc=step_fixture()
    class Teacher(nn.Module):
        def __init__(self):super().__init__();self.eval()
        @torch.no_grad()
        def verified_pair(self,context,base,modified):
            d=(modified-base).mean((1,2))
            return {'mean_delta':torch.zeros(len(base),3),'effect_embedding':d[:,None].repeat(1,16)}
    cfg={'training':{'max_updates':4,'learning_rate':.001,'weight_decay':.01,
        'warmup_updates':1,'batch_scenes':4,'gradient_clip':1.,'validation_every':2}}
    def execute(out,*,resume=False,stop_after=None):
        m=model();ng=torch.Generator().manual_seed(731);tg=torch.Generator().manual_seed(732)
        step=FlowStep(rows,records.__getitem__,recipe=FlowLossRecipe(microbatch_scenes=2),device='cpu',
            noise_generator=ng,time_generator=tg,context_batcher=bc,teacher=Teacher(),effect_scales=(1.,20.,4.))
        result=run_training(m,output=out,identity={'source_commit':'independent_cpu_fixture',
            'arm':'B1-effect','effect_protocol':'verified_pair_v1'},dataset_size=3,step_fn=step,
            cfg=cfg,device='cpu',seed=44,max_updates=4,stop_after=stop_after,resume=resume,
            stateful={'flow_noise':ng,'flow_time':tg})
        return m,ng,tg,result
    full,nf,tf,_=execute(tmp_path/'full')
    execute(tmp_path/'segmented',stop_after=2)
    segmented,ns,ts,result=execute(tmp_path/'segmented',resume=True)
    assert state_equal(full,segmented) and result['optimizer_step']==4
    assert torch.equal(nf.get_state(),ns.get_state()) and torch.equal(tf.get_state(),ts.get_state())
    a=torch.load(tmp_path/'full'/'step_000004.pt',weights_only=False)
    b=torch.load(tmp_path/'segmented'/'step_000004.pt',weights_only=False)
    for name in ('sampler',):assert torch.equal(a[name],b[name])
    assert a['consumed_scenes']==b['consumed_scenes']==16 and a['scheduler']==b['scheduler']
    assert torch.equal(a['rng']['cpu'],b['rng']['cpu'])
    for key,sa in a['optimizer']['state'].items():
        sb=b['optimizer']['state'][key]
        for field,value in sa.items():
            assert torch.equal(value,sb[field]) if isinstance(value,torch.Tensor) else value==sb[field]
    prior=deepcopy(segmented.state_dict());rng=torch.get_rng_state().clone()
    done=run_training(segmented,output=tmp_path/'segmented',identity={'source_commit':'independent_cpu_fixture',
        'arm':'B1-effect','effect_protocol':'verified_pair_v1'},dataset_size=3,step_fn=lambda *_:pytest.fail('terminal extra update'),
        cfg=cfg,device='cpu',seed=44,max_updates=4,resume=True,stateful={'flow_noise':ns,'flow_time':ts})
    assert done['segment_updates']==0 and done['termination_reason']=='MAX_UPDATES'
    assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(v,segmented.state_dict()[k]) for k,v in prior.items())
