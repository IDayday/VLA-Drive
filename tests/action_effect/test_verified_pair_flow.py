"""Production small modules on CPU fixtures; not real learned-policy evidence."""
from copy import deepcopy
from dataclasses import replace
import pytest
import torch

from starVLA.model.modules.action_effect.effect_flow_adapter import EffectFlowAdapter,EffectFlowAdapterConfig
from starVLA.model.modules.action_effect.effect_distillation import verified_pair_effect_loss
from starVLA.model.modules.action_effect.effect_model import ActionEffectModel,EffectModelConfig,FeedbackLabelSpec
from starVLA.model.framework.effect_internalized_flow import EffectInternalizedFlow
from tools.action_effect.train_effect_flow import FlowStep,FlowLossRecipe,RawActionEffectTeacher
from tools.action_effect.context_data import batch_context
from tools.action_effect.training import run_training
from test_flow_internalization import tiny_head,inputs
from test_flow_trainer import fixtures


@pytest.fixture(autouse=True)
def bounded_cpu():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def modern(arm='B1-effect'):
    adapter=EffectFlowAdapter(EffectFlowAdapterConfig(condition_dim=12,hidden_dim=16,heads=2,layers=1,effect_dim=2))
    return EffectInternalizedFlow(tiny_head(),adapter,arm=arm,effect_protocol='verified_pair_v1')


def supervision(batch):
    g=torch.Generator().manual_seed(36)
    return dict(true_delta=torch.randn(batch,2,generator=g),teacher_embedding=torch.randn(batch,16,generator=g),
        effect_valid=torch.ones(batch,2,dtype=torch.bool),effect_scales=(1.,1.))


def teacher():
    cfg=EffectModelConfig(context_dim=12,world_tokens=2,action_tokens=8,hidden_dim=16,layers=1,heads=2)
    # Explicit untrained synthetic fixture; never used as an experiment teacher.
    g1=ActionEffectModel(cfg,FeedbackLabelSpec(('score','ego_progress'),('0..1','0..1'),(1.,1.))).requires_grad_(False).eval()
    return RawActionEffectTeacher(g1)


def prepared():
    records,examples=fixtures()
    from starVLA.dataloader.foresight_dataset import decode_ego
    for e in examples:
        e['target_physical']=decode_ego(e['target_raw']).clone()
        e['effect_delta']=torch.tensor([.2,.05]);e['effect_valid']=torch.ones(2,dtype=torch.bool)
    return records,examples


def step(records,examples,model_teacher,micro=2):
    ng=torch.Generator().manual_seed(211);tg=torch.Generator().manual_seed(212)
    return FlowStep(examples,records.__getitem__,recipe=FlowLossRecipe(microbatch_scenes=micro),device='cpu',
        noise_generator=ng,time_generator=tg,context_batcher=batch_context,teacher=model_teacher,effect_scales=(1.,1.)),ng,tg


def test_all_arms_equal_allocated_initial_parameters_legacy_velocity_unchanged_and_CPU_only_init(monkeypatch):
    cfg=EffectFlowAdapterConfig(condition_dim=12,hidden_dim=16,heads=2,layers=1)
    before=torch.get_rng_state().clone()
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda,'manual_seed_all',lambda *_:pytest.fail('Adapter init touched CUDA RNG'))
        old=EffectFlowAdapter(cfg)
        with torch.device('meta'):new=EffectFlowAdapter(replace(cfg,effect_dim=2))
    assert torch.equal(before,torch.get_rng_state()) and all(p.device.type=='cpu' for p in new.parameters())
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
    assert all(not k.startswith(('effect_head.','effect_projection.')) for k in old.state_dict())
    with pytest.raises(ValueError):old(torch.zeros(1,8,12),torch.zeros(1,8,4),torch.zeros(1),return_effect=True)
    arms=[modern(a) for a in ('B0','B1-trajectory','B1-effect')]
    assert len({sum(p.numel() for p in m.trainable_parameters()) for m in arms})==1
    assert all(torch.equal(v,m.adapter.state_dict()[k]) for m in arms[1:] for k,v in arms[0].adapter.state_dict().items())


def test_verified_targets_do_not_enter_student_forward_and_gradients_reach_only_private_modules():
    m=modern();condition,pair=inputs();condition.requires_grad_();s=supervision(2)
    s['true_delta'].requires_grad_();s['teacher_embedding'].requires_grad_()
    original={k:v.clone() for k,v in m.base.state_dict().items()}
    first=m(condition,pair,torch.ones(2,dtype=torch.bool),**s)
    other=m(condition,pair,torch.ones(2,dtype=torch.bool),**{**s,'true_delta':s['true_delta']+1.,'teacher_embedding':s['teacher_embedding']+3.})
    assert torch.equal(first['prediction'],other['prediction'])
    assert torch.equal(first['student_effect'],other['student_effect'])
    assert not torch.equal(first['losses']['effect_representation'],other['losses']['effect_representation'])
    sum(first['losses'].values()).backward()
    for layer in (m.adapter.effect_head,m.adapter.effect_projection,m.adapter.context[1]):
        assert layer.weight.grad is not None and layer.weight.grad.abs().sum()>0
    assert s['true_delta'].grad is None and s['teacher_embedding'].grad is None and condition.grad is None
    assert all(p.grad is None for p in m.base.parameters())
    assert all(torch.equal(v,m.base.state_dict()[k]) for k,v in original.items())
    with pytest.raises(ValueError,match='never an endpoint teacher'):
        m(condition,pair,torch.ones(2,dtype=torch.bool),teacher=teacher(),**s)


def test_pair_teacher_receives_exact_verified_physical_endpoints_once_and_microbatch_objective_matches():
    records,examples=prepared();a=modern();b=deepcopy(a);ta,tb=teacher(),teacher();seen=[]
    hook=ta.model.register_forward_pre_hook(lambda _,args:seen.append((args[1].clone(),args[2].clone())))
    # Legacy arbitrary-endpoint teacher forward must never be called.
    ta.forward=lambda *_:pytest.fail('Queried arbitrary endpoint proxy')
    sa,nga,tga=step(records,examples,ta,1);sb,ngb,tgb=step(records,examples,tb,4)
    ra=sa(a,[0,1,2,3],0);rb=sb(b,[0,1,2,3],0);hook.remove()
    assert len(seen)==1 and ra['verified_pair_teacher_calls']==1 and ra['effect_supervised_scenes']==2
    assert torch.equal(seen[0][0],torch.stack([records[str(i)]['physical'] for i in (0,2)]))
    assert torch.equal(seen[0][1],torch.stack([examples[i]['target_physical'] for i in (0,2)]))
    assert torch.equal(nga.get_state(),ngb.get_state()) and torch.equal(tga.get_state(),tgb.get_state())
    for pa,pb in zip(a.adapter.parameters(),b.adapter.parameters()):
        assert torch.allclose(pa.grad,pb.grad,atol=3e-6,rtol=3e-5)
    assert all(p.grad is None for p in ta.parameters())
    assert abs(ra['effect_representation']-rb['effect_representation'])<1e-6


def test_direction_gap_cost_and_global_microbatch_DDP_reductions_and_empty_masks():
    p=torch.zeros(2,1,requires_grad=True);embedding=torch.ones(2,3,requires_grad=True)
    args=dict(true_delta=torch.tensor([[.95],[.02]]),teacher_embedding=torch.zeros(2,3),
        effect_valid=torch.ones(2,1,dtype=torch.bool),repair_valid=torch.ones(2,dtype=torch.bool),effect_scales=(1.,))
    full=verified_pair_effect_loss(p,embedding,**args)
    grad=torch.autograd.grad(full['effect_direction'],p,retain_graph=True)[0]
    assert grad[0]<0 and torch.allclose(grad[0]/grad[1],torch.tensor([.95/.02]),rtol=1e-6)
    rank=[]
    for i in range(2):
        subset={k:(v[i:i+1] if isinstance(v,torch.Tensor) else v) for k,v in args.items()}
        rank.append(verified_pair_effect_loss(p[i:i+1],embedding[i:i+1],**subset,
            global_denominator=2,direction_global_denominator=float(args['true_delta'].abs().sum()),
            representation_global_denominator=6,world_size=2))
    for name in full:assert torch.allclose((rank[0][name]+rank[1][name])/2,full[name])
    empty=verified_pair_effect_loss(p,embedding,**{**args,'repair_valid':torch.zeros(2,dtype=torch.bool)},
        global_denominator=0,direction_global_denominator=0,representation_global_denominator=0)
    assert all(x.requires_grad and x.item()==0 for x in empty.values())


def test_export_strips_all_effect_modules_exact_full_Euler10_no_teacher_or_aux_calls():
    m=modern().eval();condition,pair=inputs()
    with torch.no_grad():m.adapter.output.weight.add_(.001)
    exported=m.adapter.velocity_export()
    assert exported['config']['effect_dim']==0 and not any(k.startswith(('effect_head.','effect_projection.')) for k in exported['state_dict'])
    clean=EffectFlowAdapter(EffectFlowAdapterConfig(**exported['config']));clean.load_state_dict(exported['state_dict'],strict=True)
    deployed=EffectInternalizedFlow(deepcopy(m.base.action_head),clean,arm=m.arm).eval()
    h1=m.adapter.effect_head.register_forward_pre_hook(lambda *_:pytest.fail('Effect head ran during deployment'))
    h2=m.adapter.effect_projection.register_forward_pre_hook(lambda *_:pytest.fail('Effect projection ran during deployment'))
    assert torch.equal(m.predict_action(condition,initial_noise=pair.noise),deployed.predict_action(condition,initial_noise=pair.noise))
    h1.remove();h2.remove()


def test_verified_training_resume_preserves_next_update_and_both_flow_streams(tmp_path):
    records,examples=prepared();a=modern();b=deepcopy(a);cfg={'training':{'max_updates':4,
        'learning_rate':.001,'weight_decay':.01,'warmup_updates':1,'batch_scenes':4,
        'gradient_clip':1.,'validation_every':2}}
    sa,nga,tga=step(records,examples,teacher());sb,ngb,tgb=step(records,examples,teacher())
    def run(m,s,n,t,path,resume=False,stop_after=None):
        return run_training(m,output=path,identity={'source_commit':'CPU-fixture','arm':'B1-effect','effect_protocol':'verified_pair_v1'},
            dataset_size=4,step_fn=s,cfg=cfg,device='cpu',seed=42,max_updates=4,stop_after=stop_after,
            resume=resume,stateful={'flow_noise':n,'flow_time':t})
    run(a,sa,nga,tga,tmp_path/'full');run(b,sb,ngb,tgb,tmp_path/'paused',stop_after=2)
    c=modern();sc,ngc,tgc=step(records,examples,teacher());run(c,sc,ngc,tgc,tmp_path/'paused',resume=True)
    assert all(torch.equal(v,c.adapter.state_dict()[k]) for k,v in a.adapter.state_dict().items())
    assert torch.equal(nga.get_state(),ngc.get_state()) and torch.equal(tga.get_state(),tgc.get_state())
