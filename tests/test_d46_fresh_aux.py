"""Real native tiny-Qwen D kernel; not full IL46/GPU deployment qualification."""
import copy
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from starVLA.model.modules.qwen_world_compare.model import FourFrameworkVLA
from starVLA.rl.qwen_world_compare.deployment import prune_deployment
from starVLA.rl.qwen_world_compare.policy_adapter import CONDITION_SCHEMA
from starVLA.rl.d44_native_rl.config import defaults,source_config
from starVLA.rl.d44_native_rl.recovery import Boundary,checkpoint,material_digest
from starVLA.rl.d44_native_rl.adaptive_kl import AdaptiveKL
from starVLA.rl.d46_fresh_aux.actor import FreshAuxActor,optimizer_groups
from starVLA.rl.d46_fresh_aux.modules import attach_current_only
from starVLA.rl.d46_fresh_aux.deployment import policy_payload,restore_policy
from starVLA.rl.d46_fresh_aux.protocol import EXPERIMENT,PROTOCOL_SHA256

spec=importlib.util.spec_from_file_location('freshaux_native_oracle',Path(__file__).parent/'world_compare/test_d_qwen_flow_adapter.py')
oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)


@pytest.fixture(autouse=True)
def cpu_fixture():
    torch.set_num_threads(1);torch.manual_seed(42)


def policy():
    from types import SimpleNamespace
    qwen=oracle.model().qwen_model
    m=FourFrameworkVLA('D',config={'d':{'qualification':True,'activation_checkpointing':False}},
        train_heads=False,qwen_encoder=SimpleNamespace(model=qwen))
    return prune_deployment(m,copy_model=False)


def recipe():
    c=defaults('D4');c.update(experiment_id=EXPERIMENT,group=8,global_scenes=1)
    return c


def inputs():
    return dict(input_ids=torch.tensor([[3,5,7,9]]),attention_mask=torch.ones(1,4,dtype=torch.long))


def condition(actor):
    current=actor.action_expert.encode_current(inputs(),torch.zeros(1,8),('TRAIN_CPU_toy',))
    return dict(schema=CONDITION_SCHEMA,context=torch.zeros(8,1,1),current=current)


def behavior(actor):
    c=condition(actor);g=torch.Generator().manual_seed(51)
    with torch.no_grad():
        chain,times,deltas,steps=actor.sample_action_chain(c,8,generator=g)
        old,std=actor.transition_terms(c,chain,times,deltas)
        ref,_=actor.transition_terms(c,chain,times,deltas,use_ref=True)
    records=[]
    for i in range(8):
        nc=0. if i==0 else 1.;dac=1.;ttc=0. if i==1 else 1.;ep=.3+.1*i
        m=dict(no_at_fault_collisions=nc,drivable_area_compliance=dac,time_to_collision_within_bound=ttc,
            ego_progress=ep,comfort=1.,driving_direction_compliance=1.,score=nc*dac*(5*ep+5*ttc+2)/12)
        records.append(dict(metrics=m,non_drivable_body_by_time=[False]*41,
            official_first_at_fault_collision_seconds=.7 if i==0 else None,
            official_first_ttc_infraction_seconds=1.1 if i==1 else None))
    return dict(condition=c,chain=chain,timesteps=times,deltas=deltas,std=std,old_mean=old,
        ref_mean=ref,advantages=torch.linspace(-1,1,8),beta=.001,expect_fresh=True,epoch=1.,records=records)


def test_initializer_preserves_policy_CPU_RNG_zero_delta_B1_and_exact_scope(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import yaml
    from starVLA.rl.d46_fresh_aux.config import build_recipe,read_qualification
    monkeypatch.setenv('WORLD_COMPARE_ROOT','/mnt/project/d46-priority-rl-20261103')
    cfg=build_recipe(post_probe=True);file=tmp_path/'qualified.yaml';file.write_text(yaml.safe_dump(cfg))
    assert read_qualification(SimpleNamespace(qualification=True,config=str(file)))==cfg
    assert cfg['post_start_epoch']==0. and cfg['qualification_overrides']['formal_post_start_epoch']==1.
    with pytest.raises(ValueError,match='no formal'):read_qualification(SimpleNamespace(qualification=False,config=str(file)))
    m=policy();cpu=torch.get_rng_state().clone();old={n:p.detach().clone()for n,p in m.named_parameters()}
    with patch.object(torch.cuda,'manual_seed_all',side_effect=AssertionError('No CUDA RNG use')):
        fresh=attach_current_only(m)
    assert torch.equal(cpu,torch.get_rng_state())
    assert all(torch.equal(dict(m.named_parameters())[n],p)for n,p in old.items())
    assert set(m.readouts._modules)=={'bridge','evaluation'}
    with pytest.raises(ValueError,match='already-pruned'):attach_current_only(m)
    base=policy();original={id(p):p.detach().clone()for p in base.parameters()}
    ego=torch.zeros(1,8);noise=torch.randn(1,8,3);before=base.predict(inputs(),ego,noise)
    actor=FreshAuxActor(base,source_config(recipe()),recipe(),qualification=True)
    after=actor.action_expert.predict(inputs(),ego,noise)
    assert all(torch.equal(before[k],after[k])for k in before)
    assert not hasattr(actor.ref_action_expert,'readouts')and actor.ref_action_expert.decision.r_embeddings is None
    assert all(not p.requires_grad for p in actor.ref_action_expert.parameters())
    for p in actor.action_expert.parameters():
        if id(p)in original:assert torch.equal(p,original[id(p)])
    groups=optimizer_groups(actor,recipe());q=actor.action_expert.decision.r_embeddings
    assert [r['name']for r in groups if any(p is q for p in r['params'])]==['fresh_post_queries']
    assert actor.closure['fresh_auxiliary_protocol_sha256']==PROTOCOL_SHA256
    assert set(actor.closure['trainable_names'])=={n for n,p in actor.named_parameters()if p.requires_grad}
    assert not(set(id(p)for p in actor.parameters()if p.requires_grad)&set(id(p)for p in actor.ref_action_expert.parameters()))
    actor.action_expert.foreign=torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError,match='exact declared closure'):actor.scope_audit()
    print(json.dumps(dict(check='real_tiny_native_B1_zero_delta',normalized_max_abs=0.,trainable=actor.closure['trainable_numel'],fresh=fresh['new_parameter_numel'])))


def test_actual_mask_R_head_and_Qwen_LoRA_post_gradients():
    actor=FreshAuxActor(policy(),source_config(recipe()),recipe(),qualification=True);r=behavior(actor)
    loss,metrics,parts=actor(r,return_components=True)
    assert metrics['fresh_ratio_error']==0 and torch.isfinite(loss)
    params={n:p for n,p in actor.named_parameters()if p.requires_grad}
    grad=torch.autograd.grad(parts['auxiliary'],list(params.values()),allow_unused=True)
    norms={n:0. if g is None else float(g.norm())for(n,_),g in zip(params.items(),grad)}
    roles={key:sum(v for n,v in norms.items()if key in n)for key in ['decision.r_embeddings','readouts.bridge','readouts.evaluation','language_model']}
    assert all(v>0 for v in roles.values())
    for bf16 in (False,True):
        with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16,enabled=bf16),actor._context_call(r['condition'])as cond:
            prefix=actor._prepare(cond,False).tensors;state=r['chain'][:,-1];sigma=torch.zeros(8)
            hidden,*_=actor.action_expert.decision._candidate_suffix(state,sigma,prefix,include_r=True)
            query=actor.action_expert.decision.r_embeddings;original=query.clone();query.add_(100.)
            changed,*_=actor.action_expert.decision._candidate_suffix(state,sigma,prefix,include_r=True)
            assert torch.equal(hidden[:,:8],changed[:,:8]);assert not torch.equal(hidden[:,8:],changed[:,8:])
            query.copy_(original)
    assert all(p.grad is None for p in actor.ref_action_expert.parameters())
    print(json.dumps(dict(check='actual_native_R_replay_post_gradients',role_gradient_norm_sums=roles,postloss=float(parts['auxiliary']),fresh_error=float(metrics['fresh_ratio_error']))))


def test_real_tiny_mid_inner_native_restores_fresh_scope_Adam_and_RNG(tmp_path):
    from tools.d44_native_rl.train import SceneScheduler
    cfg=recipe();base=policy();restart=copy.deepcopy(base);actor=FreshAuxActor(base,source_config(cfg),cfg,qualification=True)
    r=behavior(actor);batch=dict(scenes=[r],behavior_update=0,global_std=1.,valid_scenes=1,epoch=1.)
    batch['digest']=material_digest(batch)
    identity=dict(schema='d44_native_rl_v1',sha256='tiny_fresh_aux_own',world=1,recipe=cfg,scope=actor.closure,qualification=True)
    opt=torch.optim.AdamW(optimizer_groups(actor,cfg),betas=(.9,.95),eps=1e-8,weight_decay=.01);scheduler=SceneScheduler(opt);generator=torch.Generator().manual_seed(55)
    s=dict(updates=0,collections=1,next_inner=0,phase='ROLLOUT_COMMITTED',scene_cursor=1,valid_scene_exposures=1,new_candidates=8,adaptive_kl=AdaptiveKL().state_dict())
    def step(model,optimizer,sched,state,material):
        optimizer.zero_grad(set_to_none=True);material['scenes'][0]['expect_fresh']=state['next_inner']==0
        value,_=model(material['scenes'][0]);value.backward();optimizer.step();sched.step()
        state['updates']+=1;state['next_inner']+=1;state['phase']='INNER_COMMITTED'
        return value.detach()
    step(actor,opt,scheduler,s,batch)
    cp=checkpoint.save(tmp_path,actor,opt,scheduler,generator,s,batch,identity,runtime=Boundary())
    expected=step(actor,opt,scheduler,s,batch);expected_params={n:p.clone()for n,p in actor.named_parameters()if p.requires_grad};expected_opt=copy.deepcopy(opt.state_dict())
    restored=FreshAuxActor(restart,source_config(cfg),cfg,qualification=True);opt2=torch.optim.AdamW(optimizer_groups(restored,cfg),betas=(.9,.95),eps=1e-8,weight_decay=.01);sched2=SceneScheduler(opt2);g2=torch.Generator()
    s2,b2=checkpoint.restore(cp,restored,opt2,sched2,g2,identity,runtime=Boundary())
    actual=step(restored,opt2,sched2,s2,b2)
    assert torch.equal(actual,expected)and sched2.last_epoch==scheduler.last_epoch==2
    for n,p in restored.named_parameters():
        if p.requires_grad:assert torch.equal(p,expected_params[n])
    for key,state in opt2.state_dict()['state'].items():
        for name,value in state.items():assert torch.equal(value,expected_opt['state'][key][name])
    assert torch.equal(generator.get_state(),g2.get_state())
    # Native retains R/head weights; deployment must explicitly drop them.
    adapters={n:p.detach().clone()for n,p in restored.named_parameters()if p.requires_grad}
    exported,receipt=policy_payload(adapters,restored.closure)
    assert receipt['R_and_readouts_excluded']and all('r_embeddings'not in n and not n.startswith('readouts.')for n in exported)
    deployed=prune_deployment(copy.deepcopy(restored.action_expert),copy_model=False)
    restore_policy(deployed,adapters,restored.closure)
    ego=torch.zeros(1,8);noise=torch.randn(1,8,3)
    full=restored.action_expert.predict(inputs(),ego,noise);pruned=deployed.predict(inputs(),ego,noise)
    assert all(torch.equal(full[k],pruned[k])for k in full)
    corrupt=dict(adapters);corrupt.pop('action_expert.decision.r_embeddings')
    with pytest.raises(ValueError,match='training parameter'):policy_payload(corrupt,restored.closure)
    print(json.dumps(dict(check='actual_tiny_native_mid_inner_restore',restored_update=1,next_inner=1,final_update=2,all_trainables_Adam_scheduler_exact=True)))
