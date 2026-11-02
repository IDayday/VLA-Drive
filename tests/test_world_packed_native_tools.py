"""RNG/stream comparison rejects contamination; original B1 export parity."""
import copy
import numpy as np
import pytest,torch
from tools.qwen_world_multiscene import native_review as N


def test_comparison_ignores_only_timing_and_checks_every_rank_rng_stream(monkeypatch):
    value=dict(path='fixture',seal=dict(phase='READY'),identity=dict(sha256='same',packed_execution=dict(micro_scenes=4)),
        common=dict(adapters=dict(x=torch.tensor([1.])),optimizer=dict(state={0:dict(step=torch.tensor(4.),exp_avg=torch.tensor([.1]))}),scheduler=dict(last_epoch=4)),
        ranks=[dict(rng=dict(python=(42,),numpy=np.array([r],dtype=np.uint32),torch=torch.tensor([r],dtype=torch.uint8),
            cuda=torch.tensor([r+1],dtype=torch.uint8),rollout_generator=torch.tensor([r+2],dtype=torch.uint8)),
            state=dict(updates=4,scene_stream=dict(cursor=128,rank_material=r),seconds_update=1.),rollout=None)for r in range(8)],counts=dict(updates=4))
    other=copy.deepcopy(value);other['ranks'][0]['state']['seconds_update']=77.
    monkeypatch.setattr(N,'load_validated',lambda path:value if path=='left'else other)
    assert N.compare('left','right')['status']=='PASS'
    other['ranks'][7]['rng']['cuda'][0]=99
    with pytest.raises(ValueError,match='rank7.all_native_RNG.cuda'):N.compare('left','right')
    other=copy.deepcopy(value);other['ranks'][4]['state']['scene_stream']['cursor']+=1
    with pytest.raises(ValueError,match='scene_stream.cursor'):N.compare('left','right')
    other=copy.deepcopy(value);other['common']['optimizer']['state'][0]['exp_avg'][0]+=.01
    with pytest.raises(ValueError,match='exp_avg'):N.compare('left','right')
    material=N.scientific_rollout(dict(scoring_seconds=1.,official_first_ttc_infraction_seconds=.4))
    assert material==dict(official_first_ttc_infraction_seconds=.4)


def test_real_scope_native_trainables_export_restore_single_B1_ODE_exact():
    torch.set_num_threads(1)
    from test_world_multiscene_prototype import make
    from world_postcot.test_scope_rl import TinyModel
    from starVLA.rl.simwam_recipe.lora_adapter import adapter_state
    from starVLA.rl.qwen_world_postcot.deployment import load_scope_trainable_state
    from starVLA.rl.qwen_world_postcot.rl_actor import install_deployment_lora
    from starVLA.rl.qwen_world_performance.ode_target import condition_for_batch
    from starVLA.model.modules.qwen_world_postcot.flow import CleanXFlowPolicy
    from starVLA.rl.simwam_recipe.contracts import seed_for
    actor,conditions,samples,_=make();cfg=actor._loss
    from test_world_safety_exploration_groups import cfg as make_cfg
    recipe=make_cfg(16);adapters=adapter_state(actor)
    with torch.random.fork_rng(devices=[]):torch.manual_seed(42);deployment=TinyModel()
    install_deployment_lora(deployment.actor,recipe['grpo']);load_scope_trainable_state(deployment,adapters,recipe['adaptation'])
    raw=condition_for_batch(actor,samples[0]['cond_bg'],1)
    root=torch.randn((1,8,3),generator=torch.Generator().manual_seed(seed_for('native_qwen_sourceflow_ODE',2101,'t0')))
    with torch.no_grad(),actor._context_call(raw)as current:
        prepared=actor._prepare(current,False)
        with torch.autocast('cpu',dtype=torch.bfloat16):native=CleanXFlowPolicy(actor.action_expert).sample_ode(prepared,root)
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
        memory=deployment.encode_condition(dict(current=True),raw['base_ego'])
        deployed=CleanXFlowPolicy(deployment.actor).sample_ode(deployment.actor.prepare_context(memory),root)
    assert torch.equal(native,deployed)and native.shape==(1,8,3)
