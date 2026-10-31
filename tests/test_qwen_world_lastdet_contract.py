"""CPU actual-source collection, W2 PG/anchor and native replay boundaries."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from starVLA.rl.qwen_world_postcot.scope_contracts import MEMORY,expected_scope_recipe
from starVLA.rl.qwen_sourceflow.actor import NativeRecipeActor
from starVLA.rl.qwen_sourceflow.collection import collect
from starVLA.rl.qwen_world_research_algorithms.lastdet_entry import LastDetRuntimeMixin,LASTDET_PROTOCOL
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss
from starVLA.rl.simwam_recipe.contracts import registered_stochastic_steps,registered_stochastic_steps_for_recipe,LASTDET_SCHEMA
from starVLA.rl.simwam_recipe.collection import finalize_advantages
from starVLA.rl.simwam_recipe.recovery_state import validate_boundary,validate_ranks,rollout_digest
from starVLA.rl.simwam_recipe.vendor.scheduler_continuous import WanContinuousFlowMatchScheduler


def recipe(lastdet=True):
    cfg=expected_scope_recipe(MEMORY)
    cfg['research']=dict(arm='PPO_LASTDET' if lastdet else 'PPO_FULL',policy_objective='ppo',data_policy='uniform_full')
    if lastdet:
        cfg['research']['sampler_ablation']=LASTDET_SCHEMA
        cfg['grpo']['sample'].update(train_sde_steps=2,sde_exclude_last_step=True,sde_step_indices=[7,8])
    return cfg


class TinyActor(NativeRecipeActor):
    """Use original source sampler/logprob audit, replacing only velocity net."""
    def __init__(self,cfg):
        torch.nn.Module.__init__(self);self.p=torch.nn.Parameter(torch.tensor(.02))
        self.device=torch.device('cpu');self.torch_dtype=self.grpo_action_dtype=torch.float32
        self.action_expert=SimpleNamespace(action_dim=3);self.ref_action_expert=SimpleNamespace(training=False)
        self.infer_action_scheduler=WanContinuousFlowMatchScheduler(1000,shift=5.)
        sample=cfg['grpo']['sample'];self.grpo_num_inference_steps=10;self.grpo_infer_shift=None
        self.grpo_sample_noise_std=self.grpo_logprob_noise_std=.1;self.grpo_anneal_noise=True
        self.grpo_min_std=.001;self.grpo_randn_clip=5.;self.grpo_logprob_clamp_mode='relative';self.grpo_logprob_clamp_width=7.
        self.grpo_sde_mode='rigorous';self.grpo_noise_level=.1;self.grpo_sigma_max_guard=.99
        self.grpo_train_sde_steps=sample['train_sde_steps'];self.grpo_sde_step_indices=sample['sde_step_indices']
        self.grpo_sde_exclude_last_step=sample['sde_exclude_last_step'];self.grpo_merge_sde_window=False;self.grpo_final_action_clip=None
        self.grpo_kl_beta=50.;self.grpo_kl_type='x';self.grpo_bc_coeff=.1;self.grpo_denoising_discount=1.
        self._noise_audit=[];self._probability_audit={};self.record_sampling_terms=True;self.sampling_terms=[];self.velocity_steps=[]
        self._loss=ResearchLoss(cfg);self.eval()

    def materialize(self,condition):
        return condition if 'context' in condition else dict(context=torch.zeros(8,1,1))

    def action_velocity(self,state,timestep,condition,use_ref=False):
        self.velocity_steps.append((bool(use_ref),float(timestep[0])))
        return .17*state+(self.p*0 if use_ref else self.p)

    def kl_weight_matrix(self,*,horizon,action_dim,device,dtype):
        return torch.tensor([.2,1.,1.],device=device,dtype=dtype)[None].expand(horizon,action_dim)


class Service:
    protocol=dict(sha256='official')
    def __init__(self,rank=0):self.rank=rank
    def score(self,value):
        token,log,requested=value;out=[]
        for i in range(8):
            ep=.1+i*.08+self.rank*.001;score=(5*ep+7)/12
            out.append(dict(status='OK',token=token,log=log,requested_trajectory=requested[i],scoring_seconds=0.,
                metrics=dict(score=score,no_at_fault_collisions=1.,drivable_area_compliance=1.,
                    time_to_collision_within_bound=1.,ego_progress=ep,comfort=1.),
                readouts=dict(canonical_PDMS=score)))
        return out


def descriptor(rank):
    return dict(schema='world_postcot_trainable_memory_observation_v1',token=f'scene{rank}',identity='condition',repeats=8)


def collected(cfg,rank=0):
    actor=TinyActor(cfg);condition=descriptor(rank)
    out=collect(actor,condition,condition['token'],f'log{rank}',Service(rank),torch.Generator().manual_seed(42+rank),
        variant='R_CANONICAL',version=dict(run='run',update=0,adapter_sha256='same'),recipe=cfg if cfg['research']['arm']=='PPO_LASTDET' else None)
    return actor,out


def test_W2_registration_is_exact_and_legacy_variant_windows_unchanged():
    c=recipe();assert registered_stochastic_steps_for_recipe(c)==[7,8]
    for v in ('R_SOURCE','R_CANONICAL','R_CANONICAL_X1','R_CANONICAL_X1_SAFETY25'):
        assert registered_stochastic_steps(v)==[7,8,9]
        assert registered_stochastic_steps_for_recipe(dict(variant=v))==[7,8,9]
    assert registered_stochastic_steps_for_recipe(dict(variant='R_CANONICAL_LAST_ODE'))==[6,7,8]
    for change in ('arm','schema','indices','G','mix'):
        bad=copy.deepcopy(c)
        if change=='arm':bad['research']['arm']='PPO_FULL'
        if change=='schema':bad['research']['sampler_ablation']='unknown'
        if change=='indices':bad['grpo']['sample']['sde_step_indices']=[6,7,8]
        if change=='G':bad['grpo']['sample']['group_size']=16
        if change=='mix':bad['research']['noise_level_ablation']='rigorous_eta003_v1'
        with pytest.raises(ValueError,match='exact independently'):registered_stochastic_steps_for_recipe(bad)


def test_source_collect_W2_qualification_replay_and_loss_never_score_step9(monkeypatch):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    cfg=recipe();actor,r=collected(cfg)
    assert r['old_logp'].shape==(8,2) and r['stoch_idx'].tolist()==[7,8]
    assert r['qualification_sampling_check']['status']=='PASS' and len(actor.sampling_terms)==2
    assert r['old_probability']['source_clamped_logprob'].shape==(8,2,8,3)
    finalize_advantages(r);before=rollout_digest(r);actor.velocity_steps=[]
    for _ in range(4):
        loss,metrics=actor._loss._policy_loss(actor,r);loss.backward()
        assert torch.isfinite(actor.p.grad) and actor.p.grad.abs()>0
        assert metrics['kl_loss']>0 and 'step7_ratio_mean' in metrics and 'step8_ratio_mean' in metrics
        assert 'step9_ratio_mean' not in metrics and rollout_digest(r)==before
        actor.p.grad=None
    expected=set(float(r['timesteps'][k]) for k in (7,8))
    assert {t for ref,t in actor.velocity_steps}==expected and len(actor.velocity_steps)==4*4
    old,legacy=collected(recipe(False));assert legacy['old_logp'].shape==(8,3)
    assert legacy['stoch_idx'].tolist()==[7,8,9] and len(old.sampling_terms)==3
    # Default historical native collector refuses W2 without its new recipe entry.
    with pytest.raises(ValueError,match='Invalid native source chain'):
        collect(TinyActor(cfg),descriptor(0),'scene0','log0',Service(),torch.Generator().manual_seed(42),
            variant='R_CANONICAL',version=dict(run='run',update=0))


def test_native_W2_torch_roundtrip_all_inner_boundaries_ranks_and_shape_rejection(monkeypatch,tmp_path):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    cfg=recipe();rollouts=[collected(cfg,rank)[1] for rank in range(8)]
    rewards=torch.cat([r['rewards'].double() for r in rollouts])
    import starVLA.rl.simwam_recipe.source_advantages as source
    monkeypatch.setattr(source.dist,'is_available',lambda:True);monkeypatch.setattr(source.dist,'is_initialized',lambda:True)
    monkeypatch.setattr(source.dist,'get_world_size',lambda:8)
    def gather(out,value):
        for rank in range(8):out[rank].copy_(rewards[rank:rank+1])
    monkeypatch.setattr(source.dist,'all_gather',gather)
    identity=dict(world=8,sha256='run',schema='world_postcot_cleanx_scope_RL_v1',recipe=cfg,projected_condition_identity='condition')
    state=dict(updates=0,collections=1,next_inner=0,phase='ROLLOUT_COMMITTED',scene_cursor=8,raw_scene_exposures=8,new_candidates=64)
    summaries=[]
    for rank,r in enumerate(rollouts):
        finalize_advantages(r);path=tmp_path/f'rank{rank}.pt';torch.save(r,path);saved=torch.load(path,weights_only=False)
        assert rollout_digest(saved)==rollout_digest(r)
        assert all(np.array_equal(x['requested_trajectory'],y['requested_trajectory']) for x,y in zip(saved['records'],r['records']))
        for inner in range(4):
            s=dict(state,updates=inner,next_inner=inner,phase='INNER_COMMITTED' if inner else 'ROLLOUT_COMMITTED')
            validate_boundary(s,saved,identity,optimizer=dict(state={0:dict(step=inner)},param_groups=[dict(params=[0])]),scheduler=dict(last_epoch=inner))
        summaries.append(validate_boundary(state,saved,identity))
    validate_ranks(summaries)
    for field in ('old_logp','stoch_idx','probability'):
        bad=copy.deepcopy(rollouts[0])
        if field=='old_logp':bad[field]=torch.zeros(8,3)
        if field=='stoch_idx':bad[field]=torch.tensor([8,9])
        if field=='probability':bad['old_probability']['source_clamped_logprob']=torch.zeros(8,3,8,3)
        with pytest.raises(ValueError):validate_boundary(state,bad,identity)
    old_identity=copy.deepcopy(identity);old_identity['recipe']=recipe(False)
    with pytest.raises(ValueError,match='old_logp'):validate_boundary(state,rollouts[0],old_identity)
    bad=copy.deepcopy(summaries);bad[0]['advantages'][0]+=.1
    with pytest.raises(ValueError,match='globally finalized'):validate_ranks(bad)


def test_new_entry_uses_registered_collector_and_records_W2_anchor_identity(monkeypatch):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    class Base:
        def recipe_identity(self,*args):return dict(original=True)
    class Runtime(LastDetRuntimeMixin,Base):pass
    rt=Runtime();rt.cfg=recipe();actor=TinyActor(rt.cfg)
    r=rt.collect(actor,descriptor(0),'scene0','log0',Service(),torch.Generator().manual_seed(42),
        variant='R_CANONICAL',version=dict(run='run',update=0))
    assert r['old_logp'].shape==(8,2)
    identity=rt.recipe_identity(rt.cfg,None,None)
    assert identity['sampler_window_protocol']==LASTDET_PROTOCOL and 'step9 anchor is removed' in LASTDET_PROTOCOL['anchor']
