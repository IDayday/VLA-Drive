"""Canonical-quality credit: bound, honest signal, and actual native hooks."""
import copy
import pytest
import torch

from test_qwen_world_fpo_training import recipe as original_recipe,collected,state,identity as old_identity
from starVLA.rl.qwen_world_research_algorithms import safety_feasible_score as q
from starVLA.rl.qwen_world_research_algorithms import safety_efficiency as se
from starVLA.rl.qwen_world_research_algorithms.safety_feasible_score_entry import SafetyFeasibleScoreRuntimeMixin
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss,policy_gradient_objective
from starVLA.rl.simwam_recipe.recovery_state import rollout_digest


def recipe():
    cfg=original_recipe();cfg['research'].pop('fpo')
    cfg['research'].update(arm=q.ARM,structured_advantage=q.SCHEMA)
    return cfg


def identity():
    result=old_identity();result.pop('fpo_protocol')
    result.update(recipe=recipe(),structured_advantage_protocol=copy.deepcopy(q.PROTOCOL))
    return result


def test_quality_canonical_order_strict_mixed_bound_and_no_fake_fail_signal():
    x=torch.ones(8,8,5,dtype=torch.float64)
    x[0,:,3]=1.;x[0,::2,4]=0.  # All EP1, only Comfort differs.
    x[1,:,0]=0.;x[1,:,3]=torch.linspace(0.,1.,8);x[1,::2,4]=0.  # Allfail costtie.
    x[2,:,0]=0.;x[2,4:,1]=0.  # Original least-cost fallback still works.
    old=se.safety_efficiency_parts(x[:,:,:4]);new=q.quality_parts(x)
    assert not old['A_raw'][0].any() and new['A_raw'][0].any()
    assert not new['A_raw'][1].any()
    torch.testing.assert_close(new['A_raw'][2],old['A_raw'][2],atol=0,rtol=0)
    torch.testing.assert_close(new['A_raw'].sum(1),torch.zeros(8,dtype=torch.float64),atol=1e-14,rtol=0)
    # Adversarial endpoint qualities prove the preserved bound for every m.
    for m in range(1,8):
        z=torch.ones(1,8,5,dtype=torch.float64);z[:,m:,2]=0
        z[0,0,3:]=0.;parts=q.quality_parts(z)
        gap=parts['A_raw'][0,:m].min()-parts['A_raw'][0,m:].max()
        assert gap>=1/m-1e-14
        assert bool((parts['A_raw'][0,m:]<0).all())
        score=(5*z[0,:m,3]+5+2*z[0,:m,4])/12
        quality=(5*z[0,:m,3]+2*z[0,:m,4])/7
        assert torch.equal(torch.argsort(score),torch.argsort(quality))
    # EP+.4C loses the same bound: this is CPU-only, never the pilot formula.
    z=torch.ones(7,2,dtype=torch.float64);z[0]=0
    unnormalized=z[:,0]+.4*z[:,1]
    assert float(1+unnormalized[0]-unnormalized.mean())<0
    # Fresh surrogate leaf gradient shows real measured Comfort contrast,
    # not an assertion about actual World parameter-gradient scale.
    a=new['A_raw'][0].float();logp=torch.zeros(8,3,requires_grad=True)
    loss,_=policy_gradient_objective(logp,torch.zeros_like(logp),a[:,None].expand(8,3),torch.ones(8),algorithm='ppo',clip_range=.005)
    loss.backward()
    assert logp.grad[:,0][a>0].lt(0).all() and logp.grad[:,0][a<0].gt(0).all()


def test_actual_eight_rank_collect_finalize_four_inner_native_recovery(monkeypatch,tmp_path):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    pairs=[collected(rank) for rank in range(8)]
    for rank,(_,out) in enumerate(pairs):
        for i,row in enumerate(out['records']):
            m=row['metrics'];m['comfort']=float(i%2)
            m['score']=m['no_at_fault_collisions']*m['drivable_area_compliance']*(5*m['ego_progress']+5*m['time_to_collision_within_bound']+2*m['comfort'])/12
            row['readouts']['canonical_PDMS']=m['score']
        out['rewards']=se.canonical_components(out['records'])['canonical_rewards']
    materials=[q.components_from_records(out['records']) for _,out in pairs]
    rewards=torch.cat([m['canonical_rewards'].double() for m in materials])
    components=torch.cat([m['components'] for m in materials])
    global_material=torch.cat((rewards[:,:,None],components),-1)
    monkeypatch.setattr(q.dist,'is_available',lambda:True);monkeypatch.setattr(q.dist,'is_initialized',lambda:True)
    monkeypatch.setattr(q.dist,'get_world_size',lambda:8)
    def gather(values,local):
        for rank in range(8):values[rank].copy_(global_material[rank:rank+1])
    monkeypatch.setattr(q.dist,'all_gather',gather)
    class Runtime(SafetyFeasibleScoreRuntimeMixin):pass
    runtime=Runtime();summaries=[];rng=torch.random.get_rng_state().clone()
    for rank,(actor,out) in enumerate(pairs):
        runtime.finalize_advantages(out)
        assert out['stoch_idx'].tolist()==[7,8,9] and out['old_logp'].shape==(8,3)
        path=tmp_path/f'rank{rank}.pt';torch.save(out,path);saved=torch.load(path,weights_only=False)
        for inner in range(4):
            summary=runtime.validate_boundary(state(inner),saved,identity(),
                optimizer=dict(state={0:dict(step=inner)},param_groups=[dict(params=[0])]),scheduler=dict(last_epoch=inner))
            if inner==0:summaries.append(summary)
        for field in ('components',*q.PARTS):
            bad=copy.deepcopy(saved)
            if bad['structured_advantage'][field].dtype==torch.bool:
                bad['structured_advantage'][field].reshape(-1)[0].logical_not_()
            else:bad['structured_advantage'][field].reshape(-1)[0]+=.1
            with pytest.raises(ValueError,match='component changed'):q.validate_boundary(state(),bad,identity())
        if rank==2:
            assert out['advantages'].any()  # Old source rank2 is allEP1; now C varies.
            actor._loss=ResearchLoss(recipe());before=rollout_digest(saved)
            for inner in range(4):
                actor.p.grad=None;loss,metrics=actor._loss._policy_loss(actor,saved)
                loss.backward();assert torch.isfinite(actor.p.grad) and actor.p.grad.abs()>0
                with torch.no_grad():actor.p.sub_(1e-5*actor.p.grad)
                assert rollout_digest(saved)==before
        if rank==5:assert not out['adv_mask'].any()  # NC0 deficit tie not magically solved.
        with pytest.raises(ValueError,match='once'):runtime.finalize_advantages(out)
    runtime.validate_ranks(summaries)
    bad=copy.deepcopy(summaries);bad[1]['advantages'][0]+=.1
    with pytest.raises(ValueError,match='globally finalized'):runtime.validate_ranks(bad)
    bad=copy.deepcopy(summaries);bad[2]['quality_protocol']['quality']='EP only'
    with pytest.raises(ValueError,match='rank protocol'):runtime.validate_ranks(bad)
    assert torch.equal(rng,torch.random.get_rng_state())


def test_recipe_identity_isolated_and_old_credit_not_reinterpreted():
    cfg=recipe();assert q.validate_recipe(cfg) is cfg
    for field,value in [('structured_advantage',se.STRUCTURED_SCHEMA),('arm','PPO_SAFETY_EP'),
                        ('fpo',{}),('success_memory',{}),('warmstart',{})]:
        bad=copy.deepcopy(cfg);bad['research'][field]=value
        with pytest.raises(ValueError):q.validate_recipe(bad)
    bad=copy.deepcopy(cfg);bad['grpo']['sample']['train_sde_steps']=2
    with pytest.raises(ValueError):q.validate_recipe(bad)
    bad=copy.deepcopy(cfg);bad['grpo']['train']['kl_beta']=40
    with pytest.raises(ValueError):q.validate_recipe(bad)
    class Parent:
        def load_recipe(self,args):return cfg
        def recipe_identity(self,cfg,args,condition_sha):return dict(recipe=cfg)
    class Runtime(SafetyFeasibleScoreRuntimeMixin,Parent):pass
    runtime=Runtime();assert runtime.load_recipe(None) is cfg
    assert runtime.recipe_identity(cfg,None,'cond')['structured_advantage_protocol']==q.PROTOCOL
    bad=identity();bad['structured_advantage_protocol']=se.STRUCTURED_PROTOCOL
    with pytest.raises(ValueError,match='identity/world'):q.validate_boundary(state(),{},bad)
