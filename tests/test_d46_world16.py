"""CPU arithmetic/material qualification; not a GPU/DDP world16 certificate."""
import copy
from types import SimpleNamespace
import pytest
import torch
from starVLA.rl.d44_native_rl.config import defaults46,source_config
from starVLA.rl.d44_native_rl.recovery import material_digest
from starVLA.rl.d44_native_rl.safety_credit import raw_credit
from starVLA.rl.d44_native_rl.gaussian_objective import gaussian_terms,clipped_loss
from starVLA.rl.simwam_recipe.vendor.action_math import ActionGRPOMixin
from starVLA.rl.simwam_recipe.vendor.scheduler_continuous import WanContinuousFlowMatchScheduler
from starVLA.rl.d46_world16.planning import rank_slots,fixed_scene_rng_plan,assert_scene_rng_end,reshard_pending_for_qualification


class SourceSampler(ActionGRPOMixin):
    def __init__(self):
        self.device=torch.device('cpu');self.torch_dtype=self.grpo_action_dtype=torch.float32
        self.action_expert=SimpleNamespace(action_dim=3)
        self.infer_action_scheduler=WanContinuousFlowMatchScheduler(1000,shift=5.)
        self.configure_grpo(source_config(defaults46('D0'))['grpo'])
    def _configure_action_grpo(self,_):self.action_lora_enabled=False
    def action_velocity(self,x,t,cond,use_ref=False):return .3*x+.01*t[:,None,None]/1000


def test_virtual_rng_matches_actual_source_W3_and_fixed_MC_draws():
    actor=SourceSampler();condition={'context':torch.zeros(16,1,1)}
    for arm in ('D0','D1','D5','D3_FPO','D3_AWM'):
        generator=torch.Generator().manual_seed(420003);before=generator.get_state().clone()
        global_before=torch.get_rng_state().clone();plan=fixed_scene_rng_plan(before,arm)
        assert torch.equal(global_before,torch.get_rng_state())
        originals=[]
        for j in range(8):
            originals.append(actor.sample_action_chain(condition,8,generator=generator)[0])
            if arm in ('D3_FPO','D3_AWM'):
                torch.rand((4,16),generator=generator);torch.randn((4,16,8,3),generator=generator)
            assert_scene_rng_end(generator,plan['ends'][j])
        assert torch.equal(generator.get_state(),plan['next_virtual_lane_state'])
        # Simulate two physical ranks owning the original first/last4 scenes.
        for half in range(2):
            for j in range(half*4,half*4+4):
                replay=torch.Generator();replay.set_state(plan['starts'][j])
                assert torch.equal(actor.sample_action_chain(condition,8,generator=replay)[0],originals[j])
        with pytest.raises(ValueError,match='differs'):assert_scene_rng_end(torch.Generator().manual_seed(1),plan['ends'][0])
    with pytest.raises(ValueError,match='data-dependent'):fixed_scene_rng_plan(before,'D2')


def test_global64_padding_Safety_std_and_actual_Gaussian_loss_gradient_target():
    torch.manual_seed(42);factors=torch.ones(64,16,6);factors[...,3]=torch.rand(64,16);factors[:20,0,0]=0
    raw,_=raw_credit(factors);valid=torch.arange(64)<56
    def std(world):
        total=torch.zeros(3,dtype=torch.float64)
        for rank in range(world):
            idx=list(rank_slots(rank,world));r=raw[idx];v=valid[idx]
            total+=torch.stack(((r*v[:,None]).sum(),(r.square()*v[:,None]).sum(),v.sum().double()*16))
        return (total[1]/total[2]-(total[0]/total[2]).square()).sqrt()
    torch.testing.assert_close(std(8),std(16),atol=2e-15,rtol=0)
    advantage=(raw/(std(8)+1e-8)).clamp(-5,5).float()
    original=torch.randn(64,16,3,8,3)*.01;features=torch.randn_like(original);nxt=original+torch.randn_like(original)*.05
    def grad(world):
        rankgrads=[]
        for rank in range(world):
            theta=torch.tensor([.0002,-.0003,.0001],requires_grad=True);total=theta.sum()*0
            for slot in rank_slots(rank,world):
                mean=original[slot]+features[slot]*theta
                ratio,kl=gaussian_terms(mean,original[slot],original[slot],torch.tensor([.05,.04,.03]),nxt[slot])
                pg,_=clipped_loss(ratio,advantage[slot],.1)
                total+=(pg+.001*kl.mean())*(world/int(valid.sum())if valid[slot]else 0.)
            rankgrads.append(torch.autograd.grad(total,theta)[0])
        return torch.stack(rankgrads).mean(0)
    g8,g16=grad(8),grad(16)
    torch.testing.assert_close(g8,g16,atol=2e-6,rtol=2e-6)
    assert sorted(i for rank in range(16)for i in rank_slots(rank,16))==list(range(64))
    print({'source_Gaussian_gradient_max_abs_world8_vs16':float((g8-g16).abs().max()),'not_bitwise_claim':True})


def test_pending8_to16_material_shapes_counters_and_RNG_roundtrip(tmp_path):
    cfg=defaults46('D0');identity=dict(recipe=cfg,world=8,sha256='parent-world8-test')
    state=dict(updates=1,collections=1,scene_cursor=64,next_inner=1,phase='INNER_COMMITTED',valid_scene_exposures=64,raw_scene_exposures=64,
      new_candidates=1024,valid_PG_candidates=1024,extra_B1_queries=0,adaptive_G32_queries=0,padding_scene_exposures=0,adaptive_kl={'collections':0},evaluation_gates=[])
    owns=[]
    for rank in range(8):
        scenes=[]
        for j in range(8):
            slot=j*8+rank;chain=torch.full((16,11,8,3),slot/100.)
            scenes.append(dict(condition={'token':str(slot)},chain=chain,old_mean=torch.full((16,3,8,3),slot/100.),advantages=torch.linspace(-1,1,16),behavior_version={'run':identity['sha256'],'update':0}))
        batch=dict(scenes=scenes,behavior_update=0,global_std=.4,valid_scenes=64,epoch=.1);batch['digest']=material_digest(batch)
        owns.append(dict(state=copy.deepcopy(state),rollout=batch,identity_sha256=identity['sha256'],rng={'rollout_generator':torch.Generator().manual_seed(420000+rank).get_state(),'torch':torch.get_rng_state()}))
    migrated=reshard_pending_for_qualification(owns,identity)
    assert migrated['resumable_native_checkpoint']is False
    for p in migrated['ranks']:
        assert p['state']==state and len(p['rollout']['scenes'])==4
        assert [int(x['condition']['token'])for x in p['rollout']['scenes']]==list(p['global_slots'])
        assert {x['behavior_version']['run']for x in p['rollout']['scenes']}=={identity['sha256']}
        torch.save(p,tmp_path/f"rank{p['rank']}.pt")
        after=torch.load(tmp_path/f"rank{p['rank']}.pt",weights_only=False)
        assert material_digest(after['rollout'])==p['rollout']['digest']
        assert torch.equal(after['original_global_RNG_bundle']['rollout_generator'],owns[p['source_rank']]['rng']['rollout_generator'])
    bad=copy.deepcopy(owns);bad[0]['rollout']['scenes'][0]['old_mean'][0,0,0,0]+=1
    with pytest.raises(ValueError,match='probabilities'):reshard_pending_for_qualification(bad,identity)
    bad=copy.deepcopy(owns);bad[2]['state']['scene_cursor']+=64
    with pytest.raises(ValueError,match='counter'):reshard_pending_for_qualification(bad,identity)
