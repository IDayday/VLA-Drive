"""Diagnostics retain the actual optimization graph and distinguish credit."""
from types import SimpleNamespace
import torch
from starVLA.rl.qwen_world_safety_exploration.quality import QualityLossMetrics


def test_quality_logging_does_not_change_loss_gradient_and_uses_loss_clip():
    class Base:
        ppo_clip_range=.005
        def _policy_loss(self,m,r):return m.weight.square(),{'base':m.weight.detach()}
    class Logged(QualityLossMetrics,Base):pass
    for g in (8,16):
        weight=torch.tensor(2.,requires_grad=True)
        m=SimpleNamespace(weight=weight,_probability_audit={'current':{'source_clamped_logprob':torch.zeros(g,3,8,3)}})
        records=[{'metrics':dict(no_at_fault_collisions=1.,drivable_area_compliance=1.,time_to_collision_within_bound=1.,comfort=1.,score=1.,driving_direction_compliance=1.)}for _ in range(g)]
        records[-1]['metrics']['no_at_fault_collisions']=0.;records[-1]['metrics']['score']=0.
        adv=torch.ones(g);adv[-1]=-1
        r=dict(records=records,advantages=adv,adv_mask=torch.ones(g),old_logp=torch.zeros(g,3))
        a,_=Base()._policy_loss(m,r);b,values=Logged()._policy_loss(m,r)
        assert torch.equal(a,b) and torch.equal(torch.autograd.grad(a,weight)[0],torch.autograd.grad(b,weight)[0])
        assert values['quality_strict_high_count']==g-1
        assert values['quality_positive_high_count']==g-1
        assert values['quality_active_positive_unsafe_weight']==0
        assert torch.allclose(values['quality_active_positive_high_weight'],torch.tensor((g-1)/g))
