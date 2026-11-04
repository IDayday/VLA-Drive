from types import SimpleNamespace
import pytest
import torch
from starVLA.model.modules.qwen_world_compare.qwen_flow_adapter import DPrefix
from starVLA.rl.d46_fpo_engineering.joint_flow import JointVerifiedActor


class Denoiser(torch.nn.Module):
    def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.4))
    def denoise_x0(self,state,sigma,prefix):return state*self.weight+sigma[:,None,None]*.03


class Actor(JointVerifiedActor):
    def __init__(self):
        torch.nn.Module.__init__(self);self.action_expert=Denoiser();self.device=torch.device('cpu')
        self.study_config={'verified_improvement':{'CFM_MC':2,'CFM_weight':.05,'ODE_last_weight':.02,'retention_weight':1.}}
        self.execution_fuse_fm_mc=False


@pytest.mark.parametrize('rescue,keep',[(False,False),(True,False),(False,True),(True,True)])
def test_joint_batch_preserves_energy_losses_and_parameter_gradient(rescue,keep):
    torch.manual_seed(41);actor=Actor();g=16
    x=torch.zeros(g,1,2);prefix=DPrefix(((x,x),),torch.ones(g,1,dtype=torch.bool),torch.ones(g,dtype=torch.long),x,('s',)*g)
    r={'fm_state':torch.randn(4,g,8,3),'fm_sigma':torch.rand(4,g)*.6+.2,'fm_target':torch.randn(4,g,8,3),
       'nominal_chain':torch.randn(1,11,8,3),'nominal_sigma_last':torch.tensor([.3])}
    if rescue:r.update(rescue_teacher=torch.randn(1,8,3),rescue_state=torch.randn(2,8,3),rescue_sigma=torch.tensor([.2,.7]),rescue_velocity_target=torch.randn(2,8,3))
    if keep:r.update(retention_target=torch.randn(1,8,3),retention_share=.25)
    results=[]
    for enabled in(False,True):
        actor.zero_grad();actor.execution_joint_flow_aux=enabled
        energy=actor.flow_energies(r,prefix);aux,metrics=actor.additional_objective(r,prefix)
        loss=energy.mean()+aux;loss.backward()
        results.append((energy.detach(),loss.detach(),actor.action_expert.weight.grad.clone()))
    for old,new in zip(*results):assert torch.allclose(old,new,atol=1e-6,rtol=1e-6)
    assert actor._joint_auxiliary is None
