import contextlib
from types import SimpleNamespace

import pytest
import torch

from starVLA.dataloader.foresight_dataset import decode_ego
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead
from tools.full_foresight.fm_step_sweep import sample_encoded, validate_steps, STEPS


class EchoEncoder(torch.nn.Module):
    def forward(self, actions, time):
        return actions


class KnownVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.coefficient=torch.nn.Parameter(torch.tensor(.25))
        self.times=[]

    def forward(self, hidden_states, encoder_hidden_states, timestep):
        self.times.append(int(timestep[0]))
        return self.coefficient*hidden_states+timestep[:,None,None].float()/1000


def model_fixture():
    # Execute the real production Euler loop without allocating the 803M DiT.
    head=FlowmatchingActionHead.__new__(FlowmatchingActionHead)
    torch.nn.Module.__init__(head)
    head.config=SimpleNamespace(action_horizon=8,action_dim=4,add_pos_embed=False)
    head.num_inference_timesteps=10;head.num_timestep_buckets=1000
    head.qwen_proj=torch.nn.Identity();head.action_encoder=EchoEncoder()
    head.model=KnownVelocity();head.action_decoder=torch.nn.Identity()
    return SimpleNamespace(action_model=head,amp=contextlib.nullcontext)


@pytest.mark.parametrize('n',STEPS)
def test_original_euler_times_dt_and_noise_are_preserved(n):
    model=model_fixture();noise=torch.randn(1,8,4,generator=torch.Generator().manual_seed(42))
    original=noise.clone();parameter=model.action_model.model.coefficient.detach().clone()
    encoded={'action_queries':torch.zeros(1,8,4)}
    actual=sample_encoded(model,encoded,noise,n)
    expected=noise.clone()
    for j in range(n):
        expected=expected+(1/n)*(.25*expected+int((j/n)*1000)/1000)
    torch.testing.assert_close(actual,decode_ego(expected),rtol=0,atol=0)
    assert model.action_model.model.times==[int((j/n)*1000) for j in range(n)]
    assert torch.equal(noise,original)
    assert torch.equal(model.action_model.model.coefficient,parameter)
    assert model.action_model.num_inference_timesteps==10


def test_reference_uses_exact_existing_action_head():
    model=model_fixture();noise=torch.randn(1,8,4)
    encoded={'action_queries':torch.zeros(1,8,4)}
    direct=decode_ego(model.action_model.predict_action(encoded['action_queries'],initial_noise=noise))
    actual=sample_encoded(model,encoded,noise,10)
    assert torch.equal(actual,direct)


def test_integrator_default_restored_even_on_exception():
    model=model_fixture()
    with pytest.raises(ValueError):
        sample_encoded(model,{'action_queries':torch.zeros(1,8,4)},torch.zeros(2,8,4),30)
    assert model.action_model.num_inference_timesteps==10


@pytest.mark.parametrize('steps',[[1,2], [10,10], [0,10], [10,31], [True,10], [10,1.5]])
def test_invalid_scans_are_rejected(steps):
    with pytest.raises(ValueError):validate_steps(steps)


def test_repeated_scans_do_not_chain_the_previous_solution():
    model=model_fixture();noise=torch.randn(1,8,4)
    encoded={'action_queries':torch.zeros(1,8,4)}
    first=sample_encoded(model,encoded,noise,30)
    sample_encoded(model,encoded,noise,1)
    repeated=sample_encoded(model,encoded,noise,30)
    assert torch.equal(first,repeated)
