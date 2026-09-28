"""Run explicitly on an allocated GPU; no optimizer or real-data update."""
import os
import random
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.ddpolicy_vehicle.training_state import capture_rng, restore_rng
from starVLA.model.modules.vehicle_joint.masks import VehicleRoleScheduler


@pytest.mark.skipif(os.environ.get('DDPOLICY_TEST_CUDA')!='1', reason='Requires an explicitly allocated CUDA device')
def test_local_cuda_rng_resume_including_noise_roles_and_video():
    torch.cuda.set_device(0)
    model=SimpleNamespace(rgb_model=SimpleNamespace(rng=np.random.default_rng(100),rng_2=np.random.default_rng(200),torch_rng=torch.Generator(device='cuda').manual_seed(300)))
    noise=torch.Generator(device='cuda').manual_seed(400)
    roles=VehicleRoleScheduler(500)
    saved=capture_rng(model,noise,roles)
    active=torch.tensor([[True,True]],device='cuda')
    valid=torch.ones(1,2,8,4,dtype=torch.bool,device='cuda');valid[:,1,:,2:]=False
    def draw():
        return [random.random(),np.random.rand(),torch.randn(8),torch.randn(8,device='cuda'),
                torch.randn(8,device='cuda',generator=noise),roles.known_mask(active,valid),
                model.rgb_model.rng.random(),model.rgb_model.rng_2.random(),
                torch.randn(8,device='cuda',generator=model.rgb_model.torch_rng)]
    expected=draw();restore_rng(saved,model,noise,roles);actual=draw()
    for i,(x,y) in enumerate(zip(expected,actual)):
        if i==5: assert torch.equal(x[0],y[0]) and x[1]==y[1]
        elif isinstance(x,torch.Tensor): assert torch.equal(x,y)
        else: assert x==y
    assert 'cuda_local' in saved and 'cuda' not in saved
