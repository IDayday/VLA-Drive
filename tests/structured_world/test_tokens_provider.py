import pytest
import torch
from starVLA.model.modules.structured_world.tokens import token_positions, insert_world_tokens
from starVLA.model.modules.structured_world.providers import GeometricBEVProvider, validate_feature_cache
from starVLA.model.modules.structured_world.contracts import ModelInputs


def test_token_validation_and_native_positions():
    ids=torch.tensor([[0,11,12,13,14]])
    assert token_positions(ids,[12,14]).tolist()==[[2,4]]
    for tokens in [[99],[12,12]]:
        with pytest.raises(ValueError):token_positions(ids,tokens)
    with pytest.raises(ValueError):token_positions(torch.tensor([[12,12]]),[12])
    e=torch.randn(1,5,4);p=torch.arange(5).view(1,1,5).expand(3,1,5)
    out=insert_world_tokens(ids,e,torch.tensor([[0,1,1,1,1]]),p,torch.tensor([[3,4]]),torch.randn(1,2,4),0)
    assert out[0].tolist()==[[0,11,12,0,0,13,14]]
    torch.testing.assert_close(out[3][:,:,:3],p[:,:,:3])
    assert out[-1].tolist()==[[5,6]]
    for invalid in [[[3,99]],[[4,3]],[[3,3]]]:
        with pytest.raises(ValueError):insert_world_tokens(ids,e,torch.ones(1,5),p,torch.tensor(invalid),torch.randn(1,2,4),0)


def test_real_geometric_projection_has_image_gradient_and_contract_checks():
    provider=GeometricBEVProvider(channels=16,bounds=(1.,-2.,5.,2.),cameras=('CAM_F0',))
    # Camera optical z looks along ego x; optical x along -ego y.
    extr=torch.tensor([[0.,0.,1.,0.],[-1.,0.,0.,0.],[0.,-1.,0.,1.5],[0.,0.,0.,1.]]).reshape(1,1,4,4)
    inp=ModelInputs(torch.rand(1,1,3,32,64,requires_grad=True),torch.tensor([[30.,0,32],[0,30,16],[0,0,1.]]).reshape(1,1,3,3),extr,torch.eye(3).reshape(1,1,3,3),('CAM_F0',),torch.zeros(1,1),torch.zeros(1))
    features,coords,support,meta=provider(inp)
    assert support.any() and features.shape==(1,16,16)
    features.square().sum().backward()
    assert inp.current_images.grad.abs().sum()>0
    with pytest.raises(TypeError):provider({'anns':[]})
    with pytest.raises(ValueError):validate_feature_cache({},meta)
