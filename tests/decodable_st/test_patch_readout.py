"""Deterministic operator tests; these are not native teacher acceptance."""
import torch
from starVLA.model.modules.decodable_st.contracts import SpatialContract
from starVLA.model.modules.decodable_st.vigt_readout import ContextPatchBottleneck


def test_patch_axis_order_and_input_gradient():
    c=SpatialContract(2,4,6,(-1.,-2.,-1.),(5.,2.,3.))
    model=ContextPatchBottleneck(12,c,(2,2))
    with torch.no_grad():
        model.encode.weight.copy_(torch.eye(12));model.encode.bias.zero_()
        model.readout.project.weight.copy_(torch.eye(12));model.readout.project.bias.zero_()
    x=torch.arange(2*2*4*6,dtype=torch.float32).reshape(2,2,4,6).requires_grad_()
    torch.testing.assert_close(model(x),x,rtol=0,atol=0)
    model(x).sum().backward()
    torch.testing.assert_close(x.grad,torch.ones_like(x),rtol=0,atol=0)


def test_whitening_reparameterization_preserves_context_and_gradients():
    c=SpatialContract(2,4,6,(-1.,-2.,-1.),(5.,2.,3.))
    raw=ContextPatchBottleneck(5,c,(2,2));white=ContextPatchBottleneck(5,c,(2,2))
    g=torch.Generator().manual_seed(93)
    basis,_=torch.linalg.qr(torch.randn(12,5,generator=g,dtype=torch.float64))
    mean=torch.randn(12,generator=g,dtype=torch.float64)
    scale=torch.tensor([.2,.7,1.,3.,9.],dtype=torch.float64)
    raw=raw.double();white=white.double()
    with torch.no_grad():
        for model,s in ((raw,torch.ones_like(scale)),(white,scale)):
            model.encode.weight.copy_(basis.T/s[:,None]);model.encode.bias.copy_(-mean@basis/s)
            model.readout.project.weight.copy_(basis*s);model.readout.project.bias.copy_(mean)
    a=torch.randn(2,2,4,6,generator=g,dtype=torch.float64,requires_grad=True)
    b=a.detach().clone().requires_grad_()
    ya,yb=raw(a),white(b)
    torch.testing.assert_close(ya,yb,rtol=1e-12,atol=1e-12)
    ya.square().sum().backward();yb.square().sum().backward()
    torch.testing.assert_close(a.grad,b.grad,rtol=1e-12,atol=1e-12)
