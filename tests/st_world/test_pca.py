import pytest
import torch

from tools.st_world_policy.teacher_pca import StreamingCovariance,TeacherPCA,make_patch_plan


def fit(x,k):
    s=StreamingCovariance(x.shape[1])
    for chunk in x.split(17):s.update(chunk)
    return s.basis(k),s


def test_streaming_statistics_equal_full_and_nonwhiten():
    rng=torch.Generator().manual_seed(42)
    x=torch.randn(400,8,generator=rng)*torch.arange(1,9)+torch.arange(8)
    p,s=fit(x,8)
    assert torch.allclose(s.mean,x.double().mean(0),atol=1e-10)
    assert torch.allclose(p['channel_std'],x.std(0,unbiased=False),atol=1e-6)
    t=TeacherPCA(p,dimension=8,identity='test',teacher_contract_identity='a'*64)
    z=t.transform(x);reconstructed=t.inverse(z)
    assert torch.allclose(x,reconstructed,atol=6e-6)
    # Nonwhitening preserves eigenvalue-dependent coordinate variance.
    assert torch.allclose(z.var(0),p['eigenvalues'],atol=2e-6)
    assert not torch.allclose(z.var(0),torch.ones(8),atol=.01)


def test_dimensions_share_stats_and_reconstruction_nested():
    x=torch.randn(200,8,generator=torch.Generator().manual_seed(1))
    p,_=fit(x,6)
    a=TeacherPCA(p,dimension=2,identity='test',teacher_contract_identity='a'*64)
    b=TeacherPCA(p,dimension=6,identity='test',teacher_contract_identity='a'*64)
    assert torch.equal(a.transform(x),b.transform(x)[:,:2])
    assert (b.inverse(b.transform(x))-x).square().mean()<(a.inverse(a.transform(x))-x).square().mean()
    # Delta uses the same linear transform: offsets cancel; magnitude is retained.
    y=x*.8
    expected=((y-x)/p['channel_std'])@p['components'][:2].T
    assert torch.allclose(a.transform(y)-a.transform(x),expected,atol=1e-6)


def test_zero_variance_and_nonfinite():
    x=torch.randn(80,5);x[:,2]=4
    p,_=fit(x,4)
    assert p['constant_channels'][2] and p['channel_std'][2]>0
    assert torch.isfinite(TeacherPCA(p,dimension=4,identity='x',teacher_contract_identity='a'*64).transform(x)).all()
    s=StreamingCovariance(5)
    with pytest.raises(ValueError):s.update(torch.full((5,5),float('nan')))
    with pytest.raises(ValueError):s.basis(4)


def test_role_guard_precedes_any_cache_read():
    class Reader:
        manifest={'split':'dev','scenes':[]}
    with pytest.raises(ValueError,match='train-only'):make_patch_plan(Reader(),seed=1,max_patches=32,purpose='fit')
