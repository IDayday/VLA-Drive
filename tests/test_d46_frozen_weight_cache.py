import copy
import torch
from starVLA.rl.simwam_recipe.vendor.lora import LoRALinear
from starVLA.rl.d46_fpo_engineering.frozen_weights import FrozenLinearCache


def test_frozen_base_cast_keeps_real_LoRA_outputs_and_gradients_bitwise():
    torch.manual_seed(18);model=LoRALinear(torch.nn.Linear(16,12),r=4,alpha=8)
    with torch.no_grad():model.lora_B.normal_()
    original=copy.deepcopy(model.state_dict());baseline=None
    for cached in(False,True):
        model.zero_grad();x=torch.randn(3,16,generator=torch.Generator().manual_seed(4),requires_grad=True)
        cache=FrozenLinearCache(model)if cached else None
        with torch.autocast('cpu',dtype=torch.bfloat16):out=model(x)
        out.float().square().mean().backward()
        result=[out.detach(),x.grad,model.lora_A.grad,model.lora_B.grad]
        if baseline is None:baseline=[v.clone()for v in result]
        else:
            assert all(torch.equal(a,b)for a,b in zip(baseline,result))
            assert cache.misses==2
        if cache:cache.close()
    assert all(torch.equal(original[k],v)and v.dtype==torch.float32 for k,v in model.state_dict().items())


def test_version_change_invalidates_frozen_cast():
    model=torch.nn.Linear(4,3).requires_grad_(False);x=torch.randn(2,4)
    with FrozenLinearCache(model)as cache:
        with torch.autocast('cpu',dtype=torch.bfloat16):old=model(x)
        with torch.no_grad():model.weight.add_(1)
        with torch.autocast('cpu',dtype=torch.bfloat16):new=model(x)
        assert not torch.equal(old,new)and cache.misses==3
    with torch.autocast('cpu',dtype=torch.bfloat16):assert torch.equal(model(x),new)


def test_full_precision_and_trainable_weights_are_not_cached():
    for frozen in(False,True):
        model=torch.nn.Linear(4,3).requires_grad_(not frozen);x=torch.randn(2,4)
        expected=model(x)
        with FrozenLinearCache(model)as cache:
            assert torch.equal(model(x),expected)and cache.misses==0
            if not frozen:
                with torch.autocast('cpu',dtype=torch.bfloat16):model(x).sum().backward()
                assert model.weight.grad is not None and not cache.entries


def test_reference_and_actor_share_one_cached_constant():
    a=torch.nn.Linear(4,3,bias=False).requires_grad_(False);b=torch.nn.Linear(4,3,bias=False).requires_grad_(False)
    b.weight=a.weight;model=torch.nn.ModuleList([a,b]);x=torch.randn(2,4)
    with FrozenLinearCache(model)as cache:
        with torch.autocast('cpu',dtype=torch.bfloat16):assert torch.equal(a(x),b(x))
        assert cache.misses==1 and cache.hits==1
