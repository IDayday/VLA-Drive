import pytest
import torch
from starVLA.rl.flow_grpo.data import keys_for_positions
from starVLA.rl.constrained_flow_noise.runtime import OrderedScenes, exposure_lr
from starVLA.rl.constrained_flow_noise.effective_group_collector import EffectiveGroupCollector


def test_stream_matches_existing_epoch_order_across_restart():
    tokens=[str(i) for i in range(101)]
    stream=OrderedScenes(tokens,42)
    positions=[0,1,15,16,99,100,101,102,201,202,309]
    expected=[key[0] for key in keys_for_positions(tokens,42,positions)]
    assert [stream.at(p) for p in positions]==expected
    assert [OrderedScenes(tokens,42).at(p) for p in positions]==expected


def test_scene_scheduler_warmup_terminal_and_collection_clock():
    total=203184;warmup=6096
    assert exposure_lr(16,total)<exposure_lr(32,total)<exposure_lr(warmup,total)
    assert exposure_lr(warmup,total)==pytest.approx(1e-6)
    assert exposure_lr(total,total)==pytest.approx(1e-7)
    assert exposure_lr(32,total)/exposure_lr(16,total)==pytest.approx(2.)


@pytest.mark.parametrize('valid_count',[0,1,3,15,16])
def test_eight_rank_partial_group_reduction_matches_unpadded_mean(valid_count):
    values=torch.arange(1,17,dtype=torch.float64,requires_grad=True)
    contributions=[]
    for rank in range(8):
        for micro in range(2):
            slot=micro*8+rank
            factor=16/max(1,valid_count) if slot<valid_count else 0.
            contributions.append(values[slot]*factor/16)
    total=sum(contributions);total.backward()
    target=values[:valid_count].mean() if valid_count else values.sum()*0
    assert total.detach()==pytest.approx(float(target.detach()))
    expected=torch.zeros(16,dtype=torch.float64)
    if valid_count:expected[:valid_count]=1/valid_count
    torch.testing.assert_close(values.grad,expected,atol=0,rtol=0)
