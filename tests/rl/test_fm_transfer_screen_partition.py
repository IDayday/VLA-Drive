import json
import pytest
from scripts.analysis.fm_transfer.dispatch_screen import partition_plan,admit_partition
from starVLA.rl.flow_grpo.transactions import atomic_json


def contract():
    return dict(SFT_identity='sft',SFT_model_sha256='weight',worker_source='immutable_worker',global_shards=24)


def receipt(rank):
    c=contract()
    return dict(rank=rank,SFT_identity=c['SFT_identity'],SFT_model_sha256=c['SFT_model_sha256'],
        source=c['worker_source'],global_shards=c['global_shards'])


def test_three_hosts_cover_every_scene_once_and_preserve_gpu_mapping():
    partitions=[partition_plan(24,range(start,start+8)) for start in (0,8,16)]
    tokens=list(range(101592))
    assigned=[]
    for ranks,reserved in partitions:
        assert set(r%8 for r in ranks)==set(range(8))
        assert reserved=={gpu:26000 for gpu in range(8)}
        for rank in ranks:assigned.extend(tokens[rank::24])
    assert len(assigned)==len(tokens) and sorted(assigned)==tokens
    assert partition_plan(24)[1]=={gpu:46000 for gpu in range(8)}


@pytest.mark.parametrize('ranks',[[],[0,0],[-1],[24]])
def test_invalid_partition_rejected(ranks):
    with pytest.raises(ValueError):partition_plan(24,ranks)


def test_partial_launch_can_only_admit_different_unlaunched_shards(tmp_path):
    c=contract();admit_partition(tmp_path,c,list(range(8)),True)
    (tmp_path/'dispatches').mkdir()
    atomic_json(tmp_path/'dispatches/first_rank0.json',receipt(0))
    atomic_json(tmp_path/'dispatches/first_rank1.json',receipt(1))
    # A partial launch may be repaired explicitly; never relaunch its existing ranks.
    admit_partition(tmp_path,c,list(range(2,8)),True)
    admit_partition(tmp_path,c,list(range(8,16)),True)
    with pytest.raises(ValueError):admit_partition(tmp_path,c,[0],True)
    with pytest.raises(ValueError):admit_partition(tmp_path,c,[8],False)


@pytest.mark.parametrize('change',['source','weight','shards','missing_contract','untracked','duplicate'])
def test_no_cross_identity_or_untracked_worker_admission(tmp_path,change):
    c=contract();admit_partition(tmp_path,c,[0],True)
    (tmp_path/'dispatches').mkdir()
    atomic_json(tmp_path/'dispatches/first_rank0.json',receipt(0))
    if change=='source':c['worker_source']='another_source'
    elif change=='weight':c['SFT_model_sha256']='another_weight'
    elif change=='shards':c['global_shards']=16
    elif change=='missing_contract':(tmp_path/'dispatch_contract.json').unlink()
    elif change=='untracked':
        (tmp_path/'dispatches/first_rank0.json').unlink();atomic_json(tmp_path/'manifest.json',{})
    elif change=='duplicate':atomic_json(tmp_path/'dispatches/second_rank0.json',receipt(0))
    with pytest.raises(ValueError):admit_partition(tmp_path,c,[8],True)
