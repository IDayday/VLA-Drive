from collections import OrderedDict
from types import SimpleNamespace
import torch
import pytest
from iqe.internal_baseline import cache_execution


def test_host_reader_keeps_bytes_rng_order_and_memory_bound(monkeypatch,tmp_path):
    scenes=[SimpleNamespace(scene_id=str(i)) for i in range(5)]
    values={s.scene_id:{'world':torch.full((4,8),int(s.scene_id),dtype=torch.float32),'valid':torch.ones(4,dtype=torch.bool)} for s in scenes}
    class Reader:
        def __init__(self,config,capacity):self.loaded=OrderedDict();self.capacity=capacity
        def read(self,scene):return {k:v.clone() for k,v in values[scene.scene_id].items()}
    monkeypatch.setattr(cache_execution,'JointCache',Reader)
    config=SimpleNamespace(root=tmp_path)
    rng=torch.get_rng_state().clone()
    reader=cache_execution.training_cache(config,scenes,max_bytes=1024**2)
    assert list(reader.loaded)==[s.scene_id for s in scenes]
    assert all(torch.equal(v,reader.loaded[s][k]) for s,x in values.items() for k,v in x.items())
    assert torch.equal(rng,torch.get_rng_state())
    bounded=cache_execution.training_cache(config,scenes,max_bytes=20000)
    assert bounded.capacity==1 and len(bounded.loaded)==1
    values['2']['world']=torch.zeros(5,8)
    with pytest.raises(ValueError,match='shapes changed'):
        cache_execution.training_cache(config,scenes,max_bytes=1024**2)
