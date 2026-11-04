from types import SimpleNamespace
import pytest
import torch
from starVLA.model.modules.qwen_world_compare.qwen_flow_adapter import DPrefix
from starVLA.rl.d46_fpo_engineering.prefix_cache import collection_prefix_cache


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(2.));self._policy_version='u1';self.calls=0
        self.encoded=SimpleNamespace(inputs_embeds=torch.zeros(1,2,3),ego=torch.zeros(1,1))
    def materialize(self,c):return c
    def condition(self,n):
        return dict(current=SimpleNamespace(scene_identity=('scene',),encoded=self.encoded),context=torch.zeros(n,1,1))
    def _prepare(self,c,use_ref):
        self.calls+=1;n=len(c['context']);x=(self.weight*torch.ones(1,1,2)).expand(n,1,2)
        return DPrefix(((x,x),),torch.ones(n,1,dtype=torch.bool),torch.ones(n,dtype=torch.long),x,('scene',)*n)


def test_reuses_detached_one_scene_with_different_group_sizes():
    actor=Actor()
    with collection_prefix_cache(actor)as stats,torch.no_grad():
        x=actor._prepare(actor.condition(16),False);y=actor._prepare(actor.condition(1),False)
        assert actor.calls==1 and stats['hits']==1
        assert not x.layer_kv[0][0].requires_grad and x.layer_kv[0][0].stride(0)==0
        assert y.valid.shape==(1,1) and y.scene_identity==('scene',)
    assert actor._prepare(actor.condition(1),False).reasoning.requires_grad


def test_current_gradient_graph_is_always_rebuilt():
    actor=Actor()
    with collection_prefix_cache(actor):
        for _ in range(2):
            actor.zero_grad();actor._prepare(actor.condition(1),False).reasoning.sum().backward()
            assert actor.weight.grad.item()==2
        assert actor.calls==2


def test_cache_rejects_optimizer_mutation_inside_scope():
    actor=Actor()
    with pytest.raises(ValueError,match='Optimizer mutation'):
        with collection_prefix_cache(actor),torch.no_grad():actor.weight.add_(1)


def test_cache_rejects_policy_version_change():
    actor=Actor()
    with pytest.raises(ValueError,match='Policy changed'):
        with collection_prefix_cache(actor),torch.no_grad():
            actor._policy_version='u2';actor._prepare(actor.condition(1),False)
