"""Reuse the existing no-grad graph runner with velocity and sigma in one pass."""
from types import SimpleNamespace
import torch
from starVLA.rl.flow_grpo.velocity_graph import NoGradVelocityGraph, configure_velocity_graph
from starVLA.rl.flow_grpo.rollout import velocity


class _JointOutput:
    """Duck-typed original graph head, not a registered second model/module."""
    def __init__(self,actor):self.actor=actor
    @property
    def training(self):return self.actor.training
    def named_parameters(self):return self.actor.named_parameters()
    def predict_velocity(self,x,bucket,condition):
        v,h=self.actor.policy.action_model.predict_velocity_with_hidden(x,bucket,condition)
        return torch.cat([v.float(),self.actor.noise_std(h)],-1)


def configure(actor,enabled):
    actor._cfn_rollout_graph_enabled=bool(enabled)
    actor._cfn_rollout_graph=None
    configure_velocity_graph(actor.policy,enabled)


def outputs(actor,x,buckets,condition):
    if torch.is_grad_enabled():raise ValueError('Rollout graph is never a gradient recomputation path')
    if actor.noise_head is None:
        return velocity(actor.policy,x,buckets,condition),None
    graph=getattr(actor,'_cfn_rollout_graph',None)
    if graph is None:
        adapter=_JointOutput(actor)
        graph=NoGradVelocityGraph(adapter,x,buckets,condition)
        actor._cfn_rollout_graph=graph
    joint=graph(SimpleNamespace(action_model=graph.head),x,buckets,condition)
    return joint[...,:4],joint[...,4:]
