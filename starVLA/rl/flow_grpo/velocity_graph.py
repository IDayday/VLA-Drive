"""Exact-layout no-grad CUDA replay of the existing velocity kernel.

No batching, freezing, cached condition or weight snapshots. Grad-enabled calls
remain on the native path. Graphs read live BF16 parameter storage, cast within
each replay, and return independent outputs. They are process-local execution
caches and never serialized as model/optimizer state.
"""
import torch


class NoGradVelocityGraph:
    def __init__(self, head, x, bucket, condition):
        if torch.is_grad_enabled() or head.training:
            raise RuntimeError('eval/no-grad required')
        self.head=head
        self.inputs=[v.clone() for v in (x,bucket,condition)]
        self.identity={n:(id(p),p.data_ptr(),p.dtype) for n,p in head.named_parameters()}
        def forward():
            with torch.autocast('cuda',dtype=torch.float32,cache_enabled=False):
                return head.predict_velocity(*self.inputs).float()
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):forward()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.output=forward()
    def __call__(self, policy, x, bucket, condition, checkpoint=False):
        if torch.is_grad_enabled() or self.head.training or policy.action_model is not self.head:
            raise RuntimeError('no-grad same-head required')
        if self.identity != {n:(id(p),p.data_ptr(),p.dtype) for n,p in self.head.named_parameters()}:
            raise RuntimeError('CUDA graph parameter storage changed; recapture required')
        for static,real in zip(self.inputs,(x,bucket,condition)):
            if (static.shape,static.dtype,static.device)!=(real.shape,real.dtype,real.device):
                raise ValueError('CUDA graph input profile mismatch')
            static.copy_(real)
        self.graph.replay()
        return self.output.clone()



def configure_velocity_graph(policy, enabled):
    policy._flow_velocity_graph_enabled = bool(enabled)
    policy._flow_velocity_graph = None


def graph_velocity(policy, x, bucket, condition):
    graph = getattr(policy, "_flow_velocity_graph", None)
    if graph is None:
        graph = NoGradVelocityGraph(policy.action_model, x, bucket, condition)
        policy._flow_velocity_graph = graph
    return graph(policy, x, bucket, condition)
