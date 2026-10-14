"""One residual, matched information probes and camera-only predicted memory."""
import torch
from torch import nn
from starVLA.model.modules.foresight.future_latent_head import CrossReadout
from .consequence_contract import OracleRelations, query_features


class ResidualRefiner(nn.Module):
    def __init__(self, hidden_dim=2048, dim=512, layers=2):
        super().__init__()
        self.action_memory = nn.Linear(hidden_dim, dim)
        self.geometry = nn.Sequential(nn.Linear(14, dim), nn.GELU(), nn.Linear(dim, dim))
        self.relations = nn.Sequential(nn.Linear(10, dim), nn.GELU(), nn.Linear(dim, dim))
        self.plan_query = nn.Sequential(nn.Linear(5, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(CrossReadout(dim, 8) for _ in range(layers))
        self.output = nn.Linear(dim, 4)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def _refine(self, inputs, queries, values, support):
        # Fixed scaling; no GT-dependent normalizer or correction gate.
        scale = values.new_tensor([5., 5., 1., 5., 1.])
        related = torch.cat((torch.where(support, values/scale, torch.zeros_like(values)), support.float()), -1)
        memory = self.geometry(query_features(inputs, queries))+self.relations(related.float())
        memory = torch.cat((self.action_memory(inputs.H_A.float()), memory.flatten(1, 2)), 1)
        B, T, _ = inputs.q0_encoded.shape
        times = queries.times[None, :, None].expand(B, T, 1)/4.
        x = self.plan_query(torch.cat((inputs.q0_encoded.float(), times), -1))
        for block in self.blocks:
            x = block(x, memory)
        return inputs.q0_encoded.float()+self.output(x)

    def camera_only(self, inputs, queries, predicted=None):
        shape = (*queries.poses.shape[:-1], 5)
        if predicted is None:
            values = inputs.q0.new_zeros(shape)
            support = torch.zeros(shape, device=values.device, dtype=torch.bool)
        else:
            if not isinstance(predicted, torch.Tensor) or predicted.shape != shape:
                raise TypeError('camera_only accepts predicted tensor, not OracleRelations/TrainingTargets')
            values = predicted
            kinematic = queries.kinematic_valid.clone(); kinematic[:, 0] = True
            support = (queries.in_grid & kinematic[..., None])[..., None].expand(shape)
        return self._refine(inputs, queries, values, support)

    def oracle_probe(self, inputs, queries, relations):
        if not isinstance(relations, OracleRelations):
            raise TypeError('Privileged probe requires explicit OracleRelations')
        return self._refine(inputs, queries, relations.values, relations.common_valid)
