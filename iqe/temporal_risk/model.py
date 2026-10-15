"""Keep the geometric scoring function; add training-only temporal event logits."""
import torch
from torch import nn
from iqe.contracts import require
from iqe.geometric_readout.model import GeometricReadoutScorer
from iqe.internal_baseline.scorer import initialization_rng


class TemporalRiskScorer(GeometricReadoutScorer):
    def __init__(self, *args, event_init_seed, **kwargs):
        super().__init__(*args, **kwargs)
        # Do not perturb common weights or the caller's training RNG.
        with initialization_rng(event_init_seed):
            self.event_head = nn.Linear(self.type_embeddings.shape[-1], 3)

    def forward(self, context, trajectories):
        # Reuse the exact existing decoder call, not another VLM/decoder pass.
        temporal = []
        handle = self.decoder.register_forward_hook(lambda module, args, output: temporal.append(output))
        try:
            prediction = super().forward(context, trajectories)
        finally:
            handle.remove()
        require(len(temporal) == 1, 'exactly one temporal decoder execution required')
        b, k, t, _ = trajectories.shape
        prediction['event_logits'] = self.event_head(temporal[0]).reshape(b, k, t, 3)
        return prediction
