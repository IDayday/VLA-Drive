"""Vehicle-only metric decoding of the randomly initialized query heads."""
from starVLA.model.modules.structured_world.agent_heads import AgentHeads


class VehicleHeads(AgentHeads):
    def __init__(self, qwen_dim, *, xy_scale=20., **kwargs):
        super().__init__(qwen_dim, classes=1, **kwargs)
        if xy_scale not in (1., 20.):
            raise ValueError('Supported vehicle head coordinate versions are metric_v0 and scaled_xy_v1')
        self.xy_scale = float(xy_scale)

    def forward(self, hidden):
        output = super().forward(hidden)
        # The first diagnostic used raw network outputs directly as metres.
        # Version1 predicts xy in the SAME 20m units used by the supervised
        # regression loss and joint neighbor residuals. Other channels retain
        # their original definitions. This changes no parameter initialization.
        scale = output['boxes'].new_tensor([self.xy_scale, self.xy_scale, 1., 1., 1., 1., 1., 1.])
        output['boxes'] = output['boxes'] * scale
        # AgentHeads already forms (current center + motion displacement).
        # Scaling that sum preserves the same track's physical center anchor.
        output['future_xy'] = output['future_xy'] * self.xy_scale
        return output
