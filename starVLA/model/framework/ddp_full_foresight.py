"""Four-task training; pure-current original DDP ego planning at deployment.

No DINO or trajectory teacher is instantiated by this module. Labels enter only
the loss wrapper inherited from DDPForesight, after one current Qwen forward.
"""
from .DDPForesight import DDPForesight
from starVLA.model.modules.foresight.queries import SpatialViewEncoding
from starVLA.model.modules.vehicle_joint.initialization import initialization_seed


class DDPFullForesight(DDPForesight):
    def __init__(self, config, accelerator=None):
        if config.framework.name != 'DDPFullForesight' or not config.foresight.get('full_algorithm', False):
            raise ValueError('Explicit full-method configuration required')
        super().__init__(config, accelerator=accelerator)
        cfg = self.foresight_config
        if cfg.num_queries:
            hidden = self.qwen_vl_interface.model.config.hidden_size
            # This isolated stream cannot change the original driving or head initialization.
            with initialization_seed(int(config.seed)+2250):
                self.query_geometry = SpatialViewEncoding(hidden, (cfg.dino_height, cfg.dino_width))
        if hasattr(self, 'future_head'):
            raise ValueError('FLUX/video branch is forbidden in the full DINO method')

    def query_embeddings(self):
        return self.query_geometry(self.foresight_queries)

    def forward(self, model_inputs, training_targets, **kwargs):
        cfg = self.foresight_config
        if getattr(self, '_deployment_only', False):
            raise RuntimeError('Deployment model cannot silently become a reduced training objective')
        for name, required in (('dino_head', cfg.enable_current_dino or cfg.enable_future_dino),
                               ('interaction_head', cfg.enable_interaction)):
            if hasattr(self, name) != required:
                raise RuntimeError('Configured training objective/module inventory mismatch: '+name)
        return super().forward(model_inputs, training_targets, **kwargs)

    def forward_train(self, model_inputs, training_targets, **kwargs):
        return self.forward(model_inputs, training_targets, **kwargs)

    def strip_auxiliary_heads(self):
        super().strip_auxiliary_heads()
        self._deployment_only = True
        return self
