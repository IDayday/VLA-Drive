from dataclasses import dataclass


@dataclass(frozen=True)
class ForesightConfig:
    arm: str = 'A'
    num_queries: int = 64
    readout_dim: int = 512
    readout_layers: int = 2
    latent_channels: int = 16
    latent_height: int = 32
    latent_width: int = 57
    lambda_vis: float = 0.0
    lambda_int: float = 0.0
    auxiliary_warmup: int = 1000
    normalization_eps: float = 1e-5
    gradient_checkpointing: bool = True
    enable_current_dino: bool = False
    enable_future_dino: bool = False
    enable_interaction: bool = False
    current_horizon_s: float = 0.
    future_horizons_s: tuple = (1., 2., 4.)
    dino_feature_dim: int = 1024
    dino_height: int = 16
    dino_width: int = 29
    lambda_cur: float = 0.
    lambda_fut: float = 0.
    full_algorithm: bool = False
    candidate: str = ''
    ablation: str = 'FULL'
    future_target_type: str = 'legacy_single_frame'
    future_action_condition: str = 'none'
    interaction_readout_source: str = 'world'
    future_action_injection: str = 'memory_only'
    future_action_query_scale: float = 1.0
    interaction_functional_loss: str = 'disabled'
    planner_condition_mode: str = 'action_only'
    clip_times_s: tuple = (.5, 1., 1.5, 2., 2.5, 3., 3.5, 4.)
    future_feature_dim: int = 1024
    future_grid_height: int = 9
    future_grid_width: int = 12
    future_time_intervals_s: tuple = ((.5, 1.), (1.5, 2.), (2.5, 3.), (3.5, 4.))
    video_teacher_identity: str = ''
    future_normalization: str = 'nonaffine_layernorm_per_token_v1'

    @property
    def is_tradeoff(self):
        from .tradeoff import CANDIDATES
        return self.arm in CANDIDATES and not self.full_algorithm

    @property
    def is_dino(self):
        return self.full_algorithm or self.arm.startswith('W_') or self.arm == 'R_NATIVE' or self.is_tradeoff

    @property
    def uses_interaction(self):
        return self.enable_interaction if self.is_dino else self.arm in ('C','D')

    def validate(self):
        if self.future_target_type not in ('legacy_single_frame', 'dino_sequence', 'video_clip'):
            raise ValueError('Unknown future representation')
        if self.future_action_condition not in ('none', 'gt_ego') or self.planner_condition_mode not in ('action_only', 'action_plus_W', 'action_residual_W'):
            raise ValueError('Unknown action condition / planner mode')
        import math
        if self.interaction_readout_source not in ('world', 'action'):
            raise ValueError('Unknown interaction readout source')
        if self.future_action_injection not in ('memory_only', 'memory_and_query') or not math.isfinite(self.future_action_query_scale):
            raise ValueError('Unknown/nonfinite future action query injection')
        if self.future_action_injection == 'memory_and_query' and self.future_action_condition != 'gt_ego':
            raise ValueError('Action query injection requires explicit GT ego conditioning')
        if self.interaction_functional_loss not in ('disabled', 'frozen_teacher', 'plain_head'):
            raise ValueError('Unknown interaction functional supervision')
        if self.interaction_functional_loss != 'disabled':
            raise ValueError('Functional controls require a separately registered follow-up; not part of this first comparison')
        if self.future_target_type == 'legacy_single_frame':
            if self.future_action_condition != 'none' or self.planner_condition_mode != 'action_only' or self.video_teacher_identity:
                raise ValueError('Legacy configuration cannot silently change its action/future paths')
        else:
            if not self.full_algorithm or self.candidate != 'C1' or tuple(self.clip_times_s) != (.5, 1., 1.5, 2., 2.5, 3., 3.5, 4.):
                raise ValueError('Action/video study fixes C1 and eight real physical timestamps')
            if self.future_normalization != 'nonaffine_layernorm_per_token_v1' or not self.video_teacher_identity:
                raise ValueError('Explicit frozen target identity and normalization required')
            if min(self.future_feature_dim, self.future_grid_height, self.future_grid_width) < 1:
                raise ValueError('Invalid native future feature layout')
            spans = tuple(tuple(x) for x in self.future_time_intervals_s)
            expected = tuple((t, t) for t in self.clip_times_s) if self.future_target_type == 'dino_sequence' else ((.5, 1.), (1.5, 2.), (2.5, 3.), (3.5, 4.))
            if spans != expected:
                raise ValueError('Future token times must follow image or native tubelet layout')
        if self.full_algorithm:
            from .tradeoff import CANDIDATES
            definitions = {'FULL': (True, True, True), 'NO_INTERACTION': (True, True, False),
                'NO_FUTURE': (True, False, True), 'NO_CURRENT': (False, True, True),
                'W_ACTION_ONLY': (False, False, False), 'NATIVE_ACTION_ONLY': (False, False, False)}
            if self.candidate not in CANDIDATES or self.ablation not in definitions:
                raise ValueError('Unknown full-method candidate/ablation')
            c = CANDIDATES[self.candidate]
            expected_arm = self.candidate if self.ablation == 'FULL' else self.ablation
            if self.arm != expected_arm:
                raise ValueError('Full-method name and supervision declaration disagree')
            enabled = (self.enable_current_dino, self.enable_future_dino, self.enable_interaction)
            if enabled != definitions[self.ablation]:
                raise ValueError('Full-method supervision cannot silently change')
            queries = 0 if self.ablation == 'NATIVE_ACTION_ONLY' else c.num_queries
            if (self.num_queries, self.dino_height, self.dino_width, self.dino_feature_dim) != (queries, *c.grid_hw, 1024):
                raise ValueError('Full-method query/grid mismatch')
            if (self.readout_dim, self.readout_layers) != (512, 2):
                raise ValueError('Shared two-layer 512-dimensional readout required')
            if self.current_horizon_s != 0 or tuple(self.future_horizons_s) != (1., 2., 4.):
                raise ValueError('Physical horizon contract')
            weights = (self.lambda_cur, self.lambda_fut, self.lambda_int)
            if self.lambda_vis or self.auxiliary_warmup < 1 or self.normalization_eps <= 0:
                raise ValueError('Invalid full-method normalization/warmup')
            import math
            if any(not math.isfinite(w) or (w <= 0 if active else w != 0) for w, active in zip(weights, enabled)):
                raise ValueError('Every enabled task requires a finite positive weight')
            if self.enable_current_dino and self.lambda_cur != 1.:
                raise ValueError('Current DINO weight is fixed at 1.0')
            return self
        if self.candidate or self.ablation != 'FULL':
            raise ValueError('Full-method fields on a legacy configuration')
        if self.is_tradeoff:
            from .tradeoff import CANDIDATES
            c=CANDIDATES[self.arm]
            if (self.num_queries,self.dino_height,self.dino_width,self.dino_feature_dim)!=(c.num_queries,*c.grid_hw,1024):
                raise ValueError('Registered tradeoff query/grid mismatch')
            if not self.enable_current_dino or self.enable_future_dino or self.enable_interaction:
                raise ValueError('Tradeoff permits current alignment only')
            if (self.lambda_cur,self.lambda_fut,self.lambda_int,self.lambda_vis,self.auxiliary_warmup)!=(1.,0.,0.,0.,1):
                raise ValueError('Tradeoff current MSE weight is exactly1, without auxiliary warmup')
            if self.current_horizon_s!=0:raise ValueError('Current decision time required')
            return self
        if self.is_dino:
            definitions = {'R_NATIVE':(False,False,False), 'W_ONLY':(False,False,False),
                'W_CUR':(True,False,False), 'W_FUT':(False,True,False), 'W_CUR_FUT':(True,True,False),
                'W_FUT_INT':(False,True,True), 'W_FULL':(True,True,True), 'W_INT':(False,False,True)}
            if self.arm not in definitions or (self.enable_current_dino,self.enable_future_dino,self.enable_interaction) != definitions[self.arm]:
                raise ValueError('Named ablation and supervision flags disagree')
            if self.num_queries != (0 if self.arm=='R_NATIVE' else 64): raise ValueError('Fixed W capacity')
            if self.current_horizon_s != 0 or tuple(self.future_horizons_s) != (1.,2.,4.): raise ValueError('Physical horizon contract')
            if self.lambda_vis or min(self.lambda_cur,self.lambda_fut,self.lambda_int)<0: raise ValueError('Invalid new auxiliary weights')
            if any(weight and not active for weight,active in zip((self.lambda_cur,self.lambda_fut,self.lambda_int), definitions[self.arm])):
                raise ValueError('Disabled task has nonzero weight')
            if self.readout_dim % 8 or self.readout_layers<1 or min(self.dino_feature_dim,self.dino_height,self.dino_width)<1 or self.auxiliary_warmup<1:
                raise ValueError('Invalid DINO readout dimensions/warmup')
            return self
        if self.enable_current_dino or self.enable_future_dino or self.enable_interaction or self.lambda_cur or self.lambda_fut:
            raise ValueError('Legacy FLUX configuration cannot silently enable DINO')
        if self.arm not in ('R','A','B','C','D'): raise ValueError('Unknown ablation')
        if self.num_queries != (0 if self.arm=='R' else 64): raise ValueError('Fixed campaign query capacity')
        if self.readout_dim % 8 or self.readout_layers<1: raise ValueError('Invalid readout dimensions')
        if min(self.latent_channels,self.latent_height,self.latent_width)<1: raise ValueError('Invalid VAE geometry')
        if self.lambda_vis<0 or self.lambda_int<0 or self.auxiliary_warmup<1: raise ValueError('Invalid auxiliary weights')
        if self.arm not in ('B','D') and self.lambda_vis: raise ValueError('Unexpected visual supervision')
        if self.arm not in ('C','D') and self.lambda_int: raise ValueError('Unexpected interaction supervision')
        return self
