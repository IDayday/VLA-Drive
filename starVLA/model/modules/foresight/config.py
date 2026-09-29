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

    @property
    def is_tradeoff(self):
        from .tradeoff import CANDIDATES
        return self.arm in CANDIDATES

    @property
    def is_dino(self):
        return self.arm.startswith('W_') or self.arm == 'R_NATIVE' or self.is_tradeoff

    @property
    def uses_interaction(self):
        return self.enable_interaction if self.is_dino else self.arm in ('C','D')

    def validate(self):
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
