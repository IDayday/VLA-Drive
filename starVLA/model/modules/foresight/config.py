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

    def validate(self):
        if self.arm not in ('R','A','B','C','D'): raise ValueError('Unknown ablation')
        if self.num_queries != (0 if self.arm=='R' else 64): raise ValueError('Fixed campaign query capacity')
        if self.readout_dim % 8 or self.readout_layers<1: raise ValueError('Invalid readout dimensions')
        if min(self.latent_channels,self.latent_height,self.latent_width)<1: raise ValueError('Invalid VAE geometry')
        if self.lambda_vis<0 or self.lambda_int<0 or self.auxiliary_warmup<1: raise ValueError('Invalid auxiliary weights')
        if self.arm not in ('B','D') and self.lambda_vis: raise ValueError('Unexpected visual supervision')
        if self.arm not in ('C','D') and self.lambda_int: raise ValueError('Unexpected interaction supervision')
        return self
