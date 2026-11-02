"""One EMA/beta update per fresh behavior collection, never per microbatch."""
from dataclasses import dataclass, asdict
import math


@dataclass
class AdaptiveKL:
    beta: float=.001
    ema: float=0.
    collections: int=0
    target: float=.015

    def update(self, joint_per_step):
        value=float(joint_per_step)/24
        if not math.isfinite(value) or value<0:
            raise FloatingPointError('Invalid native reference KL')
        self.ema=value if self.collections==0 else .9*self.ema+.1*value
        multiplier=math.exp(max(-math.log(1.05),min(math.log(1.05),.02*(self.ema/self.target-1))))
        self.beta=max(.0001,min(.01,self.beta*multiplier));self.collections+=1
        return self.beta

    def state_dict(self):return asdict(self)
    def load_state_dict(self,value):self.__dict__.update(value)
