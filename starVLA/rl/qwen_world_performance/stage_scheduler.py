"""Learning rate is a function of valid fresh scene exposures, never Adam reuse."""
import copy
import math


def lr_multiplier(exposure_epoch, spec):
    if not math.isfinite(exposure_epoch) or exposure_epoch < 0:
        raise ValueError('Invalid exposure clock')
    warm, hold, end = (spec[k] for k in ('warmup_epochs', 'hold_until_epoch', 'cosine_decay_end_epoch'))
    floor = spec['minimum_lr_ratio']
    if not (0 < warm <= hold < end and 0 < floor <= 1):
        raise ValueError('Invalid registered schedule')
    if exposure_epoch < warm:
        return .001 + .999 * exposure_epoch / warm
    if exposure_epoch <= hold:
        return 1.
    if exposure_epoch >= end:
        return floor
    return floor + (1-floor)*(1+math.cos(math.pi*(exposure_epoch-hold)/(end-hold)))/2


class SceneExposureScheduler:
    """Only scalar state is serialized. No reference to the runtime/state graph."""
    def __init__(self, optimizer, spec, population):
        self.optimizer = optimizer
        self.spec = copy.deepcopy(spec)
        self.population = population
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        self.exposures = 0
        self.last_epoch = 0
        self.stage = 'initial'
        self._apply()

    def _apply(self):
        self._last_lr = [p*lr_multiplier(self.exposures/self.population, self.spec) for p in self.base_lrs]
        for g, lr in zip(self.optimizer.param_groups, self._last_lr):
            g['lr'] = lr

    def set_exposures(self, value):
        if type(value) is not int or value < self.exposures:
            raise ValueError('Exposure clock cannot go backwards')
        self.exposures = value
        self._apply()

    def set_stage(self, name, peaks):
        if len(peaks) != len(self.base_lrs) or any(p <= 0 for p in peaks):
            raise ValueError('Explicit positive per-group stage peaks required')
        self.stage, self.base_lrs = name, list(peaks)
        self._apply()

    def step(self):
        self.last_epoch += 1
        self._apply()

    def state_dict(self):
        return copy.deepcopy(dict(schema='world_scene_exposure_scheduler_v1', spec=self.spec,
            population=self.population, base_lrs=self.base_lrs, exposures=self.exposures,
            last_epoch=self.last_epoch, stage=self.stage, _last_lr=self._last_lr))

    def load_state_dict(self, value):
        if value['spec'] != self.spec or value['population'] != self.population:
            raise ValueError('Native scheduler definition changed')
        for k in ('base_lrs', 'exposures', 'last_epoch', 'stage'):
            setattr(self, k, copy.deepcopy(value[k]))
        self._apply()
        if self._last_lr != value['_last_lr']:
            raise ValueError('Native LR does not match exposure clock')


def stage_for_epoch(spec, epoch):
    stages = spec['stages']
    eligible = [s for s in stages if s['start_epoch'] <= epoch+1e-12]
    if not eligible:
        raise ValueError('Missing initial algorithm stage')
    return copy.deepcopy(eligible[-1])
