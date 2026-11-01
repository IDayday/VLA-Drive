"""Independent early-convergence runtime; never resumes an older schedule family."""
import copy
import math

from .consolidation_recipe import (validate_consolidation_recipe, PROTOCOL,
    ConsolidationSchedulerLossMixin, validate_scheduler_state, peaks)
from .credit import GroupCredit
from .group_runtime import GroupRuntime
from .quality import QualityDiagnosticsMixin
from . import recovery


class ConsolidationRuntimeMixin:
    recipe_validator = staticmethod(validate_consolidation_recipe)

    def make_credit(self, cfg):
        self.recipe_validator(cfg)
        return GroupCredit(8)

    def make_actor(self, cfg, args, device):
        actor, a, b = super().make_actor(cfg, args, device)
        loss_class = type(actor._loss)
        class ScheduledLoss(ConsolidationSchedulerLossMixin, loss_class):
            pass
        actor._loss = ScheduledLoss(cfg)
        return actor, a, b

    def validate_boundary(self, state, rollout, identity, optimizer=None, scheduler=None, *, legacy=False):
        if legacy:
            raise ValueError('Consolidation requires complete native schema2, not legacy migration')
        cfg = self.recipe_validator(identity['recipe'])
        if (optimizer is None) != (scheduler is None):
            raise ValueError('Consolidation optimizer and scheduler must be validated as a pair')
        summary = recovery.validate_boundary(state, rollout, identity, optimizer, scheduler,
            credit=self.credit, recipe_validator=self.recipe_validator)
        actual = None
        if scheduler is not None:
            ss = scheduler.state_dict() if hasattr(scheduler, 'state_dict') else scheduler
            expected = validate_scheduler_state(ss, peaks(cfg), state['updates'])
            oo = optimizer.state_dict() if hasattr(optimizer, 'state_dict') else optimizer
            groups = oo['param_groups']
            if (len(groups) != 2
                    or [g.get('name') for g in groups] != ['action_lora', 'scene_predictive_memory']
                    or [g.get('initial_lr') for g in groups] != peaks(cfg)
                    or any(not math.isfinite(g['lr']) or abs(g['lr']-x) > 1e-15
                           for g, x in zip(groups, expected))):
                raise ValueError('Adam two-group LR/initialpeak differs from consolidation state')
            actual = copy.deepcopy(ss)
        summary['consolidation_training'] = dict(protocol=copy.deepcopy(PROTOCOL),
            peak_lrs=peaks(cfg), scheduler_state=actual)
        return summary

    def validate_ranks(self, summaries):
        if (len(summaries) != 8
                or any(s.get('consolidation_training') != summaries[0].get('consolidation_training')
                       for s in summaries)
                or summaries[0].get('consolidation_training', {}).get('protocol') != PROTOCOL):
            raise ValueError('All8ranks must share the exact consolidation scheduler/Adam boundary')
        return super().validate_ranks(summaries)

    def recipe_identity(self, cfg, args, condition_sha):
        value = super().recipe_identity(cfg, args, condition_sha)
        value['consolidation_training_protocol'] = dict(**PROTOCOL,
            update_unit='completed optimizer updates u; scheduler.last_epoch=u; nextstep usesfactor(u)',
            warmup_formula='factor=.001+.999*u/1000 for0<=u<=1000',
            decay_formula='factor=.1+.9*(1+cos(pi*(u-1000)/19000))/2 for1000<u<20000',
            hold_formula='factor=.1 for20000<=u<=50000',
            peak_action_lr=cfg['learning_rate'], peak_memory_lr=cfg['memory_learning_rate'],
            initialization='same complete IL70; fresh Adam/RNG42/update0; not an older nativefork',
            native_resume='all optimizer/RNG/cursor/pending4inner material retained; no rewarmup or horizon reset',
            phase_policy='initial20k runtime cap; reviewed30/50k continuation uses the same20k decay and heldfloor')
        return value


def consolidation_runtime(cfg):
    validate_consolidation_recipe(cfg)
    return type('ConsolidationFreshRuntime',
        (ConsolidationRuntimeMixin, QualityDiagnosticsMixin, GroupRuntime), {})()
