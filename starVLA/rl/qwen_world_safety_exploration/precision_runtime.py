"""FP32 action output compute in current/reference, collection and deployment."""
import copy

from .group_runtime import GroupRuntime
from .quality import QualityDiagnosticsMixin
from .standardized_runtime import StandardizedRuntimeMixin
from .precision_recipe import CONFIG, validate_precision_recipe


def enable_projection_precision(actor):
    for head in (actor.action_expert, actor.ref_action_expert):
        if getattr(head, 'output_projection_fp32', None) is not False:
            raise ValueError('FP32 study must start from the original BF16-compute parent')
        head.output_projection_fp32 = True


class PrecisionRuntimeMixin(StandardizedRuntimeMixin):
    recipe_validator = staticmethod(validate_precision_recipe)

    def make_actor(self, cfg, args, device):
        actor, a, b = super().make_actor(cfg, args, device)
        enable_projection_precision(actor)
        return actor, a, b

    def recipe_identity(self, cfg, args, condition_sha):
        identity = super().recipe_identity(cfg, args, condition_sha)
        identity['output_projection_precision'] = dict(**copy.deepcopy(CONFIG),
            unchanged='same FP32 stored weights, scope, sampler, SafetyEP advantages, PPO and IL mean anchor',
            changed='final matrix multiplication runs outside autocast on FP32 hidden activations',
            ancestry='same IL70 weights; initial executed function differs, so native0 is evaluated',
            primary='https://arxiv.org/abs/2510.13786',
            limitation='LM logits evidence motivates a driving action-head hypothesis, not a guaranteed transfer')
        return identity


def precision_runtime(cfg):
    validate_precision_recipe(cfg)
    return type('FP32OutputRuntime',
                (PrecisionRuntimeMixin, QualityDiagnosticsMixin, GroupRuntime), {})()
