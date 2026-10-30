"""Independent PPO_SAFETY_EP entry; no old-source monkeypatch or reward transform.

torchrun ... -m starVLA.rl.qwen_world_research_algorithms.safety_efficiency_entry
uses the ordinary research train arguments and a separately sealed recipe.
"""
import copy

from .safety_efficiency import (
    STRUCTURED_PROTOCOL, finalize_safety_efficiency_advantages,
    validate_safety_efficiency_boundary, validate_safety_efficiency_ranks,
    validate_safety_efficiency_recipe,
)


class SafetyEfficiencyRuntimeMixin:
    finalize_advantages = staticmethod(finalize_safety_efficiency_advantages)
    validate_boundary = staticmethod(validate_safety_efficiency_boundary)
    validate_ranks = staticmethod(validate_safety_efficiency_ranks)

    def load_recipe(self, args):
        return validate_safety_efficiency_recipe(super().load_recipe(args))

    def recipe_identity(self, cfg, args, condition_sha):
        value = super().recipe_identity(cfg, args, condition_sha)
        value['structured_advantage_protocol'] = copy.deepcopy(STRUCTURED_PROTOCOL)
        return value


def main():
    from scripts.analysis.simwam_recipe.train import main as train
    from starVLA.rl.qwen_world_research.runtime import ResearchRuntime

    class SafetyEfficiencyRuntime(SafetyEfficiencyRuntimeMixin, ResearchRuntime):
        pass

    train(SafetyEfficiencyRuntime())


if __name__ == '__main__':
    main()
