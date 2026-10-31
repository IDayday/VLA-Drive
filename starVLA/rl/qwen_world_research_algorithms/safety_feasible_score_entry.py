"""Unregistered freshIL70 canonical-feasible-quality PPO candidate entry."""
import copy
from .safety_feasible_score import (
    PROTOCOL,validate_recipe,finalize_advantages,validate_boundary,validate_ranks,
)


class SafetyFeasibleScoreRuntimeMixin:
    finalize_advantages=staticmethod(finalize_advantages)
    validate_boundary=staticmethod(validate_boundary)
    validate_ranks=staticmethod(validate_ranks)

    def load_recipe(self,args):return validate_recipe(super().load_recipe(args))

    def recipe_identity(self,cfg,args,condition_sha):
        value=super().recipe_identity(cfg,args,condition_sha)
        value['structured_advantage_protocol']=copy.deepcopy(PROTOCOL)
        return value


def main():
    from scripts.analysis.simwam_recipe.train import main as train
    from starVLA.rl.qwen_world_research.runtime import ResearchRuntime
    class Runtime(SafetyFeasibleScoreRuntimeMixin,ResearchRuntime):pass
    train(Runtime())


if __name__=='__main__':main()
