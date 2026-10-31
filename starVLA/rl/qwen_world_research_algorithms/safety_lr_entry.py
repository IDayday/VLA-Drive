"""Exact-native Safety10k LOW/COS LR forks; no source policy/reward edits."""
from .safety_lr import SafetyLRRuntimeMixin
from .safety_efficiency_entry import SafetyEfficiencyRuntimeMixin


def main():
    from scripts.analysis.simwam_recipe.train import main as train
    from starVLA.rl.qwen_world_research.runtime import ResearchRuntime

    class SafetyLRRuntime(SafetyLRRuntimeMixin,SafetyEfficiencyRuntimeMixin,ResearchRuntime):
        pass

    train(SafetyLRRuntime())


if __name__=='__main__':main()
