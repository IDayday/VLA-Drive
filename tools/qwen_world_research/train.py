"""Canonical/CISPO training with explicit new-population native ancestry."""
from scripts.analysis.simwam_recipe.train import main
from starVLA.rl.qwen_world_research.runtime import ResearchRuntime

if __name__ == '__main__':
    main(ResearchRuntime())
