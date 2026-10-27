"""Explicit World IL40k memory/time scope variants; reuse the native GRPO engine."""
from scripts.analysis.simwam_recipe.train import main
from starVLA.rl.qwen_world_postcot.scope_runtime import ScopeRuntime

if __name__=='__main__':main(ScopeRuntime())
