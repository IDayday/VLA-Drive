"""Real canonical-PDMS clean-x Flow-GRPO after full-framework IL+DEV selection.

Uses the existing native optimizer/checkpoint/chain trainer. Formal startup is
blocked until complete100k IL and its complete physical-log-disjoint DEV window
are sealed. --qualification permits engineering qualification only, never a
formal-model initialization shortcut.
"""
from scripts.analysis.simwam_recipe.train import main
from starVLA.rl.qwen_world_postcot.rl_runtime import CleanXRuntime


if __name__ == '__main__':
    main(CleanXRuntime())
