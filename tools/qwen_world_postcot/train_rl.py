"""Real canonical-PDMS clean-x Flow-GRPO after identified full-framework IL.

Uses the existing native optimizer/checkpoint/chain trainer. Formal startup is
uses the complete100k/DEV selection by default. A separate sealed explicit-user
IL40k parent receipt is accepted without pretending it completed that selection.
--qualification weights never initialize the formal model.
"""
from scripts.analysis.simwam_recipe.train import main
from starVLA.rl.qwen_world_postcot.rl_runtime import CleanXRuntime


if __name__ == '__main__':
    main(CleanXRuntime())
