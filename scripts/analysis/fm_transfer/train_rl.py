"""The existing source Flow-GRPO loop, with an explicit qualified new SFT base."""
from starVLA.rl.fm_transfer.rl_runtime import TransferRuntime
from scripts.analysis.simwam_recipe.train import main


if __name__ == '__main__':
    main(runtime=TransferRuntime())
