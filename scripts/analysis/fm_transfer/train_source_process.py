"""Unchanged source RL loop with separately registered source data and duration."""
from starVLA.rl.fm_transfer.source_process import SourceProcessRuntime
from scripts.analysis.simwam_recipe.train import main

if __name__ == '__main__':
    main(runtime=SourceProcessRuntime())
