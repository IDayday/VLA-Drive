"""Registered native Qwen-Flow safety experiments; existing trainer reused."""
from starVLA.rl.qwen_sourceflow.safe_runtime import SafeRuntime
from scripts.analysis.simwam_recipe.train import main

if __name__ == '__main__':
    main(SafeRuntime())
