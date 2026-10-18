"""Native Qwen/SourceFlow actor; shared source GRPO optimizer/transaction loop."""
from starVLA.rl.qwen_sourceflow.rl_runtime import NativeRuntime
from scripts.analysis.simwam_recipe.train import main

if __name__=='__main__':main(NativeRuntime())
