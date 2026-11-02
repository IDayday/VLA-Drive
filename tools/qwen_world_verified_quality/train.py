"""Use the original native Trainer with the explicitly independent S6 runtime."""
from starVLA.rl.qwen_world_verified_quality.runtime import VerifiedQualityRuntime
from tools.qwen_world_safety_exploration.train import main

if __name__=='__main__':main(runtime=VerifiedQualityRuntime())
