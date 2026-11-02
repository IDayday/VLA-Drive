"""Use the unchanged shared native Trainer with the S7 runtime."""
from starVLA.rl.qwen_world_structured_progress.runtime import StructuredProgressRuntime
from tools.qwen_world_safety_exploration.train import main

if __name__ == "__main__": main(runtime=StructuredProgressRuntime())
