"""Thin adapter: no second copy of the native trainer."""
from starVLA.rl.qwen_world_performance.batch_runtime import PerformanceRuntime
from tools.qwen_world_safety_exploration.train import main

if __name__=='__main__':
    main(runtime=PerformanceRuntime())
