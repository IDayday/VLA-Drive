# GPU pressure jobs — 2026-09-07 UTC

User-requested occupancy only; these jobs started after all four GPU inference shards had finished. No training or scoring process was terminated. The original `/mnt/project/gpu_stress.py` was used without edits.

| Host | Parent PID | GPUs | Target allocation/GPU |
|---|---:|---|---:|
| training-vla-zt-worker-0 | 1835186 | 0–7 | 75 GiB |
| training-vla-zt2-worker-0 | 3147176 | 0–7 | 75 GiB |
| training-vla-zt3-worker-0 | 1117834 | 0–7 | 75 GiB |
| training-rl-zt4-worker-0 | 206926 | 0–7 | 75 GiB |

Each job is detached with `nohup setsid`. Logs on **each host**: `/tmp/gpu_stress_navtest_done_20260907.log`. PID values are host-local; verify command line and process group before stopping. Workers are child processes, so stopping a parent alone is insufficient.

Arguments: `--gpus all --memory-gb 75 --dtype float16 --matmul-size 8192 --duration 0 --status-interval 60 --yield-check-interval 2`. Nested CPU thread pools are one. All 32 devices reached approximately 77,337 MiB in `nvidia-smi`, including CUDA overhead, and all workers reported `ready`. GEMM iterations are increasing. Instantaneous utilization fluctuates during occupancy checks; a continuous 100% utilization guarantee is not claimed.

The original script checks for non-pressure CUDA processes every two seconds and yields by exiting the corresponding worker. It is not an automatic restart service. Stop the pressure process group explicitly before large GPU allocations; do not rely on allocating the training model successfully into the approximately 3.7 GiB left free.
