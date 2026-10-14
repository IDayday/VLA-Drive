# Geometry diagnostic

Working directory `/mnt/project/DriveVLA-M0-iqe-robust-scorer-20261014`;
interpreter `/root/miniconda3/envs/ddp/bin/python`.

Read-only64-scene audit, no optimizer or VLM execution:

```bash
OMP_NUM_THREADS=2 IQE_SPATIAL_READER_MODE=mmap_v1 /root/miniconda3/envs/ddp/bin/python -m iqe.geometric_readout.audit \
  --source-campaign /mnt/project/iqe-specialist-scorer-study-20261014/base70k_v1 \
  --output /mnt/project/iqe-geometric-readout-20261014/v1/geometry_audit --max-scenes 64
```

Register and train the single diagnostic on a freshly verified free GPU of
permanent `training-rl-zt2`. `bind` uses the same arguments without `--device`.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 IQE_SPATIAL_READER_MODE=mmap_v1 \
/root/miniconda3/envs/ddp/bin/python -m iqe.geometric_readout.diagnostic train \
  --source-campaign /mnt/project/iqe-specialist-scorer-study-20261014/base70k_v1 \
  --control-root /mnt/project/iqe-conditional-readout-20261014/v1 \
  --geometry-audit /mnt/project/iqe-geometric-readout-20261014/v1/geometry_audit \
  --output /mnt/project/iqe-geometric-readout-20261014/v1/diagnostic --device cuda:0
```

For a native interruption, append `--resume auto` to that exact command. The
300-step endpoint does not extend on resume. A diagnostic checkpoint cannot be
exported through this entry as a formal Scorer. No formal follow-up is started
by these commands.

Tests:

```bash
OMP_NUM_THREADS=2 /root/miniconda3/envs/ddp/bin/python -m pytest tests/iqe/test_geometric_readout.py -q
```
