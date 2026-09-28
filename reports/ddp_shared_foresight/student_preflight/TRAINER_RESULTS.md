# Real student startup and resume

Training source: `087139aa4df4109e79c29a35b8e4cbdbd47c59d0`.

Actual public Qwen3-VL2B plus random original DDP1536-wide/24-layer action head and64W; frozen native vision, trainable language/state/action/W. Real synchronous three-front camera inputs and real ego labels. No teacher/visual auxiliary label in this A-only trainer test. Two local A800 GPUs, BF16 model with FP32 Adam masters, global batch4, microbatch1, FM repeat8; deterministic setting CUBLAS_WORKSPACE_CONFIG=:4096:8, TF32off. Diagnostic schedule100updates/warmup1,4updates performed. This is optimizer/resume verification, not the formal schedule or planning evidence.

Continuous4 and independent2+2 from identical generic/random initialization: all model tensors, all FP32 optimizer/moment tensors, scheduler/progress and both ranks' Python/NumPy/Torch/CUDA/noise/time/horizon RNG exactly equal. Serialized DeepSpeed LossScaler fields are compared by value. Eight actual optimizer updates across the two runs,32scene presentations total; no synthetic updates. Each run has16presentations drawn from the same64training-scene prefix. `REAL_RESUME.json` records the full comparison. A first CPU-only verifier was aborted to replace network mmap reads with sequential reads and compare LossScaler state rather than Python object addresses; neither training run was changed/repeated.

Losses in both runs:1.59094,1.42254,1.50801,1.28697. Every optimizer step changes sampled FP32 master elements; no claim of convergence from four steps. Peak recorded allocated GPU memory27,521,296,128bytes (resumed). Steady update times around1.5–1.9s at this tiny batch; full global-batch32 throughput remains to be measured before the formal plan is locked.

Separate actual2-rank NCCL test validates empty auxiliary rank and global normalization across gradient accumulation. Expected/direct/accumulated gradient25in all cases (`../distributed/NCCL.json`).14focused CPU tests pass.

Actual public models R/A/B/C/D were constructed independently. Every common action/state/language tensor matches, A/B/C/D Wmatches, construction preserves the caller RNG. `INITIALIZATION_PAIRING.json` includes full training and trainable parameter counts. All driving modules are random; no historical driving checkpoint was loaded. This is additional evidence to the pinned public Qwen hash manifest.

Current/ego split preparation completed101592training scenes/1176logs and1696development scenes/16logs,0failures; identities/counts in `DATA_COMPLETION.json`. Auxiliary labels do not affect this shared scene population. Future VAE cache remains blocked by official FLUX access401; formal students and Navtest NOT_RUN. Full random-initialized GT teacher remains running independently from frozen sourced3c05d1.

To reproduce the real startup use the tested frozen source087139a, configA.yaml, FORESIGHT_QWEN and FORESIGHT_SOURCES pointing to verified generic artifacts:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 torchrun --standalone --nproc_per_node=2 \
  -m tools.foresight.train_student --config configs/foresight/A.yaml \
  --data "$FORESIGHT_ARTIFACTS/student_train_v1" \
  --campaign-root "$FORESIGHT_ARTIFACTS" --run-id new_student_resume_check \
  --scope startup --limit 64 --global-batch 4 --micro-batch 1 \
  --updates 4 --schedule-updates 100 --warmup 1 --save-every 100 \
  --milestones '' --max-seconds 1800 --campaign-gpu-hours 120 --deterministic --stop-after 2
```

Resume the same command with `--resume --acknowledge-stop` and without `--stop-after`.
