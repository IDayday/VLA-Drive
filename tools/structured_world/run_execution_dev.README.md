# Canonical execution dev observations

`run_execution_dev.py` is a thin local-host launcher around the unchanged
`infer_checkpoint.py` and `score_navsim.py` from frozen canonical commit
`0c54f1c49e7ed2a76b22826b654d6c72625f5b6d`. It accepts native be5 +500
checkpoints, restores FP32 masters strictly, and retains FP32 inference, TF32 off,
scene seed42, the original ten-step sampler, q0 and q_final. It does not install
an execution adapter into the inference model. It does not import a model while
planning. Parent orchestration routes commands to the actual host.

CPU-only plan, using the populated artifact manifest:

```bash
CUDA_VISIBLE_DEVICES='' /usr/bin/python3 -B tools/structured_world/run_execution_dev.py \
  --manifest /mnt/project/structured-world-fgtr-round1-artifacts/20261008/perf_execution_dev_20261011/MANIFEST_P0_v1.json \
  --output /absolute/new/plan
```

After freezing the driver and admitting actual idle devices, the operator can
add `--execute --gpu-uuids GPU-full-uuid[,GPU-full-uuid]`, selecting
`--groups G1` or `G3` and `--populations dev128` or `dev1696`. `--phase both`
executes inference then CPU scoring per bank when both phases belong to the
actual host. No process or GPU count is fixed at eight. Each selected GPU has
one original torchrun model process and independently reconstructs the complete
native FP32 model; admission must include host RAM and measured GPU peak memory.
The manifest memory threshold is an explicit conservative guard, not a measured
model requirement. Repeated GPU samples reject any selected compute app;
occupied unselected GPUs are not classified or stopped.

For separately routed CPU scoring, execute `--phase infer` on the native host,
then `--phase score --inference-results-root /earlier/infer/output/results` on
the manifest score host with a new output directory. That host must see the
prediction bank and its compact `INFERENCE_RECEIPT.json`/`NATIVE_BINDING.json`.
No weight copying or FP32 exporter is implemented. G1 planned endpoints are
zt3-local +25500; G3 planned +50500 endpoints use the existing shared run root.
Planning paths does not establish endpoint completion or host reachability.

The actual manifest binds both current-input banks' identity/index/COMPLETE
bytes and the original metric indices. dev128 uses `--population debug` and
cannot supply full dev evidence. dev1696 requires all 1696 tokens, all16 logs,
both trajectory banks, zero failures, pinned canonical backend/runtime, and CSV
means matching the saved summaries. The original scorer waits for a complete
prediction bank; this bounded helper does not add streaming or multi-host pools.
CPU worker count is explicit and bounded to32. Choose it from currently spare
CPU affinity/quota/load, reserving capacity for training and image processing.
The helper's CPU pool lock coordinates cooperating launches on one host only;
it cannot reserve capacity against unrelated workloads.

GPU leases retain the original benchmark's hash-of-qualified-pool lock plus
selected UUID locks under the SAME `artifact_root/physical_gpu_locks`. This
prevents overlap with the existing full-pool native scheduler; it conservatively
serializes cooperating dev subsets on that qualified host. Locks are inherited
by the owned child process. There is no SSH dispatch, retry, resume, preemption,
filler restoration, or process signaling. Infrastructure failures retain logs
and need parent handling; fresh outputs are required rather than overwriting
historical attempts.

Every executed child uses the existing `bench.run_job`, preserving actual
`PROCESS.json` PID and `PROCESS_EXIT.json`. `*_PROCESS_COST.json` charges entire
wall time, including Python startup, native hashing/master restore, inference,
and exit, even on worker failure. GPU-hours use the actual selected device count;
CPU scoring has zero allocated GPU-hours. Original inference COMPLETE's inner
loop cost remains unchanged and is not substituted for whole-process cost.
Failure before a child starts creates no claimed process cost. Partial attempts
retain their individual costs even without dispatch COMPLETE.

For E3, make a separate manifest with `candidate_mode: precision_fast`, assign
the actual B500 path, and set A/AA `reuse_dev` per population to explicit
`{path, sha256}` references to earlier `DEV_RECEIPT.json`. Reused references must
retain their actual IO run/tag/identity/native payload binding. The helper
rehashes prediction payloads and canonical summaries/CSVs. It rejects B reuse.
New GPU cost of reuse is zero; previous inference/scoring costs remain recorded
in the bound earlier receipt. No A/AA reference is guessed from an old name.

Outputs are `INFERENCE_RECEIPT.json`, `DEV_RECEIPT.json` and PROCESS costs. Their
`protocol_complete` states observation validity, not optimization acceptance.
`passed` and `formal_authorized` remain false; historical strict failures are
untouched. To supply existing V1 evidence, bind actual q0/q_final CSV/summary
refs for A/AA/B, the original full score index and pinned `P0_V1_DEV_PROTOCOL`,
then match each bank's checkpoint ancestry to the actual +500 native comparator.
Do not copy prospective ancestry from a manifest into an observed receipt.
The helper hash is an execution-sidecar provenance record; the canonical dev
producer remains unchanged `score_navsim.py` SHA754f9c. Existing native and
forward bindings are not rewritten by this dev preparation.

CPU contract receipt:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /tmp/structured-world-round1/envs/qwen/bin/python -B \
  tools/structured_world/test_run_execution_dev.py --receipt /absolute/new/CPU_TEST.json
```

Tests use owned temporary CPU children, locks, tiny metadata and synthetic
trajectories. They do not load checkpoints, run training/inference/scoring,
signal processes, or constitute scientific numerical/performance evidence.
