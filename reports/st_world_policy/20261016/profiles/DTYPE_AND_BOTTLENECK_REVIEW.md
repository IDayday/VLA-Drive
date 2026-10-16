# Actual dtype, calls and bounded trace review

This is a trace interpretation, not a formal approval. Training/source: `ab2184bf66894c24b3d911f0e01c9b3c986d956d` (timing source58f85044 plus instrumentation). Corrected offline analyzer: `0c54ed2c8eb3dd14762ca5aa88c8c911d541f626`. Original traces remain unchanged. `trace_sp2_pca384_v2` summaries are superseded by `trace_sp2_pca384_v3`: Kineto emits CPU and GPU annotations with the same name, and they must not be added together as CPU time. A negative regression now enforces this distinction.

## Real execution precision

There are 2,959,630,724 unique parameters, of which 2,552,673,668 train and 406,957,056 are frozen. Execution parameters and buffers are BF16; the original frozen Qwen visual is eval/no-grad and emits BF16 merged and DeepStack tensors. This is the original S0/DiT training backend, not the historical Query Base FP32-visual experiment. Current DINO supervision and PCA are FP32.

Each of eight ZeRO2 ranks has 319,084,210 FP32 master elements and the same number of FP32 exp_avg/exp_avg_sq elements after a real update. Minor flattened partition padding is explicit. Captured `averaged_gradients` after backward are **BF16**, not FP32. Their absence in the final post-step inventory is buffer lifecycle, not missing gradient. All ranks agree. Native master-update checks and the independent correctness review establish actual updates/recovery; merely having a master tensor would not suffice.

At microbatch4, the frozen visual consumes 12 images and produces 6,912 merged tokens plus three 6,912-token DeepStack tensors. Language input is [4,1930,2048]. Each scene has 1,728 visual tokens and 144 W tokens. The action projection/DiT sees [32,8,...], preserving repeat8. Sparse readout emits [4,288,384], i.e. 96 sampled positions ×3 views per scene. Actual masks/layouts are passed; this hook did not measure their padding fraction, so padded length is not presented as valid-token count.

Exactly one top-level frozen-visual, language, action-projection, DiT and sparse-readout call was observed per rank per measured update. Internal checkpointed layers recompute during backward; top-level hook counts do not imply zero recomputation. This trace is training, not ten-step deployment sampling latency.

## Timing boundaries and work

Two optimizer updates (2 and3) were traced on all eight ranks. These are not 16 independent throughput samples. Timing results come from the separate unprofiled 240-update run. CPU/user annotations, CUDA events and GPU kernel interval unions have different boundaries and must not be added.

Across ranks and these two updates, representative current-stream CUDA-inclusive forward spans are:

| Region | Mean ms | Range ms |
|---|---:|---:|
| Frozen visual | 250.813 | 244.479–285.165 |
| Language forward | 201.363 | 186.119–412.447 |
| Action projection | 0.540 | 0.510–0.642 |
| DiT forward | 72.546 | 67.861–77.410 |
| Sparse readout | 6.559 | 6.065–7.349 |

The language outlier illustrates host/stream/profile effects; these spans are not disjoint attribution. CPU backward+recomputation annotation averages872.924ms; CPU data annotation819.679ms in this traced early window, including synchronization/trace overhead. In the unprofiled steady window, the actual slow-rank data-fetch field averages393.115ms. The trace data number must not replace that steady data-wait observation. Per-module backward, tokenizer, decode and H2D are not individually isolated by current coarse annotations.

GPU interval unions per rank/update: compute1055.823ms (1043.774–1090.576), NCCL490.739ms (159.579–883.632), overlap125.024ms, and NCCL active without simultaneous compute365.715ms (92.313–688.235). Total GPU-active union1454.182ms within1985.408ms activity span. These show real overlap and idle/launch/coordination gaps. NCCL-only activity is **not proof** that removing that entire interval would shorten the critical path; rank skew and dependencies require causal analysis. The literal tail after the last compute kernel is only0.012–0.263ms and must not be confused with earlier exposed communication.

The trace records per-rank logical NCCL input tensor bytes: all_reduce5,105,347,388; _all_gather_base638,168,420; small all_gather232. Input sizes are not network wire volume, and collective multiplicity/topology matters. Kernel names confirm NCCL ring LL collectives and PyTorch BF16 FlashAttention forward/backward. There are15,996–16,246 CUDA launches per rank/update. No compile/graph-break events were recorded; that does not establish a zero graph-break count.

The concrete work is concentrated in language/action backward+recomputation, frozen visual/language forward, and data/communication coordination. The new sparse readout is about6.6ms forward and is not a credible explanation of the whole update cost. There is no evidence here that shrinking it or reducing the original supervision would be an appropriate systems optimization. This review preserves the scientific definition.

## Evidence and limits

Detailed actual dtypes/shapes/call counts: `MODULE_DTYPE_AND_CALLS.json` in this report directory. Raw traces, all-rank dtypes and native complete checkpoint: `/mnt/project/st-world-policy-20261016/profiles/sp2_pca384_8gpu_trace_v2/runs/sp2_pca384_8gpu_trace_v2/`. Corrected per-trace interval summaries and SHA256: `/mnt/project/st-world-policy-20261016/audits/systems/trace_sp2_pca384_v3/`.

The trace weighs about1.22GB across16 files and is kept in runtime artifacts, not committed. No model weights, images or raw teacher tensors are in this report. Full population I/O, W96, historical input and FPO efficiency need their own concrete dependencies/evidence.
