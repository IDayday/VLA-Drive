# Fresh FP32 master fix: which performance evidence can carry forward

Old whole-model timing sources: `58f85044b01f1336b8063a36d34fb16623dfdedb` and trace/W96 runtime `ab2184bf66894c24b3d911f0e01c9b3c986d956d`. New clean master-integration source: `2d3e4f5a99c248d5ff0b66791952e2d28def1d8b`.

The new helper captures the canonical FP32 trainable source before DeepSpeed's BF16 cast and restores the correct local FP32 master partitions once during fresh initialization. Native resume/explicit parent continuation must bypass this restoration. This repairs the source of the masters; it does not change BF16 execution, BF16 native gradient accumulation, FP32 Adam moments, ZeRO2, TF32-off flags, data, original DiT/repeat8, or the per-update objectives.

Systems compared the `train_world.run` optimizer-loop AST across ab2184bf and2d3e4f5a: the one `while` loop is identical. The main model change records history migration metadata only when an actual history encoder is supplied; current-only S-P2/S-P3 have none. Current training Dataset, original DDPForesight, sparse readout and loss files are unchanged. The PCA additions validate/clone startup namespaces and do not refit or alter the per-batch transform. Other new deployment/history/Flow modules do not execute inside these current-only W update loops.

Consequently the old real8-GPU and actual simultaneous2×4 results remain useful evidence about the same steady tensor shapes, kernel/backend work, communication topology and current-only loader capacity. They can support a new independent efficiency review with this explicit source delta and successful real-model qualification; there is no reason to repeat the entire topology search solely because of this helper.

The following **do not** carry forward as measured facts:

- Fresh loading/capture/restore latency, peak startup CPU/RAM or whole-allocation time. The bounded FP32 snapshot adds real startup work; new source measurements are required for these values.
- The old initialization's numerical correctness, losses, parameter updates or a bitwise optimization trajectory. Correct FP32 masters differ from BF16-rounded masters even when the initial BF16 execution tensor is unchanged; updates can diverge thereafter.
- A formal approval fingerprint, fresh source/checkpoint provenance or a new full-population manifest. Existing reports retain their actual source and data identities.
- C1/FPO throughput. C1 has4 rollouts, MC50 old/current computation and synchronous official feedback, which W profiles do not execute.
- Native recovery of the full new W/C1 models. The observed two-rank FusedAdam CUDA/master/native-roundtrip test is narrow; full true-image qualifications were still running when this registration was prepared.

Actual helper evidence: `/mnt/project/st-world-policy-20261016/audits/flow_policy/fp32_master_init_v1/native_cuda_actual_v1/native_rank0.json` and `native_rank1.json`. Both report exact FP32 restoration, unchanged BF16 execution and one real native FusedAdam update/roundtrip. Their model is small and their startup timings cannot stand in for the2.55B-trainable-parameter W model.

The separately prepared C1 short screen pins the new source and will measure its own pipeline. The registration records `formal_launch_allowed=false` and `weights_usable_for_formal_init=false`. The final formal reviewer must bind the actual clean source/config/data/resource identity; Systems has not signed that unobserved run.
