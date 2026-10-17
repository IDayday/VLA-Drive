# PCA256 channel-control independent evidence review

Reviewer: `/root/st_teacher_channel_evidence_v2`. Date: 2026-10-17 UTC.
Role: evidence only; not a method/configuration implementer, correctness approver,
resource approver, or experiment dispatcher.
Stable report worktree: `/mnt/project/VLA-Drive-st-teacher-channel-formal-evidence-20261017`,
branch `review/st-teacher-channel-formal-evidence-20261017`.

**Conclusion: PASS for the research evidence of this one registered 5k PCA256
diagnostic comparison.** This is a falsifiable channel-supervision recipe, not a
claim of lossless compression, reduced shared-gradient interference, or improved
driving. It does not authorize a launch by itself.

The reviewed registration is
`SP2_PCA256_fulltrain_seed42_official_5k_v1`, SHA256
`a644ad60477b5dd857b44069568e9b93fde68994b4862ffcbcaaae9b57e29ca6`.
Runtime commit is `fdf25f99b702754787904215bf1c2a2b7515ea75`; execution-source
fingerprint is `bfde9af74eb615f8236d44196776f043f6293df1a85a0c01c8735ff664adcaec`.
The full exact gate fingerprint is embedded in `EVIDENCE.json` and
`SMALL_ARTIFACT_AUDIT.json`.

## Evidence and adaptation

I independently opened the [DINO-Foresight full text, v2](https://arxiv.org/html/2412.11673v2),
specifically §3.1–3.3, equations (1)–(2), Appendix A.1/Table 5 and A.5/Table 10.
§3.1 applies PCA along feature channels while retaining spatial coordinates.
Equations (1)–(2) specify masked SmoothL1, with beta 0.1. Table 5 reports
short/mid-term segmentation mIoU 71.3/59.9 without PCA and 71.8/59.8 with PCA;
its evidence is not universal dominance. A.5 reports comparable loss variants.
This supports testing compact targets, not a theorem that 256 beats 384.

I reread and rehashed the official source at commit
`d98f765d2b743ec90fb38bc657a39c07c7ca4ab0`:
[pca.py](https://github.com/Sta8is/DINO-Foresight/blob/d98f765d2b743ec90fb38bc657a39c07c7ca4ab0/pca.py),
[src/dino_f.py](https://github.com/Sta8is/DINO-Foresight/blob/d98f765d2b743ec90fb38bc657a39c07c7ca4ab0/src/dino_f.py),
and [train.py](https://github.com/Sta8is/DINO-Foresight/blob/d98f765d2b743ec90fb38bc657a39c07c7ca4ab0/train.py).
The fitting path selects training images, standardizes channels, and calls PCA
without requesting whitening. `pca_transform`/`pca_inverse_transform` implement
that affine transform and its reconstruction; `calculate_loss` and
`forward_loss` use the functional losses' default mean reduction after masking.
The paper's displayed sum therefore does not determine our runtime denominator.
No upstream training was executed.

The project uses a single DINOv3 layer and sparse W readout, unlike the paper's
multilayer DINOv2 forecasting transformer. Its 256/384 widths and 5k comparison
are project hypotheses. Existing MSE remains fixed; SmoothL1, whitening, larger
readouts and gradient interventions are not part of this registration.

## Actual comparison and control reuse

Independent JSON comparison found exactly three configuration differences from
the completed SP2: output channels, their derived transform identity, and the
isolated output root. The prepared 384 config is byte-identical to the original
registered config. The new registered 256 config is byte-identical to the
prepared 256 config.

Both bind the same source, S0 Base, training-manifest path/hash, raw-teacher
path/hash, environment hash, PCA namespace, seed, optimizer and execution:
384×288 teacher pixels; native 18×24 grid without pooling; three original
views; W48/view; K96/view; current-only targets/input; global32,
micro4/rank, four ranks and accumulation2; warmup1000 and a 20k LR horizon with
a stop at 5000. Source inspection confirms original repeat8 FM and ten-step
single-trajectory deployment remain the guarded action contract. The student
input resolution is unchanged by this configuration.

The raw teacher stays 1024-channel. `TeacherPCA.transform` uses a prefix of the
same frozen basis; no spatial interpolation or averaging is introduced.
`STWorldTrainingDataset` selects the registered width, validates the resulting
contract and detaches targets; the sparse readout receives that width. The
train-only PCA identity is
`268723c8da3151492fd20a81b0f6f541f348a1187222132b055f4e4a528bfd46`, fitted on
32768 patches with channel standardization and no whitening. Both widths and
the native training cache are explicitly covered by its metadata. I independently
recomputed both projected transform identities and the PCA/scope JSON identities.

The original 384 dimension selection stays frozen at its original hash. This
256 run is a later user-requested diagnostic; it does not reopen selection for
all temporal cells or authorize a second full spatial/temporal matrix.

The existing 384 checkpoint receipt records 5000 updates and 159992 scene
presentations; its current-label element count is exactly
`159992 × 3 × 96 × 384`. Its initialization receipt matches the common Base
and reports `RESTORED_EXACT_FP32_SOURCE`. New 256 starts from that same original
S0 with a fresh optimizer, not from the trained 384 endpoint. No 384 retraining
is needed for this comparison. I checked its historical approval/review file
hashes and fingerprint crosslinks, not its multi-gigabyte checkpoint tensors.

## Mathematical and interpretation limits

For the runtime's valid-position count N, `masked_mse` and
`optimizer_batch_counts` give `L_C = SSE/(N*C)`, with the existing DDP
compensation. Therefore

```text
L384 = (256*MSE_common256 + 128*MSE_tail128)/384.
L256 = MSE_common256 for the 256 arm.
```

For the same retained residual, the 256 recipe's coefficient is 1.5 times the
384 recipe's coefficient. Dropping the tail also removes its gradient entirely.
This is one configured channel-width change with consequential loss-weight,
output-parameter and initialization differences; it does not isolate pure
capacity or prove that more channels create larger gradients. Raw whole-width
MSE values are not a valid winner criterion. The output projection has 65664
fewer parameters; the documented final output-bias draws are not prefix-equal.
Shared-parameter initialization equality is covered by the earlier author CPU
test, not rerun or asserted as a new numerical result here.

With an orthonormal retained basis U and standardized/centered target z-c:

```text
||z - (c + U*yhat)||²
  = ||(I-U*U^T)*(z-c)||² + ||U^T*(z-c)-yhat||².
```

This separates discarded information from retained-subspace prediction error in
the standardized coordinates. Original-1024-coordinate mean prediction
contribution is `(C/1024)*MSE_C`; the identity cannot simply be transferred
through nonuniform inverse channel scaling to raw-coordinate Euclidean MSE.
This algebra does not establish the student's best achievable error.

The existing 64-train/16-dev teacher probes support a bounded test: depth MAE is
5.332822m/5.197710m for 256/384, and small-vehicle PR-AUC is
0.916009/0.914696. The probes use the same linear architecture, 300 updates,
153600 presentations and identical per-task sample-order hashes. **Their
parameter counts are 257/385, not equal capacity in the strict parameter-count
sense.** The fixed dev subset already informed the original selection. Neither
these numbers nor retained PCA variance establish semantic equivalence,
statistical noninferiority, learned-W quality, or planning benefit.

The registered diagnostic interpretation remains appropriate: compare common
256-coordinate prediction errors on joined scene/view/patch identities; show
384's tail separately without treating 256's unpredicted tail as zero error;
evaluate the same frozen-W probes and fixed full-development single-trajectory
planning/safety protocol. No new metric threshold is invented, and Navtest must
not select this width. 256 downstream results remain NOT_RUN at this review.
The S0 parent has already seen training scenes; 5k is a mechanism-continuation
screen, not an untouched-data or convergence claim.

## Independent verification and boundary

`python3 -I -B reports/st_world_policy/20261017/teacher_channel_formal_evidence_v1/verify_small_artifacts.py`
passed. It uses only Python's standard library, caps affinity at two CPUs,
address space at 2GiB and CPU time at 30 seconds. The audit rehashed 578341 bytes
of small documents/source and examined 323064 bytes of Git-tree output. It
independently checked a clean exact runtime commit and recomputed the gate's
execution-source fingerprint. Earlier manual source/document reads were also
small; no large identity, checkpoint, model, image or PCA tensor was read.

The 30MB scene manifest and 146MB teacher identity were deliberately not
rehashed by this reviewer. Their exact paths/hashes match the completed 384
registration and approval. Lead separately reported complete
`verify_registration` hashing for the precise new registration; that full-hash
work is inherited and is not represented as my independent execution. The
small-file verifier consequently does not claim to replace the full gate.

No GPU, SSH, Torch import, formal task, new teacher/cache generation, or shared
implementation mutation occurred. The evidence PASS requires separately issued
correctness and efficiency reviews for the same fingerprint and Lead's final
unchanged gate/launch checks. It grants no broader budget or scientific result.

Workflow correction: the three draft report files were initially written under
the same owned report directory in Lead's worktree. On Lead's isolation reminder,
only those reviewer-owned files were moved into the normal checked-out independent
worktree above, before issuing or committing EVIDENCE.json. The original draft
directory was removed by that move; no other Lead file was changed. Final evidence
artifact paths point to this stable independent worktree.
