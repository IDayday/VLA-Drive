# IQE fixed-commit review and Query Base result

The latest user-directed result is **Query Base step14000, full navtest PDMS85.2969/100**,12146 scenes/136 logs, zero rate8.4802%, no invalid score. [Full result/protocol](NAVTEST_REPORT.md); [raw aggregate/per-log scores](evidence/NAVTEST_NAVTEST_RESULT.json).

The original Base continues on training-vla-zt2; it was not restarted, hot-patched or interrupted. A separate finite observer waits for the already-running training job's20k/50k/100k complete checkpoints and idle authorized evaluation resources. No new full training or serving activation occurred.

| Review area | Status |
|---|---|
| R1–R6 source repairs and exact retrieval/config fixes | COMPLETE |
| Unit/regression suite |121 passed,1 skipped; separate authorized GPU/NCCL PASS |
| Real diagnostic incremental workflow |expert1→Scorer-only2→expert3 PASS; exact CPU resume, reference scores, frozen generators and bundle restore PASS |
| Base quality measurement |Full navtest COMPLETE; formal acceptance rules not preregistered |
| Expert round1 / Scorer / multi-round formal launch |Engineering PASS; formal launch withheld pending Base acceptance and its full locked bank |
| E3 real new observations |UNTESTED; no actual new observations supplied; E2 resampling is supported |
| E6 residual_v2 |Engineering READY, science UNTESTED; legacy algebraic residual is invalid ablation evidence |
| E12 aligned LoRA-DiT comparison |BLOCKED: no runnable aligned set bound |
| IQE scientific gain |UNTESTED; Base absolute PDMS is not IQE gain |

Start at [RUNBOOK](RUNBOOK.md) for actual configs, checkpoints, Base evaluation, qualification, expert/Scorer rounds, resume and bundle commands. [Fix matrix](FIX_MATRIX.md) maps source→tests→real evidence; [machine-readable readiness](TRAINING_READINESS.json) records individual gates. [Evidence index](EVIDENCE_INDEX.json) binds every committed raw evidence file. Weights, images, full scene banks and metric caches remain external.

Audit start:1e876da626b02af31459f44229248afa0cc55bed. Isolated branch:fix/iqe-formal-readiness-1e876da. No source-branch successor was folded in; original dirty workspaces and all protected Base model files are unchanged. All commits are local; no push or merge.
