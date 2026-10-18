# Independent native-resume correctness review

Frozen controller: `19a5a244cca2a07600339ba9be7c3a150ef61285`, SHA256 `693d9307033efeca90da059ce4b6a75a35537f99ddf90dac092f96e356cf618e`. Original trainer/runtime remains clean `751fcfa2b41549ae16f7b3204aef8dd18b912016`.

`CONTRACT.json` and its budget/counter supplements predeclare the actual evidence needed. `REVIEW.json` records the source-only review; `BOUND_REVIEW.json` adds the later exact small metadata binding `0a6e0eb53db85a946ccae9c9bed86ccce89eb90f5e466c90aa700df2fad7177c`. This does not revise the earlier review timestamp or pretend a GPU resume already happened.

The single final narrow run passed **51 tests** (38 implementation fixtures,13 independent counterexamples), including a real two-process CPU/Gloo exception vote. Wall8.71s, waited-child CPU9.15s, sampled aggregate RSS1,890,631,680 bytes. No GPU, official scores, real native tensors or image/GT/context payloads were read. `NARROW_TEST.log` preserves the complete output. These tests qualify observer behavior; they cannot substitute for the forthcoming real native save/load.

Independent counterexamples reject changes in module values, FP32 masters, Adam moments, optimizer groups and persistent counters. They check BF16 raw bytes, noncontiguous values, deep-frozen client state, RNG/gradient/mode preservation and original save/load return behavior. A rank-local malformed client exits both Gloo ranks before either calls original save.

The final pause test replaces the owned marker before archive and adds a still newer operator STOP after archive. Both are preserved, no acknowledgement is emitted, and launch is rejected. The actual worker always uses `acknowledge_stop=False`; original751 protects any later STOP.

Extra RNG getter reads are explicit observation. A small existing fixture measured five getter calls and exactly one original save/load/setter. Per native save cycle these are two capture checks, one after hashing, and two post-restore checks. A direct resume restore adds two getter reads. Actual GPU counts remain pending real completed worker receipts; no getter draws a random number.

Bound approval permits only Lead's exact8-update/1024-call/no-retry SMOKE with disposable diagnostic weights, conditional on the matching independent Systems review and fresh controller resource checks. It does not authorize a new formal-training branch or alter an active job. Full-state hash equality is strict; independent BF16 A/A differences remain separate and cannot be hidden by huge-parameter relative error or a post-hoc tolerance. No full FP64 update-delta/cosine measurement is claimed.

Reproduce the narrow tests on the unchanged final source:

```bash
cd /mnt/project/VLA-Drive-st-rollout-native-resume-correctness-20261018
REVIEW_EXPECTED_COMMIT=19a5a244cca2a07600339ba9be7c3a150ef61285 /root/miniconda3/envs/ddp/bin/python reports/st_world_policy/20261018/rollout_scale_native_resume_correctness_v1/run_narrow_review.py
```

Do not use this command as an experiment launcher. Lead owns the separate actual dispatch and subsequent evaluation of SAVE2→LOAD2, consumed normalization/RNG/cursor, actual next IDs and prefix/suffix replay artifacts.
