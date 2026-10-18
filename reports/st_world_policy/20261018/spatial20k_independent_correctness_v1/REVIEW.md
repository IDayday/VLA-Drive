# SP2/SP3 first 20k continuation: independent correctness

Reviewer: `/root/spatial20k_correctness`. Review workspace starts at
`99e1c6f7f6674789225795ae3641dcb10a29cd5e`, isolated branch
`review/st-spatial20k-correctness-20261018`. Actual training runtime remains
`/mnt/project/VLA-Drive-st-official-runtime-v2-20261016`,
`fdf25f99b702754787904215bf1c2a2b7515ea75`.

**PASS for the two exact new fingerprints, correctness role only.** No job was
dispatched by this reviewer. Approval assembly, other independent roles and
fresh launch-time GPU/storage checks remain with the Lead.

The source/trainer, original Flow loss, teacher, inputs, precision, optimizer,
global batch 32, microbatch 4, world size 4, accumulation 2 and original 20000
update scheduler are unchanged. Both new configs exactly match the prepared
proposals and their respective parent recipes except the explicit continuation
binding and stop boundary 5000 → 20000. Each uses its own complete parent5000,
not the other arm or a profile model. Each may add at most 15000 updates.

The actual native verifier checked both registration fingerprints and all
registered metadata dependencies. The review also checked parent registration,
training identity, complete receipt digest/state, exact native shard names and
sizes, and original four-rank evidence. No serialized checkpoint payload was
read or rehashed. The unchanged trainer still performs genuine complete
payload verification before native model/master/Adam/RNG loading. No reduced
strictness or `strict=False` was introduced.

The new first-invocation launcher preserves the old native gate, per-GPU FD
locks, protected occupied/unknown GPUs, output/LAUNCH duplication checks,
thread/backend environment and 28800 second trainer bound. It uses no
`--resume`: `config.continuation` selects the exact parent. The outer timeout is
29400 seconds with a further 180 second termination grace. New lease and
checkpoint commitments must span that entire interval. The launcher checks
4143 GiB of retained/new checkpoint obligations, the separate 1024 GiB free
floor, outstanding cache growth and conservative metadata allowance. It does
not delete old data or checkpoints and does not auto-resume.

CPU-only tests execute the launcher's actual parsed body, redirecting three
filesystem constants to temporary fixtures; gate/GPU/process APIs are explicit
test doubles. One positive case and 25 rejection cases passed. Cases cover
parent hash, recipe, global batch, world size, LR clock, stop limit, lease
identity/time, disallowed host, occupied GPU/UUID, lock contention, stale
approval/CP, CP total/hash, insufficient cache/metadata headroom and duplicate
output/LAUNCH. The positive case verifies exactly four inherited FDs, original
argv/backend flags and consumption of an existing external approval. These
tests are launcher tests, not substitutes for the inherited real model restore.

An initial test run correctly rejected a changed launcher checksum: the Lead
added parent/lease identity and metadata-space checks while the draft was under
review. `REJECTED_STALE_LAUNCHER_PIN.log` preserves that observation. After
reviewing the local additions, the pinned final launcher SHA is
`e175da58b7e9f0b128c3059c13ab3a2dd7b0c65063daead80a6253165efe79e2`.
No production defect was concealed by the test-pin refresh.

Actual metadata checks ran on 2026-10-18 after both registrations existed:

```bash
cd /mnt/project/VLA-Drive-st-spatial20k-correctness-20261018
PYTHONDONTWRITEBYTECODE=1 python -B reports/st_world_policy/20261018/spatial20k_independent_correctness_v1/test_launcher.py
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python -B reports/st_world_policy/20261018/spatial20k_independent_correctness_v1/review_actual.py --registration /mnt/project/st-world-policy-20261016/formal/spatial/registrations/SP2_fulltrain_seed42_official_20k_v1/REGISTRATION.json
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python -B reports/st_world_policy/20261018/spatial20k_independent_correctness_v1/review_actual.py --registration /mnt/project/st-world-policy-20261016/formal/spatial/registrations/SP3_fulltrain_seed42_official_20k_v1/REGISTRATION.json
```

The actual review script creates immutable per-arm report directories and will
refuse an overwrite. It is an audit command, not a training/recovery command.
`SP2/AUDIT.json` and `SP3/AUDIT.json` record the new fingerprints and exact
parent states. `SP2/CORRECTNESS.json` and `SP3/CORRECTNESS.json` are the separate
independent native-role receipts.

Existing four-rank real native restore and subsequent sample/LR/cursor evidence
is inherited by checksum. Production FlashAttention2 A/A backward variability
remains documented: exact restoration of serialized state does not promise
bitwise identical future cross-process gradients. This review changes no
tolerance and makes no claim about 20k scientific benefit before training.
