# Learned-W endpoint preparation: independent review

**CHANGES_REQUESTED** at `278f55c73253c34bb7013a046ae6f90b4293d626`. The scoped CPU/math and fixed metadata checks passed. One new protocol-pin failure remains at this source. This does not revise previous real native training results or authorize a probe launch.

## WPROBE-PIN-02

`tools/st_world_policy/spatial_probe.py:48–54` hashes a pinned path then reopens it with `read_text()`. The independent pure-JSON counterexample atomically replaces the protocol after the hash read. The replacement has a valid semantic identity and another valid label-manifest pin. `load_protocol` accepts the replacement despite the externally approved SHA referring to the old protocol.

No actual asset was changed. The fix belongs to Lead: a bounded single read whose exact bytes are both hashed and parsed, retaining all role/recipe/schema checks. Actual controls are small: protocol1527B, label manifest128955B, train scene list19746B, dev list5534B, training registrations2422–2426B. A1MiB control bound covers these registered artifacts without reading native or W payloads.

## Current evidence

- Existing owner plus prior independent CPU cases:20 passed in8.66s.
- New independent cases:4 passed,1 counterexample failed in5.01s. Passed cases check role leakage rejection, changed-recipe rejection, exact scene/view/pixel target gathering, and an empty exposure after Adam state exists.
- CPU affinity0–1, thread caps2,120s per command; no GPU or real image/native/label-array reads.
- Actual fixed metadata:64train scenes in64logs,16dev scenes in16logs; no token/log overlap. All label joins, source hash fields, current timestamps and three-view records match.
- Metadata counts: train depth64067, small positive1231/negative18232; dev depth15922, positive291/negative6889. These are current manifest counts, not a new raw label reconstruction.
- Actual protocol SHA is `7e2dcd4a938dba575f4000f641f59e2cd48a80ad1ece190b0a31000596fc956e`, matching HANDOFF. Semantic identity is `800f88595587a6f2ed7156e7ed86cd5bbf6cf130596845dee7af0dda30a37c82`; different hash domains explain their different strings.
- The shared fresh H2048 readout has495362 parameters and identical initial state across W144/288. Input length changes computation, not capacity.
- Native18x24 center queries stay at t=0. Two independent task-valid means are summed1+1. Statistics use train labels only. Empty cycles consume exposure but do not update or decay Adam state. The fixed300 count means cycles; true optimizer updates are explicitly available in exposure records.
- W-only extraction, frozen buffer check, CPU strict resume, two durable slots under interrupted pointer publication, and terminal no-new-update behavior passed. Evaluation retains empty populations and valid denominators, with no dev checkpoint selection.

No actual5000 endpoint has been loaded or fitted here. No scientific fit, convergence, generalization, throughput or launch PASS is granted. The64/16 protocol remains a bounded capacity diagnostic on previously observed development scenes, separate from full1696 planning evaluation.

Evidence and exact hashes are in `REVIEW_v1.json`; private root `/mnt/project/st-world-policy-20261016/audits/correctness/learned_w_probe_endpoint_preparation_v1/`.

```bash
cd /mnt/project/VLA-Drive-st-correctness-learned-w-probe-20261017
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
 timeout 120s taskset -c 0,1 /root/miniconda3/envs/ddp/bin/python -m pytest \
 tests/st_world/review_learned_w_probe_v3/test_boundaries.py -q
```

The reviewed source yields4 passed and1 failed. Preserve that evidence; after Lead's fix rerun only the affected pinned-reader boundary.
