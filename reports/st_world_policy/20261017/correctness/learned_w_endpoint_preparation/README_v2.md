# Learned-W endpoint preparation: pin-fix follow-up

**PASS_LIMITED_CODE_CPU_METADATA** at `a06b59e6289a19b05f4c1406d201cb8182dc2b65`. This is independent code/CPU/metadata review, not approval of a real extraction, fit, profile or formal dispatch.

WPROBE-PIN-02 is closed. The protocol reader reads one regular control file with a1MiB+1 bounded read, checks fd/path versions and the SHA of those exact bytes, then parses the same bytes. The original atomic protocol replacement with a new label source now rejects. Exact1MiB accepts; oversize and growth reject; ordinary input is opened only once.

Current independent narrow result: **5 passed,4 deselected,0 failed,0 skipped in3.41s**, CPU affinity0–1/thread caps2/120s limit. No GPU or actual image/native/teacher work. The old counterexample failure and v1 `CHANGES_REQUESTED` report remain intact.

Only `pinned` changed among34 function/class definitions; all other33 are AST-identical. Relevant readout, model/schema, observation/native loader, initialization and storage dependencies are unchanged. The earlier20 passing CPU cases,4 new passing non-pin cases and actual64/16 metadata verification therefore remain applicable without rerunning them.

The frozen protocol file SHA remains `7e2dcd4a938dba575f4000f641f59e2cd48a80ad1ece190b0a31000596fc956e`; its semantic identity remains `800f88595587a6f2ed7156e7ed86cd5bbf6cf130596845dee7af0dda30a37c82`. These are distinct hash domains, not a mismatch. The historical HANDOFF source hash describes the prior implementation; the new source hash is recorded in `REVIEW_v2.json` and Lead's pin-fix report. No label, recipe or training change is hidden by that update.

The fixed protocol still uses64train/64logs and16dev/16logs with no overlap; a fresh495362-parameter shared two-output readout; W144/288 with identical initialization/capacity; only t=0 native18x24 queries; two task-valid means weighted1+1; train-only statistics;300 exposure cycles with actual AdamW updates separately recorded. Ordinary CPU resume, interrupted two-slot publication, terminal no-new-update and empty-existing-Adam behavior remain covered. Raw-label construction/calibration evidence is inherited by fixed hashes, not rerun here.

Real5000 endpoint loading, genuine image-to-W extraction, frozen-W fit, scientific capability and measured cost remain pending. Lead must bind actual endpoints, source/runtime and cache/fit identities before dispatch. The diagnostic remains separate from complete1696-scene PDMS and does not claim an untouched test or convergence.

```bash
cd /mnt/project/VLA-Drive-st-correctness-learned-w-pin-20261017
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
 timeout 120s taskset -c 0,1 /root/miniconda3/envs/ddp/bin/python -m pytest \
 tests/st_world/review_learned_w_probe_v3/test_boundaries.py \
 -k 'pinned_control or protocol_bytes' -q
```

Private equivalent: `/mnt/project/st-world-policy-20261016/audits/correctness/LEARNED_W_ENDPOINT_PREPARATION_REVIEW_v2.json`.
