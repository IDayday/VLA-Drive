# Follow-up: strict path-resolution fix independently verified

Reviewer `/root/spatial20k_correctness`, 2026-10-18 UTC. This is an append-only follow-up to `REVIEW.md` and `INITIAL_COUNTEREXAMPLES.json`; their original CHANGE_REQUIRED finding remains preserved.

**Result: PASS_CODE_WIRING_ONLY** at author commit `65b64849458a0545dff97f4113c28f5ccdbcbe02` in `/mnt/project/VLA-Drive-st-temporal-profile-prepare-20261018`.

Exact execution dependencies:

| File under `reports/st_world_policy/20261018/full_history_train_resume_preparation_v1/` | SHA256 |
|---|---|
| `resume_records.py` | `22d059a9753ae2a2dadbea6db2ab97a67a8993804372a53d2341929a4079a360` |
| `run_resume_once.py`, unchanged | `04099e59b7c137ab1520b647a0529c867904107fa0fd8942de2ac0f49f948858` |
| `ADMISSION_PROPOSAL.json`, still a proposal | `342ef855554c356c05b615b1727b0c243f163f6da8111d3d975db40fd1fe71be` |

The only execution-code change from the rejected revision is in `resume_records.py`. `build_or_reuse` now passes the actual raw frames, already loaded by the original builder, into `get`. For reused records, `get` calls original `causal_frames`, validates current/history tokens, timestamps and offsets, and calls original `resolve_camera_path` for every view using the original ordered sensor roots. Both absent and present paths must match the committed record, including its valid flag. It verifies sensor exposure time, then retains the original one-SHA-per-unique-image and same-invocation full-version checks.

The previous missing-to-present and new-primary-over-fallback cases now raise `ValueError`; no stale reuse or fallback regeneration occurs. Unchanged missing views remain reusable, removed views are rejected, and frame-selection changes are rejected. Original geometry computation and image headers are not recomputed for unchanged old records. Uncovered scenes still call the original builder. No protected native source, ledger, monitor, model, training definition, population, typed artifact allocation or failure receipt is modified by this fix.

Independent execution of the six existing reviewer tests **without changing them** plus the18 author tests: **24 passed in4.28s**. This includes both originally failing negative tests, duplicate/log rejection, original packed-writer byte equality and immutable conflict refusal, role/self-admission restrictions, first-use image SHA, subsequent inode/version rejection, parser scope, new-scene behavior and failure accounting.

```bash
cd /mnt/project/VLA-Drive-st-spatial20k-correctness-20261018
python -B -m pytest -q -p no:cacheprovider \
  reports/st_world_policy/20261018/history_resume_strict_correctness_v1/test_independent_resume.py \
  /mnt/project/VLA-Drive-st-temporal-profile-prepare-20261018/reports/st_world_policy/20261018/full_history_train_resume_preparation_v1/test_resume.py
```

The original metadata census/expected78,657 reused and22,935 new scenes were not rescanned or executed by this review. No real image, checkpoint, large shard/index or model was read. Actual completion, IO consumption and source availability remain execution results to obtain. The independent resource owner must issue the fresh actual admission; the proposal is not such an approval. Only Lead dispatches the bounded continuation. Both initial failed attempts and this review's original rejection evidence remain retained.
