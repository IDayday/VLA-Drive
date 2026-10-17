# Full causal-history readiness — read-only inventory

Inspected source: `0d36e570` (Lead's actual T1 phase2 launch record). Current research contracts, AGENTS and the preceding temporal inventory were read. This inventory never polls or modifies the live T1 run, frozen e85/59c256 trees, future builder, storage ledger, models, image bodies, raw logs or target tensors. It is not a dispatch approval.

**Full train/dev causal-history metadata is not present in the inspected campaign.** A campaign-wide filename-only search found exactly three `HISTORY_MANIFEST.json` files: train64, dev16, and a real registered **train256** preparation. The latter is useful prior evidence beyond smoke64, but does not cover full training. Missing population metadata is **WAITING_INPUTS**, not a model failure. Full future-label metadata cannot satisfy this dependency.

## Actual assets

Base path below: `/mnt/project/st-world-policy-20261016/audits/history_data/`. Small COMPLETE/REGISTRATION controls and fresh stats are retained in `ASSET_CONTROLS.json`; hashes and historical configuration/source provenance in `SMALL_HISTORY_CONTROLS.json`.

| Asset | Verified small control / fresh stat | Meaning |
|---|---|---|
| `smoke_train64_v1` | COMPLETE 64 scenes / 64 logs; HISTORY manifest SHA `14d1b59a…`; 64 record files, 936,677 logical bytes | Real causal-history qualification population, not full train |
| `smoke_dev16_v1` | COMPLETE 16 scenes / 16 logs; HISTORY SHA `b06b93ad…`; 16 record files, 233,782 bytes | One scene per dev log, not full dev |
| `train256_current_cost_v1/preparation` | COMPLETE 256 scenes / 256 logs; HISTORY SHA `15402291…`; 256 record files, 3,746,611 bytes | Existing larger causal-history preparation with **current-only target times `[0]`**; original registration at `train256_current_cost_v1/PREPARATION_BUDGET.json`, source `73d1be9b…`, writer SHA `d5bf7b5d…` |
| `full_current_population_v1/train` | COMPLETE 101,592 scenes / 1,176 logs; selected SHA `58dffa07…`, 30,071,766 bytes stat-only | Exact full TRAIN population exists. Registration explicitly `raw_history_read=false`, `future_read=false`, times `[0]`; **no HISTORY manifest** |
| `full_current_population_v1/dev` | COMPLETE 1,696 scenes / 16 logs; selected SHA `4ce40893…`, 502,548 bytes stat-only | Exact full DEV population exists; same current-only limitation; **no HISTORY manifest** |

The full selection contents and 170 MB train TEACHER_REQUESTS were not deserialized. Full population counts and pinned hashes come from actual small controls, not a new population revalidation. The three small history manifest hashes were freshly recomputed; individual history record bodies were not read. Searching was limited to the actual campaign root, not every filesystem.

## Implementation → dependency

| Requirement | Actual code / evidence | Minimal remaining action |
|---|---|---|
| True causal −1.0/−0.5 s history | `st_world_history.py::causal_frames/build_history_record`: same log, ≤50 ms tolerance, strictly before t0, ambiguous/duplicate timestamps rejected; missing frame/view stays masked. Camera intrinsics, distortion, crop/resize and camera-to-current-planar-ego transforms are recorded | Reuse unchanged selector/record function for every original train/dev observation; do not reinterpret future labels as history |
| Full population and order | `prepare_history.run` supports `--mode registered --scene-list`; original source/index/partition role and log separation checked | Explicitly pass existing full SELECTED_SCENES to preserve its order. Do not let default `stable_selection` create a different full ordering. Independently bind the existing selected hash and verify result coverage/order/source pins |
| Build history without new future work | Existing `prepare_history --target-times 0` already demonstrated by real train256 | New versioned history output per role; reuse original current/future teacher/PCA elsewhere. No teacher tensor or VLM cache creation needed |
| Resume | Existing `--resume`, immutable config/record/manifest writes and PREPARE flock | Existing implementation **revisits all logs/scenes and rehashes images**, not per-log skip-resume. Budget its actual repeated IO; do not claim shard-level cheap recovery |
| Runtime reader | `HistoryReader` requires complete manifest identity/hash and checks each record; `STWorldDataset` rejects missing history scenes and wrong split | Full coverage receipt plus full-scale startup/RSS qualification. Reader currently loads the complete manifest/index into RAM; no need to invent a new model interface |
| Bound full metadata preparation | `prepare_history` registered budget checks phase/split/max-scenes; it does **not** enforce registration workers/max_seconds/read/RSS or cumulative output space | Smallest execution gap: a reviewed bounded wrapper/admission for this existing builder, with explicit output and read limits. Existing future metadata dispatcher is not a history permission. Do not launch the bare CLI at full scale or mutate the active writer's registration |
| T1 | Current targets/PCA plus causal-history population; official64 qualification now separately launched by Lead | Full history preparation and matched full-population runtime/admission remain separate from live64 engineering qualification |
| T3/T4 | Full history **and** full future labels/provider; `teacher_provider.validate_profile_recipe` currently requires `history_root=None` | Narrow composition qualification and explicit admission, after actual dependencies exist; do not wholesale remove the guard or call T2 approval T3/T4 approval |

## Available paths and preparation command

The original source roots and partition below were found from current-population registrations and checked by stat. Raw/sensor directories exist, but their contents were not opened. This does not prove every required image/log is readable. Original TRAIN and DEV have separate source identities, kept in the evidence JSON.

`python -m tools.st_world_policy.prepare_history --help` was actually executed on CPU2–3 with CUDA hidden, 20-second cap and no data arguments. The documented flags below are real. They are **preparation templates requiring a real new registration and admitted output**, not an approved launch. No such full-history registration/output was created here.

```bash
cd /mnt/project/VLA-Drive-st-full-history-readiness-20261017
ROLE=train                         # repeat independently with dev
COUNT=101592                       # dev: 1696, from actual COMPLETE controls
python -m tools.st_world_policy.prepare_history \
  --student-root "/mnt/project/ddp-foresight-artifacts/20260928/student_${ROLE}_v1" \
  --raw-log-root /mnt/navsim/trainval_navsim_logs/trainval \
  --sensor-root /mnt/navsim/trainval_all/trainval_sensor_blobs/trainval \
  --fallback-sensor-root /mnt/navsim/trainval_sensor_blobs/trainval \
  --partition /mnt/project/VLA-Drive-st-history-population-agent-20261016/reports/ddpolicy_vehicle_from_scratch/NAVTRAIN_PARTITION.json \
  --scene-list "/mnt/project/st-world-policy-20261016/audits/history_data/full_current_population_v1/${ROLE}/SELECTED_SCENES.json" \
  --split "$ROLE" --mode registered --max-scenes "$COUNT" \
  --seed 20261017 --tolerance 0.05 --target-times 0 \
  --registration "$ACTUAL_NEW_HISTORY_REGISTRATION" \
  --output "$ACTUAL_ADMITTED_NEW_HISTORY_ROOT"
# Same exact admitted identity + --resume is supported; not a free retry budget.
```

## Resource interpretation and next useful action

Metadata output is small relative to tensor caching, but its input IO is substantial: the builder hashes raw logs, hashes/reads headers of up to six past images plus three current references per scene, and hashes original current/ego records without decoding GT values. `image_reference` performs file SHA and `Image.open` header access; no zero-image-IO claim is valid. It runs one Python worker and loads one raw log at a time, but also holds population/manifest dictionaries. CLI itself has no read-throttling or RSS ceiling.

Stat-only train256 history records average 14,635.2 bytes/scene; linear scaling of **history records alone** is about 1.385 GiB for 101,592 and 23.7 MiB for 1,696. This is an unverified estimate, excluding full manifests, selection/config copies, current-only teacher requests, temporary atomic writes, filesystem allocation and retained failures. Do not reserve this final-size estimate as the complete peak budget. Full image/hash IO, cold-read duration, RSS and missing-view distribution remain unmeasured; no full scan was authorized here.

The next bounded, parallelizable preparation is to pin both existing populations, define separate immutable history outputs/receipts and register a small cost/coverage check using the existing train256/dev16 evidence. Then independently admit one bounded full-history builder with shared-IO limits and complete checkpoint/free-space reservations. This does not wait on future *semantics*, but actual IO/storage writer coordination must preserve the active full-future job. No new GPU or teacher model is required for preparation. No mass build, retry or ledger operation is authorized by this inventory.

No new tests were needed for this read-only inventory beyond actual `--help`; the earlier 51 PASS/5 skip history tests remain prior evidence, not rerun counts. The live T1 experiment is not assessed or polled here. Space selection, eventual 20k endpoints and T3/T4 real execution remain their existing scientific dependencies, not failures caused by this missing metadata.
