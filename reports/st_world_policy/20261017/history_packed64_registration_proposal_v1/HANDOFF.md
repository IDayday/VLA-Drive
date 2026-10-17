# Exact history64 packed cost-pilot proposal

This directory contains static proposals, **not admission, a claim, a migration, or a launch**. No source code, production ledger, output directory, images, raw log or tensor was changed/read by the proposal generator. Active TRAIN metadata construction must finish before Lead performs any production mutation.

Frozen runtime: `/mnt/project/VLA-Drive-st-history-packed-runtime-v1-20261017`, commit `b847f5a99dd44ee5256362d380d9d96efb47d35b`. The original `full_causal_history_preparation_v1/run_pilot_once.py` is reused unchanged; no new controller is needed. All builder callback dependencies and five controller/helper files have actual path/SHA pins in `PROPOSAL.json`.

The exact old policy is `59371867175a65dd032cc27c88a050227d8ac4ba34a681c2e31b250c58a0978d`. `PROPOSED_POLICY.json` was produced by the existing `prepare_controls.extend_policy`: all 10 existing claims and every non-claims field are equal, in order. Only the 31 MiB history artifact declaration is appended. The existing 768 GiB campaign cap, 8 GiB metadata package, frozen-W declarations, legacy reservations and 1 TiB floor are unchanged. No ledger was read; its fresh SHA remains explicitly pending.

Budget and population:

- Original immutable selection: `smoke_train64_v1/SELECTED_SCENES.json`, SHA `dee94e75957fd8e9734f1669b47552b5cf6c39f719aa9d3298bc87f94aac1c01`: **64 scenes in 64 logs**, unchanged. This implies up to 64 log shards, not one 128-scene shard.
- CPU 6/7, one writer, 900 seconds work plus 40 seconds owned cleanup, 16 GiB aggregate read allowance, 32 MiB/s builder IO pacing, 4 GiB aggregate RSS, no GPU, no automatic retry.
- Explicit command option `--record-layout packed_log_v1`; record max 1 MiB, shard max 4 MiB/128 records, at most 64 shards. These byte/count limits are implementation bounds, not predictions of actual output.
- Only target time 0.0. Historical frames remain current-input metadata; no future target generation or teacher computation.
- Artifact **31 MiB** includes all final records, manifests, current requests, lock, controls and retained temporary/failure files. Independent supervisor control directory **1 MiB** includes its lock and receipts. Combined **32 MiB**.
- Existing separate 4 MiB full-future monitor addendum remains pinned at SHA `2946083d7ed65aa931bcdf12bee333b340d9e0c9ec742509314ebfed9de61ddf`, and is not borrowed. Prior planned total 5,620 MiB becomes **5,652 MiB / 8,192 MiB**, including the already planned future indexes. Planned totals are not a claim that pending indexes have been built/admitted.

`CONTROL_RESERVATION_PROPOSAL.json` requires Lead to adopt the independent 1 MiB control allocation before creating that directory. It is deliberately not disguised as an existing ledger claim. No new legacy reservation is invented. The new metadata claim and this admin allocation together consume the original pilot's 32 MiB budget.

## Lead execution sequence after writer exit

1. Recheck active and queued metadata writers and current policy/ledger under the existing procedure. If the current policy has advanced, preserve its additional claims and regenerate this **additive** proposal rather than restoring this snapshot. Do not acquire an indefinitely blocking ledger lock while TRAIN remains active.
2. Verify all proposal/source hashes, original selection, fresh shared free space, and empty prospective artifact/control roots. They did not exist during preparation. Existing unexpected files must be preserved and reviewed; do not clear them to retry.
3. Adopt the separate 1 MiB control allocation using existing Lead administrative receipt practice. Preserve the existing 128 MiB admin scope and 4 MiB monitor addendum. The control directory accepts only its five documented lock/receipt/temp filenames; do not put admission/config files there.
4. Use the existing same-ledger migration interface with fresh actual ledger/quiescence/inventory evidence. `NO_PILOT_ADOPTIONS.json` states the requested transition adopts no prior pilot roots; Lead must verify that premise. No new migration implementation is provided.

```bash
cd /mnt/project/VLA-Drive-st-history-packed-runtime-v1-20261017
CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.metadata_storage migrate \
  --old-policy /mnt/project/st-world-policy-20261016/audits/full_future_metadata_preparation_v1/prep_phase_v1/PROPOSED_POLICY.json \
  --old-policy-sha256 59371867175a65dd032cc27c88a050227d8ac4ba34a681c2e31b250c58a0978d \
  --new-policy /mnt/project/VLA-Drive-st-history-scale-independent-v2-20261017/reports/st_world_policy/20261017/history_packed64_registration_proposal_v1/PROPOSED_POLICY.json \
  --new-policy-sha256 a6984d65bfcb47e32153ac3b8b395dcbf8a694ef3844ea644f82e0c58ffbd8fd \
  --expected-old-ledger-sha256 "$LEAD_FRESH_LEDGER_SHA256" \
  --quiescence-receipt "$LEAD_QUIESCENCE_PATH" --quiescence-sha256 "$LEAD_QUIESCENCE_SHA256" \
  --pilot-inventory /mnt/project/VLA-Drive-st-history-scale-independent-v2-20261017/reports/st_world_policy/20261017/history_packed64_registration_proposal_v1/NO_PILOT_ADOPTIONS.json \
  --pilot-inventory-sha256 1c3b4503a88d29dfdffe9e75415e7b8161e59f9d25c6891b2db913f3660c4d23 \
  --checkpoint-receipt /mnt/project/st-world-policy-20261016/smoke/official_history_T1_native4_v1/controls/CHECKPOINT_COMMITMENTS_2303G_V1.json \
  --checkpoint-sha256 13f3f4b282db2aff24528bc5434db51a9351cf43cf02dcdea1fbef38d5434cb3
```

The shell variables intentionally require real fresh evidence, not guessed hashes. Do not run this command directly from the unsigned proposal. New registration SHA is `01de79cac8f49804a662d8f1454ac9b85eb6358c8ea572e1ef13e9a3bd5d15a0`; if any JSON path/bytes change, refresh dependent registration, policy and admission hashes.

5. Lead and the relevant independent reviewer issue a separate actual admission from `ADMISSION_NOT_APPROVED.json`, filling real hostname/reviewer/times and status only after prerequisites. The template is rejected by the controller as written. Keep `latest_start + 940 <= valid_until <= CP.valid_until`. The pinned CP preserves 2,303 GiB and is valid to Unix **1792250452.226822**. If that window remains sufficient, no CP or code change is needed. If it expires, do not edit the historical CP: the wrapper hardcodes its pin, so a narrowly reviewed new CP pin/runtime version is needed.
6. Create the explicitly reserved plain control directory, then run exactly once:

```bash
cd /mnt/project/VLA-Drive-st-history-packed-runtime-v1-20261017
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /root/miniconda3/envs/ddp/bin/python \
  reports/st_world_policy/20261017/full_causal_history_preparation_v1/run_pilot_once.py \
  --admission "$LEAD_ACTUAL_HISTORY64_ADMISSION_PATH" \
  --admission-sha256 "$LEAD_ACTUAL_HISTORY64_ADMISSION_SHA256"
```

The wrapper sets parent and child CPU6/7, uses the existing nonblocking ledger adaptation and whole-invocation alarm, supervises its own session, and retains a terminal receipt even on ledger contention. Only `COMPLETE` plus actual final accounting and no live owned descendants can support a cost result. Preserve FAILED/UNKNOWN and partials; no automatic retry.

The standalone registered builder argv is in the JSON for auditing, **not an alternate unmonitored launch**. `--resume` is intentionally absent: this is the original single-attempt 64-scene cost pilot, not an unbounded full build. This proposal supplies no formal training, full metadata build, GPU, teacher or scientific permission.

Preparation reused existing pure helper `read/document/callback_pins/extend_policy/write`. Pinned helper reads total 399,189 bytes; generated JSON totals about 67 KiB. Source inspection was read-only; no heavy Python/model CLI imports or regression reruns were needed. These are proposal preparation sizes, not measured pilot costs.
