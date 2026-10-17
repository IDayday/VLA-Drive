# Matched probe execution handoff — not yet launch-ready

The original runtime and its CLI already support the required experiments. See
`ORIGINAL_CLI_HANDOFF.json` for the exact five-stage argv for each of SP1/SP2/SP3.
They were derived from the actual parser at frozen source
`581399eb309677c20fe7464791b4f18685e947a6`; no runtime imports or `--help` process
were launched because this assignment is restricted to source/metadata reads.

Existing dependencies are real: the registered 5000-step checkpoints, original
runtime, fixed 64/16 populations, current raw labels and SP0 fit. Planned new
control paths are explicitly absent. The two `REQUIRED_FRESH_ADMISSION_*` values
are unresolved resources, not invented receipt paths or hashes.

1. Create a small isolated fixed-arm control revision. Bind only SP1/SP2/SP3 to
   the exact registration/export/native receipt pins in `ASSET_INVENTORY.json`.
   Keep old SP0 controls and all failure/reconciliation evidence immutable.
2. Mechanically reuse the original `prepare_bindings` source with those exact
   fixed-arm constants, 144/144/288 W lengths, original protocol and separate
   `audits/spatial_probe_actual_SP{1,2,3}_5000_v1` namespaces. Reuse the existing
   current-observation request/image hashes; generate no new labels. Emit
   `PREPARATION.json`, train/dev binding/expected-signature and fit registration.
3. Independently review the minimal controller revision: rediscover active
   pidfd-owned processes after `leader_done()` confirms exit, then retain true
   descendant failure. Keep original observer/counters and storage bridge
   verification. Fixed-arm enumeration replaces literal SP0 pins; no arbitrary
   command execution, generic failure acceptance or algorithm rewrite.
4. Resource reviewer allocates the three added W packages under the shared
   1 GiB cap, separately accounts 192 MiB maximum fit outputs and bounded
   controls, checks native checkpoint commitments and issues each exact stage
   admission with CPU/GPU/read/RSS/wall limits. No allocation is made by this
   handoff. Preserve the one-loader lock and current real work.
5. For each arm, run the original observer under the newly reviewed thin
   supervisor in this order: `dev_plan`, `dev_extract`, `train_plan`,
   `train_extract`, `fit`. Actual plan identity must match expected signature;
   fit requires both complete, fully validated caches. Do not run three native
   checkpoint loads concurrently without explicit I/O/RAM admission.
6. Use the original `spatial_probe fit` at 300 CPU updates with the same labels,
   seed and recipe. Compare its actual train/dev metrics with SP0 after all
   matched probes complete. No checkpoint/arm selection is made here.

The downstream **original CLI payloads** are these real interfaces, repeated
for the three fixed arms using the exact argv file; they are not a request to
bypass the supervised entry point:

```text
python -m tools.st_world_policy.spatial_probe plan-cache --binding <new reviewed split binding> --output <arm split cache root>
python -m tools.st_world_policy.spatial_probe extract --binding <same split binding> --output <same cache root> --storage-policy <new reviewed policy> --checkpoint-commitments <fresh exact commitment receipt> --checkpoint-commitments-sha256 <its actual SHA256>
python -m tools.st_world_policy.spatial_probe fit --registration <new reviewed fit registration> --output <arm job>/fit
```

`--resume` exists only on `fit`, and only when an actual checkpoint state exists;
it is not used to create these three runs. No automatic retry is authorized.
Failure retains its result and requires a specific next decision. In particular,
SP0's exact legacy DEV/TRAIN artifact reconciliation cannot authorize SP1–3.

Current affected source pins are in `CONTROL_SOURCE_PINS.json`:
`prepare_bindings.py`, `supervise_once.py`, `observed_cli.py`,
`w_storage_bridge_v1/storage_bridge.py`; the existing `4b2f...` core is the
reference for the two-line rediscovery behavior, not a replacement that drops
GPU/logical-load accounting. Original `spatial_probe.py`, native checkpoint
reader, readout, labels and fixed protocol stay unchanged.

Only this metadata inventory command was executed in the current handoff:

```bash
cd /mnt/project/VLA-Drive-st-matched-spatial-probe-handoff-20261017
taskset -c 10,11 /root/miniconda3/envs/ddp/bin/python \
  reports/st_world_policy/20261017/matched_spatial_probe_handoff_v1/capture_metadata.py
```

It does not register or launch anything. `CAPTURE_COST.json` records final
metadata checks and resource evidence; earlier resource-accounting assertion
failure is retained separately. This report is not an independent launch PASS.
