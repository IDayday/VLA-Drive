# Native preparation and training handoff

Run from the final clean integrated runtime. `COMMANDS.json` contains exact argv arrays for source migration, T0--T4 preparation and the two finite T4 profile modes. The four main formal recipe drafts and matched T0 draft are metadata templates; the preparation CLI produces the real config with verified provider/PCA pins. Drafts are not passed directly to the trainer.

```bash
python -m tools.st_world_policy.temporal_formal --help
python -m tools.st_world_policy.qualify_indexed_provider prepare-profile-migration \
  --parent-descriptor /mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_native2_v1/PROVIDER.json \
  --parent-runtime /mnt/project/VLA-Drive-st-indexed-qualified-runtime-v1-20261017 \
  --output /mnt/project/st-world-policy-20261016/formal/temporal/preparation_v1/SOURCE_MIGRATION.json
```

Then execute the selected cell's `prepare_formal` argv from `COMMANDS.json` under the existing bounded metadata supervisor. Explicit required flags are `--scope FORMAL --world-size 4 --micro-batch 1`; these are provisional, not a lease. T1/T3/T4 require `full_causal_population_v1/train/COMPLETE.json`. T2 and optional T0 reject `--history-complete`. Preparation validates the entire actual current/future population and history attachment; it does not open teacher weights or cache new tensors. Original PCA statistics are loaded on CPU by the existing scope verifier. Failed staging is retained; a successful immutable staging version is published atomically. Use the canonical stage paths from `PREPARATION_COMPLETE.json` when registering.

The original source migration preflight in this report is author evidence only. Do not reuse it after relocation: its exact new-file paths must match the chosen runtime. New inputs always receive a new output directory; no old descriptor, prepared directory or failed attempt is overwritten.

Registration uses the existing `st_world_run_v1` schema and unchanged `check_formal_approval`:

- `config`, `data_manifest`, `teacher_manifest` and `target_contract` come from the preparation result, with exact actual pins.
- `base_binding` remains the actual common Base SHA `84544c2242c0c7ecdcbe4450073b23a3c556cad4d822860d5b8652ecee9f1e38`.
- Source SHA/fingerprint identify the final clean runtime; implementers include this agent.
- `environment` and `resource_lease` are fresh, real Lead-issued artifacts. No placeholder is accepted by the native gate.
- Formal registrations require independent evidence/correctness/efficiency reviews and actual RUN_APPROVAL. Diagnostic registrations have `weights_usable_for_formal_init=false` and no formal approval.

Native command after registration and resource admission:

```bash
python -m tools.st_world_policy.check_formal_approval --registration ACTUAL_REGISTRATION --approval ACTUAL_APPROVAL --repo FINAL_RUNTIME
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
  -m tools.st_world_policy.train_world --registration ACTUAL_REGISTRATION --approval ACTUAL_APPROVAL
```

Native continuation adds `--resume` (and `--acknowledge-stop` only for an explicitly acknowledged prior stop). It resumes that run's model/master/Adam/scheduler/RNG/consumed cursor. Never substitute a profile checkpoint or restart the budget. Diagnostic timing forbids partial-window resume through the existing native guard.

For capacity first, use `prepare_profile["4"]` (PROFILE/none/global32), then a separately registered native invocation without formal approval. The timing follow-up `prepare_profile["240"]` uses `profile_mode=timing`,240 updates, two20+100 windows and a distinct output. Both retain the same model/loss/PCA/precision recipe. No profile is launched by preparation. T4 includes the temporal branches; its full-history dependency remains real. These are a finite qualification sequence, not a topology matrix.

Reuse the mature monitor from `reports/st_world_policy/20261017/full_population_profile_launch_v1/controller.py` (original SHA `bd802d47f187e494c2dde72ab8c04c670cb529d2b32b187ca06dfad3f3cc6bd1`) through the established T3 thin-binding pattern. Its current T3 wrapper is pinned to world1/4 updates and must receive a reviewed new binding for this topology, not be invoked unchanged. Similarly `resources/launch_reviewed_spatial_v1.py` hardcodes an old runtime and the spatial registration root; do not use it unmodified for temporal runs. This change intentionally does not copy or replace either monitor.

Resource handoff: old SP2 world4/micro4 supports current-only compute choice; old T3 world1/micro1/global2,4-update peak reserved63,149,441,024B supports only that observed case. It does not qualify world4/micro1 or micro4. Bound actual RSS/read/GPU/checkpoint/output limits in the new registration/controller; preserve all existing checkpoint commitments. A provisional240GiB per formal5k checkpoint allowance is not yet allocated. Full history manifest parsing and repeated provider/PCA validation must be counted in metadata preparation; the old exact32 preparation cost is not a full-population estimate. No new TB-scale teacher cache is required.
