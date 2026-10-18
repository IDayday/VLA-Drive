# Fixed endpoint preparation and execution boundary

Preparation is complete; execution is **WAITING_CHECKPOINT_AND_FRESH_LEAD_ADMISSION**.
The formal training runtime remains751fcfa2b41549ae16f7b3204aef8dd18b912016.
Only Lead may dispatch after independent review. There is no automatic watcher,
checkpoint selection, phase chaining or retry. Do not edit the running runtime.

These preparation commands run without GPU, model, image, cache or score reads:

```bash
cd /mnt/project/VLA-Drive-st-endpoint-supervisor-20261018
/root/miniconda3/envs/ddp/bin/python resources/st_endpoint_supervisor_v1.py --help
/root/miniconda3/envs/ddp/bin/python resources/st_endpoint_supervisor_v1.py --status
/root/miniconda3/envs/ddp/bin/python reports/st_world_policy/20261018/rollout_scale_endpoint_supervisor_v1/run_cpu_checks.py
```

Once the real4096 native receipt is complete, Lead creates the actual admission
described in ADMISSION_CONTRACT.md. This report deliberately supplies no fake
path/hash. The following exact shell commands require Lead to set
`ENDPOINT_ADMISSION` and `ENDPOINT_ADMISSION_SHA256` to the actual reviewed file
and its SHA256. They are not current executable authorization.

```bash
: "${ENDPOINT_ADMISSION:?actual independently reviewed admission required}"
: "${ENDPOINT_ADMISSION_SHA256:?actual admission SHA256 required}"
/root/miniconda3/envs/ddp/bin/python resources/st_endpoint_supervisor_v1.py \
  --admission "$ENDPOINT_ADMISSION" --admission-sha256 "$ENDPOINT_ADMISSION_SHA256" --phase export
```

The preceding invocation checks real lineage/receipt/DEV/control metadata but
does not launch or claim payload loading. Runtime resource checks occur at
execute time. Lead then executes one explicit phase at a time:

```bash
/root/miniconda3/envs/ddp/bin/python resources/st_endpoint_supervisor_v1.py \
  --admission "$ENDPOINT_ADMISSION" --admission-sha256 "$ENDPOINT_ADMISSION_SHA256" --phase export --execute
# After all424×4 exports COMPLETE and owned sessions exited, refresh resource
# observations/admission as needed, keeping the reviewed stable binding intact.
/root/miniconda3/envs/ddp/bin/python resources/st_endpoint_supervisor_v1.py \
  --admission "$ENDPOINT_ADMISSION" --admission-sha256 "$ENDPOINT_ADMISSION_SHA256" --phase score --execute
# After valid official score COMPLETE1696/1696, refresh resources if needed.
/root/miniconda3/envs/ddp/bin/python resources/st_endpoint_supervisor_v1.py \
  --admission "$ENDPOINT_ADMISSION" --admission-sha256 "$ENDPOINT_ADMISSION_SHA256" --phase analysis --execute
```

Partial results remain in their fixed output namespace. Lead must inspect the
failure and unresolved owned sessions; there is no auto retry. If a second
same-identity invocation is explicitly approved, use the same phase command
with `--execute --resume`. This consumes the final per-unit attempt. Existing
partial intent directories and failed launch admissions remain charged. A
changed stable binding, unknown previous exporter state, exhausted attempts,
wrong native/control, revoked host, insufficient lease/RAM/quota/free space,
unknown GPU occupant or absent independent reviews is refused. Do not delete
receipts to bypass an exhausted or unresolved budget.

Outputs are the predeclared export/official_score/control under
`/mnt/project/st-world-policy-20261016/formal/evaluation/jobs/C1_rollout_std_step004096_dev1696_seed42_v1/`;
analysis is
`/mnt/project/st-world-policy-20261016/formal/evaluation/analyses/C1_rollout_std_minus_C1_COMMON_ENDPOINT_dev1696_v1/ANALYSIS.json`.
Exit0 is reported only after exact completion receipt validation; partial or
blocked exits2, unavailable endpoint exits3. No ACTIVE/navtest update exists.
