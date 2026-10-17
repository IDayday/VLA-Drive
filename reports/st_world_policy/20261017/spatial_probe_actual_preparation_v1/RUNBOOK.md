# SP0 native5000 learned-W preparation — candidate, no dispatch

This package implements only the five already registered stages: fixed dev16
plan, dev16 extraction, fixed train64 plan, train64 extraction, and the original
CPU probe fit with 300 exposure cycles. It does not run full-dev PDMS, select a
checkpoint, create a second training path, or authorize any job. Lead alone
dispatches after independent resource/controller review. No real plan, native
load, image decode, W extraction or fit was executed to prepare this package.

Training remains `fdf25f99b702754787904215bf1c2a2b7515ea75`, exact tag
`step_005000_1792217229597178910`, receipt SHA
`1583c98960120ad2cde5859919507386b27259c4b3524a82b40de6ce0f1299fe`.
The 5k checkpoint is the declared screening boundary on the unchanged 20k
training clock. Receipt inventory is 37,366,035,746 bytes; no payload was rehashed.
Checkpoint deployment identity is
`c87c59804b6237205e4b4a7a7e355b569c7cef5eaa499023f699111ec36d0beb`.

`prepare_bindings.py` constructs expected bindings from the existing SP0 export
IDENTITY control (1,257,323 bytes), exact native COMPLETE, the pinned original
protocol, fixed populations, current source identities, existing current-only
teacher-request image hashes, and 80 small current JSON records. It compares
token/log/view order and protected source ASTs. It reads neither the full source
index nor any image, label array, model or native tensor. Existing image hashes
are expectations; real `cache_plan` still verifies their bytes and all native
shards. No extra `checkpoint_identity()` or `STObservationDataset()` call is
needed merely to construct the bindings.

The generated `controls_cf580501_metadata_only/` snapshot is real metadata-only
evidence for the initially supplied clean runtime
`/mnt/project/VLA-Drive-st-spatial-probe-runtime-v1-20261017` at
`cf5805016b5e23bdab9541c77b562ed5fda83dc3`. It is not an executable approval. That
runtime's generic storage-control bound is **16 MiB**, while SP0 run identity is
19,105,073 bytes. Lead reported the dedicated 32 MiB run-identity fix integrated
at `8fdc8be5` and its independent review at `6e12fc2a`. After integrating this
candidate, Lead freezes the final v2 runtime and reruns only the metadata builder
into a NEW controls directory. Do not modify old bindings or substitute the
training/evaluation SHA for the final extraction SHA.

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  /root/miniconda3/envs/ddp/bin/python \
  "${ST_PROBE_HELPERS:?}/prepare_bindings.py" \
  --runtime "${ST_PROBE_RUNTIME_V2:?clean final reviewed runtime}" \
  --source-sha "${ST_PROBE_SOURCE_V2:?exact final SHA}" \
  --output "${ST_PROBE_NEW_CONTROLS:?new immutable controls path}"
```

The path templates below are fixed, not allocated by this preparation:

- Control/attempt root: `/mnt/project/st-world-policy-20261016/audits/spatial_probe_actual_SP0_5000_v1`.
- W roots: `/mnt/project/st-world-policy-20261016/caches/learned_w_probe_SP0_5000_v1/{dev,train}`.
- Fit root: the control root's `fit/`; only the original small-head two-slot checkpoint implementation writes there.
- Campaign lock: `/mnt/project/st-world-policy-20261016/resources/SPATIAL_PROBE_ONE_LOADER.lock`.
- Same existing ledger: `/mnt/project/st-world-policy-20261016/resources/cache_storage_ledger_v1/CLAIMS.json`.
- Extract also holds the established local `/tmp/foresight_gpu_<index>.lock`; it requires a freshly idle exact UUID and rejects any compute owner. It never signals an existing job.

Lead supplies a separate stage-specific admission JSON, pinned by SHA, after
independent review. `ADMISSION_DRAFT.json` is deliberately PENDING and rejected.
It is not an approval template that the implementer may self-sign. Admission
binds the new PREPARATION pin, final runtime, all three helper hashes, exact host,
two available CPUs, lease expiry, and the GPU for extract only. The supervisor
and its child share those two CPUs; BLAS/PyTorch uses two threads. CPU plan/fit
use an empty CUDA mask. Stage lease and checkpoint-space receipt must cover
the full outer interval plus 30 seconds of cleanup, not only the 900-second
cooperative extraction interval.

The sequence is strictly `dev_plan → dev_extract → train_plan → train_extract →
fit`. One global shared-flock lock serializes this finite pilot across hosts.
Each stage creates its durable `ADMITTED.json` before Popen and can be attempted
once. A failed start, partial result, limit breach or interrupted child remains
spent. No automatic retry/resume, marker deletion, best-checkpoint selection or
larger budget exists. The broader protocol's separately reviewed extra lifecycle
or fit `--resume` is not implemented by this first-attempt supervisor.

```bash
# Execute ONLY after Lead obtains the independent, exact stage admission.
env PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python \
  "${ST_PROBE_HELPERS:?}/supervise_once.py" \
  --admission "${ST_PROBE_STAGE_ADMISSION:?approved exact stage JSON}" \
  --admission-sha256 "${ST_PROBE_STAGE_ADMISSION_SHA:?exact SHA256}"
```

The thin child calls the existing `spatial_probe.cache_plan`, `extract`, or `fit`
once. Plan also saves the actual returned immutable `SIGNATURE.json`, which the
normal CLI does not persist; its identity must equal the metadata expectation.
This is the signature for typed-W admission. Do not run the raw CLI in addition
to the supervisor. Extraction retains its original internal cache-plan and
native revalidation calls; none are removed or cached by this wrapper. Per split
there is one explicit plan and one extraction process/model load. Plan is a real
~37 GB native checksum job, not a status check or cheap metadata operation.

Before each extraction, Lead must migrate the typed-W package using the existing
reviewed mechanism and obtain an immutable MIGRATED receipt. The planned v7
baseline retains both 32 MiB metadata index declarations (`future32_request_metadata_index_v1`
and `original_pca_fit64_metadata_index_v1`) in addition to all earlier claims.
Use Lead's final stable policy/ledger transition; this package never reads or
modifies the live ledger during preparation. The same 768 GiB campaign cap and
1 TiB free-space floor remain, with checkpoint obligations additional. The
supervisor rejects another ledger, an expanded cap, an uncompleted migration,
changed policy SHA, or a short-lived checkpoint-space pin. The original extractor
passes that pin to every reserved write through the independently reviewed
`5114055f` guard. StorageBudget still validates actual package claims and bytes.

| Stage | Outer wall | RSS limit | GPU allocator peak | Read allowance |
|---|---:|---:|---:|---:|
| Each plan | 600 s | 64 GiB | no GPU | shared with same split extraction |
| Each extraction | 960 s | 64 GiB | 24 GiB | 256 GiB per split lifecycle |
| CPU fit | 1260 s | 2 GiB | no GPU | 2 GiB observed/known read allowance |

The outer process samples existing `metadata_storage.process_tree_rss` and wall
time once per second. The child records PyTorch's historical maximum allocated
and reserved GPU bytes every 0.5 seconds; a peak over 24 GiB stops the stage.
These are observed limits with a reaction interval, not cgroup hard ceilings or
guarantees against an uninterruptible kernel call. Only the newly created child
session/process group receives TERM, then KILL after 30 seconds if needed; no
foreign PID or existing task is accepted as a signal target. A final over-limit
counter rejects a result even if the child exits before the next outer sample.

Accounting reports successful full-file hash calls and logical bytes, admitted
hash bytes, nominal `torch.load` input sizes/calls, whole native-loader wall,
and `/proc/self/io` deltas from before runtime imports. Wrappers call the original
function exactly once and do not change tensors, seeds, initialization or data.
Hash and torch-load times overlap the native-loader interval and must not be
summed as disjoint phases. Successful hash-byte accounting uses the full-file
function contract and pre-call stat size; a failed/in-progress hash remains
admitted but not completed. mmap/safetensors/image-decode/library work is not
individually instrumented. `rchar`, `read_bytes`, logical file work and storage
occupancy are separate quantities; there is no physical disk bandwidth cap.
Known logical reads are rejected before exceeding their allowance; observed
`rchar` is also checked. This is not a claim to intercept every possible mmap
page or bound all filesystem traffic. The conservative 256 GiB allowance is
unchanged; metadata-only binding construction avoids an extra native hash pass.

Each stage has a 5 MiB combined logical-byte envelope, including at most 4 MiB
plain child log. Logs are not embedded or escaped into a JSON receipt. All writers
share a small-directory write lock (at most two seconds waiting), inventory the
current files, and admit the **serialized** JSON plus conservative temporary/link
occupancy before atomic publication. Normal controls are at most 1 MiB; COUNTERS
and CLI_RESULT are at most 64 KiB each. Other writes reserve 256 KiB for the final
failure/result receipt (at most 128 KiB, including staged duplication), and an
additional 128 KiB for counter staging. COUNTERS is replaced in place; there is no
unbounded one-second JSONL. Once an attempt is admitted, exceptions retain prior
files and write a bounded FAILED receipt. A rejected admission creates no child;
an existing consumed stage is never rewritten. The fit's original
RESULT retains metrics/exposure in its 64 MiB budget; CLI_RESULT stores only its
small summary. SP0's two exact W claims total 112,340,512 bytes including staging
and metadata. All eight already planned W claims total 546,104,992 bytes, below
1 GiB. Four fits add at most 256 MiB separately. Allowing 32 MiB controls/logs per
arm gives 948,758,176 bytes combined, still below 1 GiB without another tensor
cache or copy of native masters. Actual retained partial bytes remain charged.

Real native load/extraction speed, GPU/RSS peaks and CPU-fit cost remain
NOT_MEASURED. The current four dev exports and SP2 training were not touched.
The test record covers only control checks and tiny owned CPU child processes.
The final independent review must bind the integrated v2 runtime, generated
controls, actual stage admission, stable v7-derived migration and live space
receipt. This package is implementation/preparation evidence, never an efficiency
or dispatch PASS.
