# Resource handoff and original-batch replay

Observed 2026-10-14, UTC. This changes worker placement only; all existing expert,
Scorer, sampling, optimizer, and scoring definitions remain fixed.

- All four E00/E01/E10/E11 pilot experts completed 5000 updates.
- The local waiting completion controller 2610513 had no children and was
  stopped before dispatch. Its single replacement is tsinghua-zt6 PID5452.
- Existing extension jobs: interaction PID6335, GPU0; safe_progress PID6410,
  GPU1. These are the final two of the six authorized expert attempts.
- tsinghua-zt5 GPU0 has an unknown real occupant and was untouched. GPU1–4
  restarted only four failed spatial worker identities (6395–6398). They read
  the same fixed topology and skip committed tensors. Worker1/5 have completed;
  worker3/4 are finishing their original missing records.
- Future S00/S01/S10/S11 remain limited to the original four 6000-update cells.
  No Base, Navtest, external teacher, or ACTIVE job was started.

The original W/A cache was produced using fixed microbatch layouts. Regrouping
an original two-scene tail into four, or turning a four-scene record into a new
two-scene tail, changed AMP language outputs. A naive restart with the same
regrouping would fail again. The source plan proves the original scene IDs and
positions; all its checksums are verified. Only mismatching records replay that
original batch. No old W/A or prediction is rewritten and the original 1e-5
check remains. All six offending scenes now have exactly zero raw/W/A error.
The numeric capture module and its cache key are unchanged. Additional cache
builder forwards are explicitly counted; deployment remains one shared forward.

Evidence: evidence/ORIGINAL_BATCH_REPLAY.json, evidence/REAL_SPATIAL_READER.json,
and runtime/tsinghua56 under the campaign root. The reader test compared every
stored tensor of 16 actual scene payloads and verified that repeated reads do
not rehash a complete unchanged payload. This is distinct from the full-manifest
coverage gate, which remains incomplete until every original scene is present.

Latest repository regression: 465 passed, 3 skipped, 4 warnings, 54.30 seconds.
The skipped tests are opt-in NCCL tests; actual campaign training is single GPU.
