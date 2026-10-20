# Independent review: Route A repair admission reuse

Target: `c38747dc9f5bc96d367817eae7a92a9a75edcdf9`, authored by Lead.
Reviewer: feedback/G0 agent, independently reviewing this Route A parser delta.
Conclusion: PASS for replacing repeated first-access native reconstruction with
the already independently reviewed strict native-inventory admission.

The changed production block calls `validate_compact_dataset(kind='repair')`.
That helper admits only the exact deterministic native output and reuses it only
under the full current source and dependency inventory. Every subsequent line
from `self.reader = context_reader` to the end of the trainer is byte-identical
to the prior version, including individual row/native/context/label checks,
role isolation, physical trajectory reconstruction, sampling, losses, gradients,
training identity and checkpointing. The validation receipt is stored as an
instance attribute; it is absent from the input manifest and training identity.

Two targeted negative tests passed in 3.97 seconds: independently rehashed compact
labels cannot substitute native provenance; invalid roles and observation joins
are rejected. No production source was edited by this independent review.

## Actual 64-scene CPU metadata preflight

`REPAIR_ADMISSION_REUSE_REVIEW_64.json` contains actual full comparison evidence.
Old and new classes read the same pinned real Base context and repair manifest.
The old class's expensive reconstruction call was delegated **only within this
review process** to the actual warm strict owner, avoiding another whole-bank
native reconstruction. Its individual per-row native checks remained enabled.
This is an admission/metadata comparison, not a benchmark of the old full-build
implementation.

- Both classes executed 128 individual feedback/native checks.
- All 64 entry contents were identical, digest
  `6d1f438ae25d146f4a381c3755aec65e4df877bb3685d510d09e5deef9f51282`.
- The 45 fit scene IDs, 19 validation IDs, their order and roles were identical.
- All 11 label names, units and normalization scales were identical.
- Manifest, Base/context and G0 registration identities were identical.
- New validation reused the standard immutable receipt with 6436 pinned files.
- Receipt information did not enter the manifest identity.
- No model forward, optimizer update, simulator call or GPU operation occurred.

The old compatibility read took 28.9863 seconds and the subsequent new read
1.2988 seconds. These runs have different cache warmth and do not establish a
pure speedup ratio. The demonstrated structural improvement is removal of the
unnecessary whole-bank strict build before each Route A arm, while preserving
each row's native and context checks.

This review permits the unchanged A0/A1/ScoreRefiner parser to consume an already
verified repair bank. It does not establish any learned correction gain, teacher
generalization, scientific superiority or deployment qualification.
