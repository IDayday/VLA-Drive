# Retain all original source logs under an explicit bounded input limit

The independent resource precheck of existing full-train source metadata found
three original logs larger than the previous 128 MiB per-file admission ceiling;
the largest file is 171,848,564 bytes. This was a stat-only measurement, not a
successful deserialization or a measurement of peak parsed-object memory.

The preparation parser still defaults to 128 MiB. `validate_limits` now permits
an explicitly registered ceiling up to 192 MiB. Full-train preparation will use
one worker and its existing 4 GiB RSS hard limit. No scene is removed or log
rewritten to fit the previous arbitrary file-size ceiling. The source bytes,
selection, timestamps, target masks and teacher computation are unchanged.

The builder still checks each actual raw-log size before deserialization and
preserves bounded reads, disk use, wall time and resumable log shards. Actual
large-log RSS remains to be measured; this patch does not assert it will fit.
Development metadata can retain the old explicit 128 MiB setting.

Actual CPU validation: **60 passed** in 6.59 s, one existing pynvml deprecation
warning, using `test_prepare_future_metadata.py` and
`test_review_future_metadata.py`. The added test verifies default128, explicit192,
the unchanged RSS limit, and rejection of larger/noninteger/boolean/zero limits.
No live data, policy or runtime was changed by these tests.
