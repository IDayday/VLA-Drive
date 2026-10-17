# Actual full-population provider preparation

The single admitted invocation completed the two original metadata CLIs on
runtime `a8b0036f`: source migration followed by the indexed teacher provider.
The original factory performed its full database/dependency checks and ordered
comparison against the 101,592-scene training population. It bound the existing
PCA basis; it did not refit PCA, run a teacher, read images or train a model.

Measured wall time was 178.623 seconds, peak aggregate RSS 707,788,800 bytes,
and charged logical reads 9,664,517,081 bytes. These reads include original
validation passes; they are not new disk storage. New control/config outputs
occupy 50,459 logical bytes. New teacher tensor storage is zero. The supervisor
returned code 0 with no failure and zero remaining owned processes (three
verified identities).

The original runtime, completed indices, registered data population, teacher
precision and original PCA remain unchanged. The independent wrapper reviews
are in `full_population_provider_prep_correctness_v1` and
`full_population_provider_preparation_resource_v1`. Exact output checksums and
the immutable actual result receipt are in `RESULT.json` beside this report.

This completes actual metadata/provider preparation, not full-population
throughput or formal temporal-training qualification. The next bounded test
will use the full original sampler, at most 64 cumulative optimizer updates
and 128 scene exposures, and its own checkpoint/resource reservation. Its
weights cannot initialize formal training. There was no automatic retry,
ledger rewrite, checkpoint allocation, multi-TB cache build or new Navtest.
