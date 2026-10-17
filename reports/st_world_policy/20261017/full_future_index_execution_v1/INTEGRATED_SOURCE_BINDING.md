# Explicit index controller dependency rebind

This source-only integration follows index adapter
`15967d1de0615ed104a2c326a09b4ac7dd63d2ea`, bounded completion-reader correction
`ae585cb4bf710983c5b1efa814815133e910909d`, and the isolated core exit-race fix
`4f28ee0462a74a05da763b98cc896d116c683425` (cherry-picked here as `ce0bf395`).

Exact integrated sources:

- `full_future_index_execution_v1/run_index_once.py`:
  `eb9713666a84e8ffba7a55c75f656bb6133218dc626d772200189adad6d41a9a`.
- `full_future_metadata_preparation_v1/run_build_once.py`:
  `8674aea1f2a3df894bb3c451287e6a14b5c20d59b01efe4883158f73a3690644`.

`CORE_SHA` explicitly changes from `a775a875…` to `8674aea1…`. No runtime source,
limits, claims, ledger, actual admission or old evidence is rebound implicitly.
An independent reviewer must approve this exact pair before any dispatch. The
existing development/train builder runtimes and original helper hashes remain
unchanged. This integration does not itself authorize an index run.

`COMBINED_REGRESSION_FIXED_OUTPUT.txt`: **136 passed**, 4.16 seconds pytest time.
Wrapper elapsed 4.624 seconds; peak RSS 68,395,008 bytes. CPU0/1, 2 GiB
address-space cap, 180-second timeout, <=21 MiB total fixture payload bound.
The suite includes metadata preparation, index schema/admission/policy,
bounded descriptor reads, deterministic exit races, real tiny child lifecycle,
and pre/post-wait accounting. No GPU, real builder, real indexer or live ledger.
The CLI `--help` exits 0 with the existing admission and admission-sha256 flags.

The initial combined **113 passed / 1 failed** remains intact in
`COMBINED_REGRESSION_INITIAL_OUTPUT.txt`; its false descendant failure is not
reclassified. This file records a new source binding and new test execution.
No live TRAIN tree or in-progress job has been modified.
