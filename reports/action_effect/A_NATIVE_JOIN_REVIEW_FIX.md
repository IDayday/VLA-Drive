# Independent review correction: compact repairs must rejoin native receipts

A self-consistent compact JSON hash and individually valid feedback pins do not alone prove that every compact training row was derived from those native receipts. Following the shared G1 review rule, the production RepairDataset first deterministically rebuilds the compact datasets from the registered G0 complete scene/native receipts. It checks the rebuilt repair SHA against the actual training artifact before loading any row. Existing immutable outputs are reused; this is a checksum/provenance pass with zero simulation calls.

The production CLI exposes no bypass. Synthetic parser fixtures explicitly bypass native verification and never claim integration. A new negative test returns a different native reconstruction hash for an otherwise self-hashed compact artifact and verifies rejection. Focused entry suite:9 passed, one existing pynvml warning,3.90s, full log `A_NATIVE_JOIN_REVIEW_TEST.log`.
