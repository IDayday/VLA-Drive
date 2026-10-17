# SP1 exact storage-runtime compatibility resource review

Independent result: **PASS_NARROW_SOURCE_COMPAT_RESOURCE_ONLY**. Seven CPU control tests passed; 1.368 s wall, 28,123,136 B sampled peak RSS, 3,551,285 B observed rchar. Tests read only source/control metadata and exercised the real shared-package guard. Deep per-signature qualification was stubbed to previously admitted bindings to keep this review narrow; actual migration retains that original validation.

The original mixed-source rejection remains in SP1_dev_typed_migration_resource_v1. This explicit adapter proves that 581399 and 2a1e have identical non-storage tracked sources, and canonicalizes only the package-wide source comparison. Actual signatures, source identities, Base/environment checks and same-arm endpoint checks are unchanged. Unknown sources are rejected. The observer captures its read baseline before imports, so compatibility setup remains part of its original read budget.

The proposed policy preserves all 15 old claims, metadata baseline, legacy reservations and allowed roots. It adds only the completed SP1 dev plan declaration (26,085,776 B), resulting in 138,426,288 B current W claims. All four planned arms remain within the separately registered 1 GiB W package; 75 MiB controls and 192 MiB fit are separate. The conservative 2,383 GiB checkpoint commitment and 1 TiB free floor are retained without subtracting completed profile usage.

SP1 dev plan consumed 37,533,211,956 B of its 256 GiB role budget. Dev extraction therefore has **237,344,694,988 B remaining**, not a fresh 256 GiB. One 960 s extraction with 40 s cleanup, 64 GiB sampled aggregate RSS, one authorized GPU and the original one-loader lock is the next possible stage after actual additive migration. This report does not dispatch or claim extraction success.

Train plan is a separate role with 256 GiB total plan-plus-extraction reads. Its exact signature and consumed count must come from the actual stage before train extraction can be admitted. Fit remains the original 300 CPU updates after both actual extraction receipts. No plans, payloads, checkpoint tensors or prior successful stages were repeated.
