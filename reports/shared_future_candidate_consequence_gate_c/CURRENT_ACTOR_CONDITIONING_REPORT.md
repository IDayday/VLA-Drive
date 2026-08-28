# Current-actor Conditioning Audit

- Derived target scenes covered: 45,377/45,378 (99.998%)
- Shards: 32/32; failures: 1
- Sampled shape/finite checks: 512/512
- Status: FAIL

O1–O13 condition on current-time dynamic actor slots in the current-ego frame,
in addition to the six current-scene summary values. These fields use current
annotations only, never logged-future annotations. They are a conservative
structured oracle control; the deployable model must infer this information from
its current images rather than consume annotation tensors.
