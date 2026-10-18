# Lead admission contract

No executable admission is supplied with this implementation. Placeholders are
rejected. Lead must seal actual references only after the formal endpoint exists
and independent evidence/correctness/resources review is complete.

All references are absolute `{"path": ..., "sha256": ...}`; optional `bytes`
is also checked. Control JSON reads are bounded to2MiB. The admission has:

* `schema: st_rollout_endpoint_admission_v1`, `status: APPROVED`, real `issuer`.
* `binding`, `binding_sha256 = digest(binding)`, and
  `stable_binding_sha256 = digest(stable_binding(binding))`.
* `reviews.evidence/correctness/resources`, each a pinned independent review.
  Each review has schema `st_endpoint_supervisor_review_v1`, statusPASS,
  review_kind, distinct reviewer (neither issuer nor implementer),
  `binding_sha256` equal to the **stable** binding digest, and nonempty pinned
  evidence_artifacts. Evidence artifacts may be text/binary; their checksums
  are checked without forcing JSON.

Binding fields consumed by the implementation:

* Exact `source_sha`, formal `run_id`, current controller file SHA256 and literal
  `limits` from the source. These cannot change silently between phases.
* `sealed.commands/decision/comparison/fixed_inputs/analysis_template`, the
  exact five frozen scientific files identified in SOURCE_PINS.json.
* `training.registration/approval/identity/launch`. The only accepted run is
  `C1_rollout_std_fulltrain_seed42_official_4096_v1`; registration must be the
  actual formal path/hash in the source. Environment must equal the registration
  environment pin. Original mature three-review approval is checked again.
* `checkpoint.tag`, `checkpoint.complete`, `checkpoint.receipt_identity` from
  the actual complete `step_004096_<timestamp>` native. No latest selection or
  early fallback exists. The unchanged exporter verifies native bytes fully.
* `analysis_plan` with actual formal pins and reviewed world4 execution
  amendment. Statistical definitions and old C1 control remain sealed.
* `environment`: exact formal runtime environment reference.
* `route`: actual `host`, `hostname`, GPU indices0–3, four unique real UUIDs,
  `valid_from_unix`, `valid_until_unix` per the mature lease helper.
  Expected canonical host is training-vlawm-tsinghua-zt6. Declared route, UUIDs
  and lease are stable identity, independently reviewed and immutable.
* `dispatch_policy` must point to `/mnt/project/server_dispatch_policy.json`;
  `authorization` must point to campaign
  `/mnt/project/st-world-policy-20261016/resources/AUTHORIZATION_20261016T203442.json`.
  Both actual current files are checksum pinned and freshly checked. A frozen
  historical allowlist copy is not an accepted substitute.
* `storage_reservations` pins an actual current observation with
  `observed_unix` and nonnegative `future_reserved_bytes`. Observation age must
  be at most300s; shared free capacity must preserve1TiB floor plus other
  declared future commitments and this evaluation's1GiB output allowance.

Canonical digest is SHA256 of UTF8 JSON with sorted keys, compact separators
and allow_nan=False. `stable_binding` excludes only storage_reservations,
dispatch_policy and authorization. All other fields, including actual
checkpoint/analysis/output route/lease/controller identity, are immutable.
Fresh observations can be refreshed by a new signed/checksummed admission;
they cannot silently change the stable model, route or output identity.

At execution the stable binding is immutably saved to control/BINDING.json and
the complete fresh admission to a new RESOURCE_ADMISSION_<time>.json. Attempt
ADMISSION binds the stable digest, preserving restart continuity and the
two-invocation cap. A failed/partial/unpublished launch is not free. Global
official scoring remains the same mature1728 ledger, not a per-invocation cap.

Static review can be source/contract-bound with WAITING_CHECKPOINT. It must not
invent a real binding digest, GPU UUID or resource lease. Future real admission
is a Lead responsibility and must close these identities before execution.
