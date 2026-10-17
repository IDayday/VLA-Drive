# Full-future metadata controls, prepared without execution

Source start: `23d56d2f`. This change owns only this report/control directory.
It reuses `metadata_storage.migrate_policy`, `run_registered_metadata`, the real
future preparation API and the existing request-only index API. It never creates
teacher tensors, changes a model, migrates the ledger or dispatches a build.

Lead authorized a reversible metadata preparation under
`/mnt/project/st-world-policy-20261016/audits/full_future_metadata_preparation_v1`,
within the already separated 128 MiB administrative allowance: CPU 0/1, up to
180 seconds, 2 GiB RAM, 32 MiB source reads and 1 MiB new controls. The actual
preparation took 0.128 seconds, peaked at 26,996,736 bytes RSS, counted 2,172,178
metadata source bytes and 2,452,845 process rchar bytes. Inclusive output before
the small final owner receipt was 78,213 bytes. No GPU, image, raw-log or tensor
was opened; no data, PLAN or live policy/ledger was modified.

The exact private receipts are indexed in `EVIDENCE.json`. They are proposals,
not launch approvals. The on-disk prepared status is
`PREPARED_NOT_MIGRATED_NOT_AUTHORIZED`.

## Real inputs and immutable build sources

Both already completed full-population PLANs are bound exactly:

| Role | Scenes | PLAN SHA256 | Build runtime |
| --- | ---: | --- | --- |
| dev | 1696 | `58fb4439094bf488becd0e4b9a1944be23899c96d603213e06dbde7939082781` | `VLA-Drive-st-metadata-build-dev-v1-20261017`, commit `1265b2376864ee929d95513f4f91e58d0910d147` |
| train | 101592 | `bffbad1f6378aa9e2a766bc0cd9445a511af6459f0a3b43dc18f969fcdfa2ef0` | `VLA-Drive-st-metadata-build-train-v1-20261017`, commit `838dd21f` |

Every PLAN implementation hash was checked against that role's build runtime.
The dev original 128 MiB and train 192 MiB raw-log implementations remain
distinct. The original PLANs were not rewritten to accommodate new source.

The separately reviewed metadata API change permits up to 16 preexisting pinned
metadata files. Both roots have exactly nine: PLAN/lock plus seven controls.
All nine are individually hashed in each registration, with exact filename and
directory patterns. None is deleted, relocated or ignored to satisfy admission.
The helper explicitly rejects the old eight-pin runtime. The two real root
inventories were also checked using the original API: 49,152 charged bytes for
dev and 327,680 for train, nine files each.

## Additive storage proposal and phase boundary

The prepared policy preserves all eight old claims, every legacy reservation,
all campaign limits and the complete frozen-W package. Only two preparation
claims are added. The metadata envelope makes the explicit version transition:

`st_metadata_storage_package_v1`, exactly 4 GiB →
`st_metadata_storage_package_v2`, exactly 8 GiB, retaining
`previous_package_sha256` and every original baseline field.

The original API confirms that reconstructing the prior package yields the
original identity. Frozen-W package bytes remain unchanged.

| Allocation | MiB | Status |
| --- | ---: | --- |
| Existing metadata claims/reservations | 1136 | Preserved |
| Train future metadata preparation | 2048 | Exact proposed claim |
| Dev future metadata preparation | 128 | Exact proposed claim |
| Train request-only index | 2048 | Planned reserve, not a claim yet |
| Dev request-only index | 128 | Planned reserve, not a claim yet |
| Separate administrative metadata | 128 | Preserved, never converted into a claim |
| Final total | 5616 | Within the single 8192 MiB metadata envelope |

The first policy therefore contains 3312 MiB of metadata claims/reservations.
Adding the separate 128 MiB administration gives 3440 MiB at this phase.
The remaining 2176 MiB index reserve is explicitly pending; the final total is
5616 MiB. Prior temporary PLAN-control allowance is absorbed by its prep root's
inclusive claim and is not charged a second time. Cache cap remains 768 GiB;
checkpoint space remains separate (2183 GiB commitments and 1 TiB free floor).

An index requires the **real future request identity**, which exists only after
the corresponding metadata build succeeds. The first phase therefore reports
`WAITING_REQUEST_ID`, rather than fabricating a runnable index claim. The same
small preparation helper supports a later index phase after matching COMPLETE
receipts are supplied; that phase adds only the two index claims and requires a
fresh explicit same-ledger migration. It does not run a build or change existing
registrations. The metadata policy alone never authorizes GPU work or training.

## Validation and remaining steps

`11 passed` CPU tests cover unchanged original package/claims/W/legacy, exact
version expansion, duplicate/replaced claims, invalid package size, nine retained
preexisting files, the 16-file limit, payload/symlink rejection, and output-budget
failure before file creation. CLI `--help` was executed. These fixtures do not
stand in for real future metadata builds.

`READONLY_API_CHECK.json` records original API checks of registration/resource
schemas, full existing file census/pins and metadata-package preservation. It
does not repeat native checkpoint or W tensor validation. An initial check-script
lookup incorrectly assumed every retained claim had `registration_path`; the
typed W claim does not. That diagnostic failed without changing anything. The
metadata-only lookup was corrected and its failure is retained in the report.

Pending before migration/build:

- Independent review of these exact controls and new runtime sources.
- Fresh Lead writer/queue census and a valid QUIESCENT receipt; the generated
  `QUIESCENCE_NOT_APPROVED.json` deliberately cannot pass the migration API.
- A fresh full-budget checkpoint receipt whose lifetime covers the actual build.
- Original outer finite CPU/RSS/read/root monitor and separate per-role dispatch
  admission. A short PLAN token cannot authorize the train build's four hours.
- Actual future builds, followed by real request identities and index controls.

See `RUNBOOK.md` for exact existing API commands with actual prepared paths and
hashes. No migration or build was run by this agent, and no launch approval was
self-issued.
