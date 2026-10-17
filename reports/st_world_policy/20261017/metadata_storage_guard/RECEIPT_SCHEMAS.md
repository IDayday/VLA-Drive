# Exact metadata admission records

These describe implemented inputs, not claimed actual registrations. Paths and
SHA values must come from Lead's actual evidence. No example uses fake runtime
paths or creates an approval.

`st_metadata_pilot_inventory_v1` has `roots`, each exactly the return from
`tree_inventory(root)`: root, charged_bytes, entries. Entries include relative
path, kind, device, inode, logical size, allocated bytes, mtime_ns, ctime_ns.
Directories count; no payload bytes are read. Use a fixed bounded census.

`st_metadata_writer_quiescence_v1` requires:

- status `QUIESCENT`, owner_role/future_admission_owner `Lead`.
- bindings: old_policy_sha256, new_policy_sha256, old_ledger_sha256,
  pilot_inventory_sha256, checkpoint_sha256.
- checked_writer_roots: exact old declarations + legacy roots + pilot roots.
- active_or_unknown_writers `[]`, queued_writers_disabled `true`.
- process_evidence: nonempty list of `{path, sha256}` pinned actual JSON receipts.
- checked_at_unix and valid_until_unix.

`st_checkpoint_storage_commitments_v1` requires status `CURRENT`, checked and
expiry unix times, and current remaining_bytes. Its pinned bytes are rechecked
at every writer check. Commitments add to the separate free-space floor and are
not charged as cache. Keep its validity horizon bounded to the actual build.

New policy `metadata_package` fields:

- schema `st_metadata_storage_package_v1`, maximum_bytes <= 4294967296.
- baseline_policy_sha256, baseline_ledger_sha256.
- baseline_claims: original exact root → declaration mapping.
- baseline_legacy_reservations: original exact ordered list.

New declaration keeps existing output_bytes/staging_bytes/metadata_bytes and
adds identity_kind `st_metadata_build_registration_v1`, registration_path and
registration_sha256. Its cache_identity equals registration_sha256, explicitly
identifying the metadata build rather than its future result.

Build registration fields:

- schema `st_metadata_build_registration_v1`; output_root.
- writer_kind: current_index, future_index or future_preparation.
- storage_requirements: exact declaration amounts.
- limits: max_disk_bytes (sum of all three amounts), max_rss_bytes <=4 GiB,
  max_seconds <=14400, max_entries <=20000, max_depth <=4 (index requires0).
- file_patterns: exact anchored-full-match regex list, including admission,
  registered controls, SQLite/temp/journal files or future shards as applicable.
- directory_patterns: exact allowed relative directory patterns; empty for index.
- optional preexisting_pins: at most eight `{relative_path,sha256}` small files;
  their set must exactly match preexisting files on first admission.
- thin runner only: exact command argv, exact cwd,
  callback_protocol `st_metadata_storage_check_v1`.

For future preparation `--plan` parent is the output root. For index build,
`--output` binds the root; index verify uses `--index`. The actual owner parser
retains all scientific/source/identity checks and rejects unsupported options.

CLI help is implemented for:

```text
python -m tools.st_world_policy.metadata_storage migrate --help
python -m tools.st_world_policy.metadata_storage run --help
```

No actual migration or run command is supplied until real pins and independent
review exist. The legacy v4 policy is never edited in place.
