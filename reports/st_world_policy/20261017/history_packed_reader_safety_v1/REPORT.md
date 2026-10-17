# Packed history reader no-follow repair

Base `df68dc710d6a27cbc6009eb42a9db19d5e9b24fb`. Independent reviewer identified check-then-Path.open replacement and FIFO blocking. This patch changes only the packed reader and adds focused fixtures/reports. No history mathematical fields, selectors, labels, default per-file behavior, writer, ledger, current64 pilot or active runtime was changed.

The implementation follows the existing `prepare_future_metadata.metadata_parent/metadata_reader` practice locally to avoid a circular runtime import: walk from a directory FD with O_DIRECTORY|O_NOFOLLOW; open the leaf with O_NOFOLLOW|O_NONBLOCK; reject nonregular or multiply-linked files with fstat; enforce the1MiB record bound and within-file offset; compare dev/inode/mode/nlink/size/mtime/ctime before/after; re-walk the original namespace no-follow and compare the published file identity. Existing record SHA, semantic validator and token/log/current-input checks remain mandatory. Descriptors close on every failure. A FIFO is never read or waited on.

Tests cover normal bounded reads; FIFO/directory/hardlink/leaf or ancestor symlink rejection; leaf replacement exactly at open (pread is forbidden in this test); mutation after opening; ancestor rename/symlink replacement; range and traversal rejection; original history behavior and packed scale/recovery tests. `TEST_LOG_FINAL.txt`:27 PASS. CUDA hidden, CPU10/11,35s outer test cap; no real data/model access or actual dispatch.

The first test run retained in `TEST_LOG.txt` had26 PASS/1 FAIL: an in-place same-length rewrite occurred within the filesystem's coalesced timestamp tick, so fstat versions did not change. The fixture now gives that write an explicit observable mtime change to deterministically test the version check. We do not claim stat versions detect every same-tick overwrite. The selected record still must equal its pinned SHA, so a changed record read from disk fails checksum; returning already-read bytes matching the pinned immutable record is not an outside-file read. Negative evidence is retained, not overwritten.

The reviewer's previous Path.open-hook counterexample targets the removed open call; its next review should inject at os.open/pread instead. This patch has a corresponding precise at-open symlink race test and a nonblocking actual FIFO fixture. Independent re-review is pending, not self-approved.

No full build, new admission, claims, model/teacher tensors, GPU work or push occurred. Full-history performance/coverage remain untested.

Reviewer follow-up: files/default mode must not borrow a packed-only census registration. `prepare_causal_history.validate` now rejects any packed layout or inventory_strategy on its files branch; legacy files registrations with no declared strategy remain valid. Added negative/legacy behavior test. `TEST_LAYOUT_BINDING.txt`:28 PASS. This additional change is confined to configuration consistency; no read/mathematical path changed.
