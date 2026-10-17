# Completion receipt bounded-read correction

Baseline: `15967d1de0615ed104a2c326a09b4ac7dd63d2ea`.
Isolated worktree: `/mnt/project/VLA-Drive-st-index-bounded-read-20261017`.
Adapter SHA256 at this correction:
`eed6464ae67304136320a94fd29308ed1f232ae34f69a66766fb192628ad0b18`.

The original `read_small_actual` bounded the initial stat size but then used
`Path.read_bytes()`. A growing file could therefore exceed the claimed read
budget before the later version check rejected it. The corrected code opens
one descriptor with `O_NOFOLLOW | O_NONBLOCK`, checks its regular-file, nlink
and stable version against the path snapshot, and reads at most 1 MiB + 1 byte.
It rejects an oversized read, changed fd/path version, or parent symlink before
parsing or returning a new content pin. Atime remains excluded from content
identity. A FIFO replacement cannot make open wait for a writer. Short regular
reads are supported within the same total byte cap.

`BOUNDED_READ_TEST_OUTPUT.txt`: **65 passed** (12 new read/race tests plus 53
original index tests), 0.22 seconds pytest time. Wrapper 0.668 seconds, peak RSS
67,321,856 bytes, CPU0/1, 2 GiB address-space cap, 180-second wall limit.
Fixture payload upper bound 9 MiB; no live input/index/tensor/model read.
The growth fixture writes over 2 MiB but proves consumption is exactly
1,048,577 bytes before rejection. Other cases cover FIFO/symlink/large-file
replacement before open, path-inode or fd-mtime changes during read, parent
symlink change, pre-existing hardlink/oversize, valid exact-limit JSON and
short reads. No real indexer, builder, GPU or live ledger runs.

The original failed combined test evidence is retained unchanged.
This commit does not change `CORE_SHA` or certify real indexing; the reviewed
exit-race controller will be rebound in a separate explicit integration commit.
