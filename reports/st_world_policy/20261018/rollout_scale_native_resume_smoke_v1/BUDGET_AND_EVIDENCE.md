# Finite execution and evidence boundaries

Training paths total8 updates/256 scene exposures,1024 primary official calls,
zero retries. Each arm has a durable512-call ledger across interruption. Exactly
three planned invocations: A initial4, B initial2, B one resume2. No automatic
restart or extra invocation is allowed; unknown/partial attempts remain charged.

The resource layout is pinned from the independent Systems plan, not fabricated:
RESOURCE_LAYOUT.json SHA256 d5b4dbd394b9ef2563b8bffee4bc0da2336d4e74235844ed99d75ce2b97d2dfb.
Actual available host/UUID/lease must be rechecked at dispatch. Fresh RAM640GiB;
each group256GiB RSS, aggregate512GiB, each GPU48GiB driver usage. Eight GPUs
are reserved for this finite test; no existing task is displaced.

Observed metadata/stat only: actual751 native0 is4,917,269,512bytes. Independent
Systems' mature post-update native is11,473,306,504bytes. Normal two-arm0/2/4
storage is55,727,765,040bytes. Allow104GiB total: at most8 native directories
including partial, each12GiB, plus8GiB feedback/control/temp. No cleanup of old
checkpoints is performed. Shared free launch minimum1634GiB; runtime preserves
1530GiB for the agreed floor and other projects' commitments.

ORIGINAL_S0_STAT_ONLY.json lists17 original S0 payload sizes without reading
their contents:37,597,351,782bytes. Twelve rank reconstructions plus three
rank0 checksum passes nominally imply15 passes/563,960,276,730bytes before other
framework/native work. This explains using a cumulative1TiB sampled rchar guard
instead of the initial512GiB draft. /proc/io rchar and read_bytes are separate;
samples are preserved across PID reuse and across initial/resume phases. These
are diagnostic lower bounds for short-lived children/mmap and do not claim a
physical disk quota, full read accounting or bandwidth. Output/IO scans are10s
sampled guards; RSS/ownership are checked more often, GPU driver queries5s.

Training controller uses one global3600s allocation clock across both phases,
reserving its final60s for owned cleanup and applying remaining-time GNU timeout
to every child. Comparison is separate: CPU120s, RSS2GiB, input64MiB, output16MiB,
two Torch threads. Additional outer owned cleanup reserve is180s: total admitted
execution envelope3900s. These scopes are distinct;3600s is not claimed to cover
all comparison/cleanup cost. Only owned sessions created by the controller may
receive cleanup signals. Missing cleanup confirmation is failure, not success.

Observer costs are persisted per rank and label as audit_seconds, separately
from original update timings. Complete master/module hashing transfers data and
synchronizes; this SMOKE cannot be used as a throughput result. No duplicate
multi-GB tensor dumps are created. The CPU comparison does not reopen multi-GB
native payloads. Strict live-state identity uses SAVE versus actual LOAD
observations while the actual trainer verifies native bytes.

Prerequisites still unexecuted: actual fresh admission, two independent observer
reviews, real GPU4+4 run, real pause/native2 receipt, actual strict load and live
RNG match, and final A/A/continuation outcome. Existing older fdf native recovery
is only a helper reference, not evidence that751 passed this test.
