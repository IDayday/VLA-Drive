# Road-failure opportunity: metadata-only frame plan

Status: `NOT_APPROVED_FOR_DIAGNOSTIC`. This preparation never imports torch,
opens a `.pt`, reads a metric-cache payload, initializes CUDA, or scores a scene.
It reads already completed TRAIN compact receipts only. The single formal C1
scale candidate proceeds independently.

The evidence population is the original C1 4096-update TRAIN ledger. Eligible
groups have nonempty `support_scene_slots.D`, i.e. the four original candidates
have different official DAC values under one context. A passing sibling is an
existing feasibility witness, not a trajectory to copy or a replacement target.

Fix seed42 and domain `st-road-failure-direction-v1`. Sort the 4096 ledger keys
by SHA256 of UTF-8 `domain + NUL + 42 + NUL + ledger + NUL + key`; break ties by
the key. Visit compact receipts in that order. Within each ledger, sort its
eligible D slots by SHA256 of `domain + NUL + 42 + NUL + scene + NUL + token`,
then token and slot. Keep the first occurrence of each unique scene and stop
at32. This is a deterministic two-stage ledger-prefix sample, not a claim of
uniform independent sampling over all road-failure candidates. No proxy value,
gradient, geometry result, DEV score, policy progress or outcome affects order.

The future, separately admitted preparation must inspect only these selected
groups' original records, validate one shared context, and sort candidates with
DAC=0 by SHA256 of `domain + NUL + 42 + NUL + candidate + NUL + candidate_id`;
select the first. Select one DAC=1 sibling by the same rule as a witness. If a
group cannot supply both, a checksum fails, a native road flag at t0 is true, or
an asset is absent, retain the selected scene as unresolved; do not replace it.
The final32 candidate IDs/raw hashes/trajectory hashes/context identities and
selection hash must be frozen before geometry, derivative, or scoring execution.

Compact receipts preserve group IDs and input hashes but omit per-candidate
DAC flags, physical trajectories and raw action tensors. This preparation must
therefore report candidate binding as pending rather than infer DAC from total
PDMS=0. The frame is a concrete metadata pointer set, not a fake execution-ready
registration. Original scores can be reused only after strict original identity
verification in the separately admitted preparation.

The metadata reader has a6MiB additional read cap,120CPU-second hard process
limit, two-core affinity and2GiB actual RSS check. Prior exploratory reads get
a conservative8MiB allowance, keeping this task below16MiB. Missing metadata or
budget exhaustion yields a partial frame with an explicit blocker, never a new
sampling rule. There is no automatic second draw.
