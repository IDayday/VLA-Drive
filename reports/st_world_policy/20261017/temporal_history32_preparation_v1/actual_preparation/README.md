# Exact32 actual preparation artifacts — not an admission

These are real typed configuration files for the fixed32 controller, with
proposed resource/approval records kept explicitly unsigned. No builder,
GPU/SSH call, live-ledger read/write, output-root creation or reservation was
performed. The wrapper source is unchanged from `0d44879b`.

`PREPARATION.json` contains every exact path/hash and the input provenance.
`HISTORY32_STORAGE_REGISTRATION.json` uses the original real schema
`st_metadata_build_registration_v1`; its exact argv references the supplied
`BUILDER_BUDGET.json`, fixed32 selected manifest, original source roots and
packed history builder. CPU2/3 was within this preparation process's affinity;
fresh availability and lease remain the dispatcher's responsibility.

`PROPOSED_POLICY.json` copies pinned current policy
`230b052a037f844709403151ad025823cb7a5c17f213136430829b4ee88b634e`
and **only appends** one typed31MiB claim:23MiB output +4MiB staging +4MiB
metadata. The previous16 claims and every non-claim field, including core,
allowed roots, legacy reservations, metadata package and frozen-W package,
compare equal. There is no policy migration or ledger rewrite here. Existing
first admission must still validate the current state and precharge the new
claim; this static comparison is not an independent storage approval.

`CONTROL_RESERVATION_PROPOSAL.json` separately describes the1MiB owned control
directory. It is inside the combined32MiB invocation limit and is **not**
misrepresented as part of the31MiB artifact claim or existing ledger16MiB.
Lead must account for/adopt and create the plain control directory before launch.

The CP arithmetic is **2703GiB = historical actual2383GiB + conservative
unissued80GiB + new PCA256240GiB**. The actual2383 receipt is pinned as
`d16430ae3cc2cbd087f08d50c279add42e6a0616425d56f24bc6478286de9e47`;
its historical CURRENT status does not make its old time window current now.
The2463 number was only a would-be total in a STATIC HOLD review with
`new_reservation_issued=false`, not an issued receipt. PCA256's actual storage
boundary is pinned; its completed5000-step pause does not release its240GiB
commitment. No obligation is reduced using training progress.

`CHECKPOINT_COMMITMENTS_2703G_PROPOSAL.json` is deliberately
`PROPOSAL_NOT_CURRENT`, with null checked/expiry times and no reviewer.
`ADMISSION_PROPOSAL.json` is deliberately `PROPOSAL_NOT_APPROVED`, with null
lease times and no reviewer. The controller must reject these as admission.
Do not overwrite or simply relabel these historical proposal files: Lead and
the independent reviewer must create new CURRENT/APPROVED artifacts containing
fresh times, actual host/resource checks and new hashes; the actual admission
must point at that fresh CP rather than this proposal.

The admission retains the old field structure plus the fixed32 wrapper's
`builder_budget` pin. Its six controller source pins include the original five
reviewed files and the actual wrapper. The limits remain900s+40s, two CPUs,
4GiB RSS,16GiB aggregate logical reads and31+1MiB disk. No new monitor,
algorithm, source/runtime change or enlarged scope is introduced.

After independent review and real artifact issuance, use the exact
`run_history32_once.py --admission PATH --admission-sha256 SHA` command in the
parent RUNBOOK, under its CPU binding and bootstrap-inclusive GNU timeout.
No actual admission file is fabricated here. Wrapper tests were not rerun for
this static preparation. The generator explicitly read84,527B of small
source/JSON, besides bounded preliminary inspection; it did not read tensors,
large source/index manifests or the live CLAIMS file.
