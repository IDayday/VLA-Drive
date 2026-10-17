# Bounded cache decision and current evidence

The user's storage constraint is binding: no multi-TB future feature prebuild.
This is an execution/storage change; the registered teacher precision, labels,
sample order, horizon/patch sampling and training objectives stay unchanged.

The active aggregate teacher-cache budget is **768 GiB**, including existing
reservations, indexes and staging. It is not permission to add another768 GiB.
Current train tensor files total586.123482 GiB according to their complete
receipts. The budget ledger reserves645.318253 GiB in total, leaving122.681747
GiB for any separately admitted additions. The independent filesystem free
floor is1 TiB; native checkpoint reservations are separate. Filesystem free
space above this floor is not permission to expand the cache cap.

Policy: `/mnt/project/st-world-policy-20261016/resources/CACHE_STORAGE_BUDGET_POLICY_v4.json`.
Budget/reuse evidence: `audits/temporal_bounded_storage_feasibility_v1/` under
the same campaign root. Existing current caches and active training are protected.
No cache or checkpoint was deleted and no full future build was launched.

The implemented future provider reuses complete same-role records by exact
image SHA plus teacher contract, then computes genuine misses. A bounded RAM
LRU stores independently owned records. It writes **zero tensor bytes to disk**.
The first version still fetches all four declared future times, matching the
existing dataset semantics. It does not skip a missing valid target, change a
mask, select a different sample, or use a lower-precision substitute. A genuine
missing source retains its original mask; compute failure raises instead.

Production source and tests were cherry-picked from `a2a383c1` plus
`214cf43b`; the three implementation/test files are byte-identical after
integration. The latter commit fixes an independently discovered source-lineage
check missing from the executor-only path. Independent tests from `f57c7a92`
were also integrated. Frozen running runtimes `fdf25f99` and `6714b1e9` were not
edited or restarted.

Actual evidence is deliberately separate:

| Evidence | Result and limit |
|---|---|
| Initial provider/cache/request CPU suite |46 passed in8.66s on initial provider source; not a final-source aggregate rerun |
| Source-lineage fix focused suite |12 passed in3.66s, including9 new rejection cases |
| Independent final provider CPU/Gloo review |28 provider cases passed in8.69s;7 independent adversarial cases passed in3.45s |
| Independent Evidence review |21 static/source/hash checks passed; source-lineage finding closed |
| Real DINO/real-image numerical equivalence |NOT RUN at this report snapshot; bounded64-scene runner preparation is separate |
| End-to-end temporal throughput/resume |NOT RUN; module tests are not trainer qualification |

Independent reviews, under the campaign root:

- `audits/evidence_bounded_future_provider_v1/EVIDENCE_REVIEW.json`, SHA256
  `67de99a51c0a60d725d7fc60019f3957cee1778aab617d23057efdb62fa8a888`.
- `audits/correctness/BOUNDED_FUTURE_PROVIDER_REVIEW_v1.json`, SHA256
  `8fa494f519840ec0a6a12e1fe13c85ad6985a2321a21956e828c46c8dc4c01a8`.

Remaining integration is explicit: trainer/config registration, main-thread
provider injection (`loader_workers=0` for this synchronous reference path),
an explicit new PCA application scope using the unchanged basis, and real
teacher/gradient/RNG/throughput verification. The provider is not currently a
selectable production training backend. Legacy C1 and cache-derived SP1 miss
generation are rejected until a matching derivation is explicitly implemented;
the spatial winner has not been selected. There is no temporal launch approval
or claim of training speedup from these CPU results.

The pending64-scene qualification reads existing small immutable references,
keeps generated tensors in bounded RAM and stores only small measurements. Its
purpose is to establish numerical and execution costs before choosing a temporal
deployment. A new tensor cache, automatic precision change or larger disk budget
is not its fallback.
