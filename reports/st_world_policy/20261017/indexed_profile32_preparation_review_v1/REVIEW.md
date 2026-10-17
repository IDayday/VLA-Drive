# One CPU PROFILE metadata factory: independent narrow resource review

**PASS_ONCE_METADATA_ONLY**, for the exact wrapper and pinned inputs in `ADMISSION.json`. Candidate commit: `f91df6d2e44e20544388fb2c412d800fab14d947`. Wrapper SHA256: `ebd7f0eb002606ed688267189e9db5cf46487266df6637ae57d8f68f75fe90d5`.

The underlying PROFILE source at clean runtime `2d5e9483a8c8bc80b71348211756f3cab6aa21b1` already has independent correctness evidence. This review only checks the bounded wrapper, exact CLI routing, and storage scope. It does not repeat numerical/model tests or execute the real factory.

## Verified scope

- Three sequential commands share one observed Python process: `prepare-profile-migration`, indexed `teacher_provider`, and the pinned `relocate_config.py`. Their option names and paths match the actual source parsers. No command invokes training or a teacher builder.
- Source migration, descriptor, three configured metadata files, relocated config, and config lineage are written under the fresh `profile32_v1` child only. The relocation writes the future checkpoint path as a config string; it does not create that directory. The proposed 80 GiB training allowance is not admitted here.
- Runtime/source, helper, handoff, recipe, data manifest, Base binding, original descriptor/reference config, target contract and relocation script are pinned. Original parent artifacts remain in place.
- CUDA visibility is empty. The child imports torch for the existing counter/PCA machinery but does not instantiate a model or optimizer. The already reviewed factory may read the original 1,590,864-byte PCA payload; that read is reserved explicitly. No new teacher tensor is generated or written.
- The original strict counter reader is retained. Only the exact attempt's `COUNTERS.json` read takes the existing writer lock. Small CPU checks show lock/read/unlock ordering, unchanged handling of immutable controls, and propagation of a lock timeout without reading. No real factory was called.
- The two-second lock wait remains fail closed. It can add up to two seconds to monitor responsiveness, plus ordinary I/O/scheduling; this is polling, not a hard cgroup cap. The separate freshness `stat` is not itself under the wrapper's lock, but publication is atomic replacement with no absent-name interval. Other read failures remain fail closed under the original supervisor.
- Fresh `ROOT.mkdir()` consumes the one-shot scope before a child is spawned. Existing scope refuses retry. Only the new owned child session can be stopped by the reused supervisor.

## Budget

The existing parent contained 15 paths and **65,536 charged bytes**, including directories and allocated blocks. Both the new profile child and future checkpoint root were absent. The original parent allowance remains **32 MiB total**, not 32 MiB per child. There is no added campaign metadata reservation: the existing effective reservation remains **1,264 MiB of 4 GiB**.

The exact seven application outputs contain bounded control schemas and pinned paths/identities/configs; no scene tensor or large per-record index is serialized. Original configs and descriptor were checked as small metadata. The wrapper checks 8 MiB initial headroom, uses a 4 MiB per-file ceiling and the original 5 MiB control-directory envelope, reserves 256 KiB before the final receipt, and checks the entire parent again afterward. This approval relies on this exact finite set of small output schemas and pinned inputs; it is not a generic guarantee that seven arbitrary 4 MiB files plus controls fit the parent. Adding outputs, copying complete indices into JSON, or changing the input/source pins requires a new review.

| Resource | Bound |
|---|---:|
| CPU affinity | 0, 1 |
| Child observed RSS | 2 GiB |
| Child runtime | 180 seconds |
| Abnormal owned-child cleanup | Up to 40 additional seconds |
| Total logical-read allowance | 4 GiB |
| Explicit PCA reserve | 1,590,864 bytes |
| Parent read reserve | 16 MiB |
| Parent-inclusive metadata | 32 MiB, existing allowance |
| GPU / new teacher tensors / optimizer steps | 0 |

The logical-read counter and RSS are sampled; transient overshoot between samples is an existing documented boundary. The finite 16 MiB parent reserve covers this wrapper's small controls/source reads, not a permission to scan a cache. No cache ledger, cache policy, old artifact, or global reservation was modified by this review.

## Single permitted invocation

Admission SHA256: `30f6c8db6ed20fab5731d677bdb4159296ff43ba95e5424e28393324bcec730c`.
Valid until Unix `1792225549.3007345`; start no later than `1792225329.3007345` to preserve the full runtime and cleanup. Lead must verify the exact script hash and these times before invoking it; this wrapper does not itself parse the independent admission token.

```bash
env PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python \
  /mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261017/indexed_profile32_preparation_v1/prepare_bounded.py
```

Success requires its actual exit/RESULT and expected output hashes. No throughput, training correctness, or completed preparation is claimed by this approval. An incomplete attempt stays in place and is not automatically retried.

Review reproduction (source/metadata only):

```bash
PYTHONDONTWRITEBYTECODE=1 taskset -c 0,1 /root/miniconda3/envs/ddp/bin/python \
  reports/st_world_policy/20261017/indexed_profile32_preparation_review_v1/check_wrapper.py
```

The check script records this particular admission; rerunning it does not authorize renewing an expired token without review. No actual factory, GPU, model, tensor cache, training, or ledger operation was executed by the reviewer.
