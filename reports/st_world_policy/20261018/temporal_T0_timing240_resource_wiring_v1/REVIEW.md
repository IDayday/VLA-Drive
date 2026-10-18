# T0 timing240 resource wiring review

**PASS_RESOURCE_WIRING_ONLY; actual dispatch BLOCKED_STORAGE.**

Reviewed frozen author commit `0becb15ef2e36179a61120fb70b7ed878dc84923`. This is an implementation resource review, not an execution admission. Independent correctness review remains a separate prerequisite. No commands were dispatched.

The thin controller retains the native fresh original-Base training definition: world4 / micro1 / accumulation8 / global32, original20k schedule and warmup1000, stop_after=save_every=240. It permits only one checkpoint directory including partial writes and exposes no resume or retry. All new training output, logs, monitor evidence and control artifacts are inside the inclusive40GiB bound. The existing periodic save is reused at the matching pause boundary. The source/runtime profile evidence is inherited; no model or throughput tests were rerun.

The actual storage function adds the independent4GiB C-evaluation reservation to CP remaining +768GiB cache +10GiB metadata +1TiB floor +40GiB minus own already-written bytes. Original4143GiB scientific checkpoint retention remains unchanged. Native/child/wrapper/cleanup limits are3600/3720/3900/180 seconds,1TiB logical reads and192GiB aggregateRSS. These are bounded admission limits, not measured timing results.

At the recorded observation, available bytes were4,246,827,794,432. Existing required free bytes were4,205,203,637,876; adding40GiB would require4,248,153,310,836, a deficit of1,325,516,404 bytes. Therefore no GPU admission is issued. The observation is not a quota guarantee.

The six immutable prepared-input JSON files total34,040 logical /45,056 allocated bytes. There is no measured33MiB requirement. Propose a separately identified1MiB allowance inside the existing10GiB metadata ceiling:10,107→10,108MiB, leaving132MiB. This proposal does not allocate timing resources. Generated runtime metadata is already included in own40GiB.

SP3 was observed at11,847;12,000 COMPLETE was not yet present. A future V4 may credit only that new known complete checkpoint using receipt/stat and inode deduplication. This prevents counting already-occupied storage twice; it does not create free space. A new checkpoint write and its proper credit reduce free and remaining together.

The running history attempt3 keeps its immutable V3 guard and original receipt. Future SP checkpoints can be conservatively counted twice by that static guard, causing a safe stop. No running pin, source, floor or ledger was changed. Timing remains blocked until fresh resource arithmetic and device observations pass; existing jobs are protected.

Evidence and exact hashes are in `REVIEW.json`. This review read only bounded source/JSON and stat/hash of six tiny metadata files, with no tensor payloads, metadata census, profile, launch or signal.
