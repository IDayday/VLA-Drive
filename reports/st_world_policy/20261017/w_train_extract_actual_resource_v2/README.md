# Independent resource admission: one SP0 train64 W extraction

`ADMISSION.json` permits only the fixed TRAIN64 frozen-W extraction. Lead is the sole dispatcher. This does not approve a second plan, DEV repeat, fitting, new model training, index building, or automatic continuation.

The original computation remains clean source581399eb. The exact storage-only derivative2a1e140a and bridge sources are those independently reviewed in `correctness/w_storage_bridge_review_v2/SOURCE_CHECK.json`; its45-case evidence is inherited, not represented as tests rerun here. The actual bridge validation completed all runtime/dependency/signature/policy checks. The new policy retains all12 existing claims and the metadata envelope exactly and adds only the86,254,736-byte typed TRAIN W claim. The completed migration and actual ledger checksum were verified read-only.

The original immutable train plan is reused. Its37,575,105,055 charged logical-read bytes leave237,302,801,889 bytes within the original256GiB role budget. No standalone plan or native/model/image read was performed by this reviewer.

Actual limits: GPU0, CPUs2/3,960s plus40s cleanup coverage,64GiB RSS,24GiB GPU allocator limit, original32MiB control envelope. Old checkpoint commitment13f3 remains valid over this finite window. One inventory probe found the exact GPU UUID, at most64MiB allocated,0% utilization and no compute owner. Narrow retries did not probe GPU again. The original launcher must freshly recheck GPU/host/lease/locks at dispatch.

Two reviewer-harness failures are retained. First, a clean-tree git command returned128 while the review imposed a512MiB address-space ceiling; no stderr was retained, so its cause is unconfirmed. A separate identical git command subsequently returned clean. In the explicitly authorized retry, bridge validation passed, then the original validator correctly rejected reviewer affinity8/9 against requested workload2/3. Neither failure launched a workload or wrote an admission. The authorized final narrow check ran only original control validation and stage-input resolution on CPUs2/3: PASS in0.0431s,21.6MB actual self HWM,1.86MB process rchar. The earlier retry's external sampled peak was30.4MB/1.618s/3.41MB terminal reads. These are control costs, not extraction performance.

The admission and command are immutable outputs; actual execution success is unclaimed. No live ledger mutation, model payload, tensor creation, GPU allocation, or dispatch occurred in this review.
