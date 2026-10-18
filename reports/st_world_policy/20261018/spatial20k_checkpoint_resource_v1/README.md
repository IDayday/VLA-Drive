The existing 2,703 GiB commitment covers the original nine runs and later bounded tasks. It does not include SP2/SP3’s new 5k-to20k saves. The new resource allocation retains all of it and adds 720 GiB per arm: 15 regular saves, two nonperiodic or failed-save slots and one in-flight slot, using the existing 40 GiB checkpoint bound. Logs and controls remain inside each inclusive720 GiB root cap. Additional attempts cannot silently enlarge it.

The combined checkpoint receipt is4,143 GiB. Capacity separately retains the1 TiB floor and all potential cache growth up to768 GiB, crediting only539,427,538,104 B from the immutable native384 completion receipt. Other completed cache variants are not credited. A separate10 GiB metadata buffer is also retained. The exact fresh byte arithmetic is in REVIEW.json.

Lead’s snapshot binds SP2 to tsinghua-zt6 GPUs0–3 and SP3 to tsinghua-zt2 GPUs0–3. Both resource leases last48 hours; each invocation remains28,800 seconds, with the old outer29,400+180-second limit. Fresh device/environment/host capacity and the three independent formal reviews are still required. This author provides resource facts and does not sign a formal EFFICIENCY gate.

No model/checkpoint tensor, SSH, GPU query, launch, policy mutation or cache deletion was performed. Subsequent full-history TRAIN admission must use this combined checkpoint receipt.
