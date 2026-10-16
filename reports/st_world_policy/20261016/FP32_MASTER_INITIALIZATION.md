# Fresh optimizer initialization

The installed DeepSpeed 0.16.9 engine converts execution parameters to BF16
before ZeRO2 clones its FP32 master partitions. Loading an FP32 parent before
`deepspeed.initialize` alone therefore did not preserve all parent values in
the new optimizer. The previous diagnostic artifacts remain unchanged.

`tools/st_world_policy/fp32_master_init.py` captures bounded, named FP32
partitions before that conversion. After initialization it validates the actual
native parameter order, group layout, padding and fragment addresses on every
rank, then restores the original FP32 values to the existing native masters.
The BF16 execution tensors, gradient dtype and RNG remain unchanged. This is
an initialization repair, not a new optimizer or precision backend.

Both `train_world` and `train_flow_policy` call the helper only for fresh
initialization. Native resume skips it. A world-run budget continuation also
skips it and keeps the parent's actual optimizer state; it does not retroactively
claim to repair the initialization of an older parent. New run identities record
this distinction. Flow numeric execution schema v2 records the repair, while
the strict deployment reader continues to recognize the original v1 boundary.
Unknown schemas and incomplete v2 precision contracts are rejected.

Evidence:

- Helper owner: `050de7ae0a00ef241f744fa4df15e7b18ada38a3`; 39 tests passed,
  including actual installed DeepSpeed CPU/Gloo native initialization, update,
  save and strict load.
- Independent Correctness review: 38 CPU/Gloo tests and a separate actual
  two-rank engine probe passed. This does not substitute for CUDA qualification.
- Integrated trainer/reader regression: 89 passed, 1 explicitly opt-in native
  CPU probe skipped in this invocation, 21.34 seconds. That probe has separate
  owner and independent evidence above. Log:
  `/mnt/project/st-world-policy-20261016/audits/flow_policy/fp32_master_init_v1/LEAD_WIRING_REGRESSION.log`.
- CUDA/FusedAdam and full-model integration: pending at this commit.

The old two-step Flow smoke's parent-to-step-zero parameter difference was
startup BF16 rounding, not learning. Independent all-element measurement found
step-zero to step-two update L2 0.005863143605, versus parent to step-zero
L2 0.6192573078. Neither diagnostic constitutes a planning-gain claim.

New formal runs still require approvals bound to their final integrated source,
data, targets, numeric execution and resource registration. The helper does not
grant formal qualification.
