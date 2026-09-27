# P1 no-update injection evidence

64 unchanged overfit-manifest scenes, original three front cameras, original Qwen BF16/DiT FP32, one proposal,10 FM steps, same per-scene noise. No parameter updates. Actual zero-gate adapter invoked64 times.

| Layout | Native action hidden max difference | Physical future XY max difference | Heading max difference |
|---|---:|---:|---:|
| legacy_pre_action, untrained |124|6.882837m|.347617rad|
| append_tail, untrained |0|0|0|
| append_tail + gate0 |0|0|0|

Every append-tail native prefix embedding, mRoPE position, attention mask, visual mask and final prenorm hidden tensor is bitwise equal to native A0. Native action positions are separately asserted by CPU tests. No tolerance relaxation: existing BF16 atol/rtol .02 retained, actual differences zero. Appended world tokens run through Qwen; adapter also actually executes. This is initialization fidelity, not learned planning effectiveness.

`INJECTION_SHIFT.json` reports differences of normalized 4-channel actions under its historical trajectory field names. `INJECTION_SHIFT_PHYSICAL.json` decodes saved outputs via actual `infer.deal_action_1225` and reports metres/radians. Do not read the normalized maxima as metres.

Runtime charged .03900546 GPU-hours,0 optimizer steps. No PDMS rerun for these diagnostic outputs. P1 compares outputs only; it does not claim planning improvement.
