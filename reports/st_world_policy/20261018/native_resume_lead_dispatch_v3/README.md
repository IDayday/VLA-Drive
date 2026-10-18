# Lead dispatch: bounded native recovery check

At 2026-10-18 20:48:35 UTC Lead actually started the reviewed v3 diagnostic
controller on Tsinghua zt6. The live launcher and four cross-session torchrun
ranks were independently observed; no process was called foreign or killed in
this v3 observation. Continuous4 loads first; interrupted2 loads on the other
four cards only after the first group has a real initial native checkpoint.
This is one recovery comparison, not two new scientific training arms.

The v1 cold-import and v2 SID-topology failures remain preserved. v2 is charged
as one real trainer attempt even though there were zero optimizer or score calls.
Only the affected import/lifecycle contracts were repaired and reviewed; the
original training runtime, final evaluation controller, mathematical update,
32 TRAIN scenes,4096 schedule and four-rank batch remain unchanged.

Main formal consequence training remains live on zt5 GPU0–3. At20:45:39 it
had completed362/4096 updates and46,336 official calls, with finite gradients
and real FP32 master updates. Other projects' real training remains protected.
Additional hosts are available, but no unreviewed seed, obsolete IQE/MOPD job,
held feature branch or unsupported proxy optimizer was launched to occupy them.

DISPATCH.json is a historical start receipt. It is **not** a native replay result
or a planning-gain conclusion. Final actual SMOKE output must be reported
separately, including failures or A/A nonbitwise uncertainty without loosening
predeclared tolerances. Raw resource inventories remain in private project storage.
