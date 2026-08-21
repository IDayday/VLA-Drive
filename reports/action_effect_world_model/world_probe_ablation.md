# World-probe ablation status

Status: **not implemented and not run in this Gate-2 delivery**.

The completed work compares factual-only consequence and structured-future
probes against scene-only, trajectory-only, shuffled-action, and exact
same-parameter no-action controls. Multi-candidate absolute supervision,
global action separation, AEE, and confidence-weighted AEE belong to Phase 6.
They are intentionally deferred because the requested first execution stops at
Gate 2; implementing them earlier would weaken the required phase attribution.

The next authorized experiment should keep the completed data split and probe
architecture fixed and vary only the loss/supervision source in this order:

1. factual-only;
2. multi-candidate absolute supervision;
3. global separation;
4. unweighted AEE;
5. identifiability-weighted AEE.

No result for these methods is claimed here.
