# Qwen+DiT planning pilot status

Status: **intentionally not implemented and not run at the Gate-3 stopping
point**.

Qwen was used only once as a frozen, current-observation feature extractor from
the clean `action-effect-gate2-v0` snapshot. The cache records that no expert
action was passed into Qwen. DiT was not trained or modified, no world loss was
attached to shared layers, and the original action-only inference path,
flow-matching loss, checkpoint format, and NAVSIM evaluator semantics remain
unchanged.

The stop is evidence-driven: unweighted AEE failed the predeclared Gate-3
comparisons against multi-candidate absolute supervision and global separation.
Accordingly, this delivery does not create a Phase-7 launcher, does not run
planning continuation, does not claim world-to-planning transfer or gradient
conflict, and leaves PDMS/EPDMS and their components as N/A.

Any later planning experiment requires a new authorization and a new probe gate;
the current AEE objective must not be attached to Qwen+DiT as-is.
