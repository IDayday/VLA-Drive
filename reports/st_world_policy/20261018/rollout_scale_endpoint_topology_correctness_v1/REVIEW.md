# Independent endpoint topology review

**PASS_STATIC_TOPOLOGY_AMENDMENT; WAITING_CHECKPOINT_AND_FINAL_RESOURCE_BINDING.** The proposed four independent exporter ranks are compatible with the fixed single-trajectory protocol. This is a static correctness conclusion, not evidence of actual GPU alignment, endpoint completion, efficiency, or permission to dispatch.

The frozen runtime is `751fcfa2b41549ae16f7b3204aef8dd18b912016`; the proposed plan is integrated at `611127df`. Its exporter, scorer and analyzer are byte-identical to the original6714 control versions. Model and DEV per-scene result assets were not opened.

## What changes and what does not

Only the execution partition changes: `rows[rank::4]` gives four disjoint424-scene assignments, totaling the original1696. Each call remains batch1 and one candidate. `scene_noise(token,42)` uses a private CPU generator and produces FP32 noise independent of rank, order and global RNG; the actual CPU function was tested. No rank enters the current observation, original attention/mask path, planner condition, or ten-step FP32 master action solver. Four ranks are not four deployment candidates.

The export identity includes world size, so the new world4 namespace is distinct from historical world1. The mathematical runtime request identity excludes execution shard count while retaining environment, seed, solver, precision and noise contract. The paired analyzer permits this distinction but still checks full observations/environment and scene-wise canonical context/reference; no old receipt is rewritten. Actual paired eligibility remains contingent on the completed new endpoint passing those checks.

## Independent tests

16 tests passed (1.28s pytest;1.98s wall;1.86s CPU;454848KiB maxRSS). Exact-source helper execution covered1696 and uneven/tiny populations; private noise under reorder/shard/RNG changes; concurrent identical immutable writes and conflicting writes; missing/wrong/duplicate shard records and receipt hashes; single durable ledger re-entry, cached-result reuse, hard total budget and interrupted unknown admissions. All fixtures are explicitly synthetic and do not replace real inference evidence.

Shared IDENTITY/OBSERVATIONS use no-replace atomic publication. Prediction writes are disjoint under per-rank locks. All four completion receipts are required before scoring, with exact token populations and per-prediction receipt hashes. The single coordinator's EVALUATION lock protects one ledger;1728 is the total admitted-call cap across workers and restarts. Unknown interrupted ADMITTED records remain charged. CPU worker completion order is restored to request order.

## Remaining execution boundaries

The new4096 checkpoint does not yet exist in this review; its tag and native receipt remain null. Before launch, bind that exact formal endpoint, all four resource leases/environment identities, shared output root and final analysis plan. Do not substitute diagnostic/early weights, switch world size mid-export, reset the ledger, or replay historical C1 merely to match topology. The fixed protocol still requires1696 valid scores and no silently dropped scenes.

Identical source, private noise and batch1 establish a sound topology change; they do not prove actual cross-device floating-point equality or fourfold acceleration. Those are deliberately not claimed. No training, model loading, GPU execution, official scoring, DEV score reading or dispatch was done.
