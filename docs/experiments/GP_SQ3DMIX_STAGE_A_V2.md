# GP-SQ3D-Mix Stage-A-v2

Status: **completed — NO-GO**

Branch: `feature/gp-sq-3d-mix-stage-a-v2`

Evaluated code commit: `cb786e4a243a848beba8c7b75920e166c571c8ab`

Final decision: neither `projected_residual` nor `gated_residual` passed the immutable Stage-A-v2 gates. `selected_variant` is therefore `null`, `scene_conditioning_useful=false`, and Stage B was not started.

## What changed

The GP-SQ3D-Mix architecture contract was preserved. Stage-A-v2 changed the causal experiment around it:

- fixed, global, same-command hard donors with moderate action distance and geometry-far selection;
- deterministic within-view spatial feature derangement with no fixed points;
- paired `FlowMatchingState` and paired dropout streams for real/hard/spatial losses;
- ratio-of-means paired bootstrap statistics;
- matched `projected_residual` and `gated_residual` variants;
- fail-closed utility, causal-gap, residual-distribution, gradient, alpha, and retention gates.

The implementation also added sharded/resumable slot statistics, fixed negative-map manifests, dataset-side donor loading, single-PPU gradient accumulation, conditional Stage B/C launchers, and regression tests. Existing GP/SQ3DMix frameworks, checkpoints, reports, and results were not rewritten.

## Phase 0: implementation and verification

- `compileall`: passed.
- GP/Stage-A-v2 tests: 42 passed.
- legacy SQ3DMix and action-only regressions: 55 passed.
- targeted evaluator/launcher regressions after runtime fixes: 7 passed.
- shell syntax, launcher dry-runs, `git diff --check`: passed.
- single-PPU topology used for smoke and formal Stage A: `1 × batch 2 × accumulation 16 = global batch 32`.
- exact action-only warm start SHA256: `f8b8ee2ee9f3161c07f0123c70d1ee11ac45044715b5588f50b6e86cc340b9aa`.

Two runtime defects found only on the real PPU path were corrected before the reported experiment: ZeRO-safe gradient capture through parameter backward hooks, and evaluator-side Accelerate `PartialState` initialization. The final data/evaluation binding is commit `cb786e4a243a848beba8c7b75920e166c571c8ab`.

## Phase 1: full statistics and fixed negatives

The complete training cache was processed on CPU in 16 resumable shards. The measured filesystem wall span from stats-contract creation through the merged manifest was approximately `04:04:35` (2026-08-24 18:26:23 to 22:30:57, UTC+8).

| Asset | Value |
|---|---:|
| Samples | 103,288 |
| Pooling | 3 views × 6 × 10 = 180 slots |
| Descriptor projection | 2048 → 128, seed 20260824 |
| Source datalist SHA256 | `37ea8b153d788096f3fa1b6d852003f9b1c6fd5df42b51b9e7a670401958e0e5` |
| Dense-cache manifest SHA256 | `f5c6e95bfb5591d26a1aca022698214d12aad54aafa8c13b9da2a483f40d8ee3` |
| Slot-stats SHA256 | `8e7658980036ea912fd80113976f9d7a11634d0498de74464732b4d352d46bae` |
| Descriptor file SHA256 | `3022894a0b147c279c4b42733bcd0082f24852109face9170698c149cf98b2d9` |
| Projection SHA256 | `9a501439ed09935e059111667aaf4b986e3ce7eb158a5d688f7a71dd614d1444` |
| Token-order SHA256 | `e1e2cda898650673702cfa3a0d99226489efbda4a11a9e30f42db6c714f52d28` |

Resume was exercised after completion and skipped every completed shard without recounting samples.

All constructed hard-negative maps had `fallback_rate=0`, `self_donor_count=0`, and `same_log_violation_count=0`:

| Map | Samples | SHA256 | Max observed reuse | Binding commit |
|---|---:|---|---:|---|
| smoke train | 256 | `04b8a9d97b37bb25ec7aea2e60e1ec107ae17ef266a23109a62aab7d5777f6eb` | 2 | `cb786e4` |
| smoke selection | 128 | `4f4b202b33bf87c4e1ca55c60b2a5200f4c105b9a42f979f6bbf1bdaf841b77e` | 2 | `cb786e4` |
| Stage-A-v2 train | 8,000 | `5fa84eea6230c8708570ab71f3e63d41333c6454c3a13156b5611542e030a8fd` | 16 | `cb786e4` |
| Stage-A-v2 model selection | 1,000 | `86f341f0c1dc1de41baf9fbec842cf6fe361ab3efd70c82c3ba67e911c720ad6` | 5 | `cb786e4` |
| Stage-A-v2 final gate | 1,000 | `14855bbeadad7cbd9f41754cc87286dd57b0ab5826930591a9e3324d90f9a51d` | 5 | `cb786e4` |
| full train | 103,288 | `9c7412943289eafdd32c0abd3c2a26c2ac0d3c2debb7f48716cf15c8424c8fb2` | 16 | `c12d8ab` |
| fixed navtest-2k | 2,000 | `a10c4d7656baf0c257b913743312eb79de7f53b31a55821c5e89ddc1a6b981ec` | 9 | `c12d8ab` |

The three Stage-A-v2 splits are pairwise disjoint: train/model-selection/final-gate sizes are 8,000/1,000/1,000 and their token-list SHA256 values are `604771…3efe`, `4e046c…49ce`, and `c32ab9…ce9b` respectively.

## Phase 2: matched smoke

Both 100-step smoke variants passed every implementation invariant before formal Stage A started.

| Check | projected | gated |
|---|---:|---:|
| Strict checkpoint reload | pass | pass |
| Step-1 reader/up-projection gradient | `1.9204e-4` | `1.4124e-4` |
| Step-10 adapter gradient | `1.4363e-5` | `1.1107e-5` |
| Step-10 reader gradient | `1.5880e-4` | `1.3169e-4` |
| Step-10 gate gradient | N/A | `2.2183e-6` |
| Slot-mean identity max abs | 0 | 0 |
| Spatial fixed points | 0 | 0 |
| Shared flow/dropout condition count | 4 / 4 | 4 / 4 |

## Phase 3: formal Stage-A-v2

Each variant trained for 2,000 optimizer steps on the 8,000-sample split, with warmup 100, save interval 250, one PPU, per-device batch 2, accumulation 16, and effective global batch 32. Qwen, Action DiT, and `action_input_model` remained frozen. Observed training wall times were approximately 7 h 24 min for projected and 8 h 38 min for gated.

All eight checkpoints per variant were evaluated on the same 1,000-sample model-selection split. The immutable selection rule chose projected step 1500 and gated step 2000; both chosen checkpoints were then evaluated exactly once on the independent 1,000-sample final-gate split with 10,000 paired bootstrap draws.

### Final-gate statistics

| Metric | Gate | projected step 1500 | gated step 2000 |
|---|---:|---:|---:|
| Mean base loss | — | 0.00592213 | 0.00592213 |
| Mean real loss | — | 0.00480655 | 0.00484359 |
| Relative real−base | ≤ 0.5% | −18.8375% | −18.2121% |
| Utility 95% CI | upper ≤ 2% | [−30.1344%, −11.1777%] | [−29.0954%, −10.7604%] |
| Relative hard−real gap | > 5%, CI lower > 0 | +0.3652% [−0.4880%, +1.3821%] | +0.2266% [−0.1366%, +0.7520%] |
| Absolute hard−real gap | CI lower > 0 | `1.7552e-5` [`−2.0050e-5`, `5.4826e-5`] | `1.0976e-5` [`−6.4011e-6`, `2.8886e-5`] |
| Relative spatial−real gap | > 2%, CI lower > 0 | −0.0476% [−0.1773%, +0.0583%] | −0.0216% [−0.0938%, +0.0341%] |
| Absolute spatial−real gap | CI lower > 0 | `−2.2901e-6` [`−7.3749e-6`, `2.3822e-6`] | `−1.0475e-6` [`−3.5457e-6`, `1.5255e-6`] |
| Residual/action mean | [0.01, 0.15] | 0.3355 | 0.3698 |
| Residual/action p95 | ≤ 0.25 | 0.4167 | 0.4561 |
| Residual/action p99 | ≤ 0.40 | 0.4272 | 0.4682 |
| Residual/action max | ≤ 0.50 | 0.4320 | 0.4782 |
| Max per-horizon mean | ≤ 0.20 | 0.3831 | 0.4222 |
| Alpha | [0.05, 0.20] | 0.10014 | 0.10014 |
| Slot-mean identity max abs | < 1e-6 | 0 | 0 |
| All passed | true | **false** | **false** |

### Complete immutable gate result

| Gate | projected | gated |
|---|---:|---:|
| 1. slot-mean identity | pass | pass |
| 2. utility point estimate | pass | pass |
| 3. utility non-inferiority CI | pass | pass |
| 4. hard relative causal gap | **fail** | **fail** |
| 5. hard absolute causal gap | **fail** | **fail** |
| 6. spatial relative causal gap | **fail** | **fail** |
| 7. spatial absolute causal gap | **fail** | **fail** |
| 8. residual mean | **fail** | **fail** |
| 9. residual p95 | **fail** | **fail** |
| 10. residual p99 | **fail** | **fail** |
| 11. residual max | pass | pass |
| 12. per-horizon residual bound | **fail** | **fail** |
| 13. geometry route active | pass | pass |
| 14. named losses finite | pass | pass |
| 15. alpha bound | pass | pass |
| 16. retention non-collapse | N/A | pass |

Projected adapter/reader mean gradient norms were `3.2057e-3` / `5.4393e-3`. Gated adapter/gate/reader mean gradient norms were `2.2899e-3` / `8.4002e-4` / `5.4614e-3`. The failure is therefore not a disconnected geometry route.

For gated, retention mean/std/min/max were `0.11395 / 0.03142 / 0.06921 / 0.24160`; both lower- and upper-bound saturation fractions were zero. Scene-shuffling changed loss by only `2.1628e-7`, with paired 95% CI [`−2.9393e-7`, `7.6382e-7`]. Although it changed retention L2 by `0.9869` and residual L2 by `0.05698`, it did not measurably worsen flow loss. Consequently `scene_conditioning_useful=false`.

## Interpretation and decision

The new utility path can lower flow loss relative to the frozen action-only baseline, but it does not use correct geometry in the required causal sense:

1. hard counterfactual geometry is statistically indistinguishable from correct geometry;
2. destroying feature-to-UV/ray correspondence is also statistically indistinguishable, and its point estimate is slightly better than real on the final split;
3. the learned residual grows far outside the intended mean/p95/per-horizon range;
4. the scene gate changes internal retention but has no positive paired loss effect and is worse than projected on final real loss, hard gap, and residual scale.

No variant is selected. If a diagnostic-only follow-up is pursued later, `projected_residual` is the simpler and less harmful ablation, but it is **not** approved for Stage B or long training.

## Fail-closed outcome

- Stage A-v2: completed, NO-GO.
- Stage B two-seed 10k pilot: not run; blocked by Stage A.
- Stage B full navtest: not run.
- Stage C formal 30k: not permitted and not run.
- 100k extension: not permitted and not run.
- `formal_30k_allowed=false`.
- `formal_100k_allowed=false`.

Machine-local paired sample CSVs and full JSON reports remain under:

`/mnt/zhangt_workspace/project/DriveDreamer-Policy/navsim_eval/gp_sq3dmix_stage_a_v2_cb786_exact_warmstart_eval/stage_a_v2/`
