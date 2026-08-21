# Failure analysis at Gate 2

This report includes negative and conflicting cases from the completed phases.
Phase-6/7 failure modes are explicitly marked unavailable rather than inferred
from models that were not trained.

## Factual prediction is accurate while candidate risk is ignored

On 1,497 accepted candidates in 102 held-out scenes, the representative
scene-action consequence probe has 96 candidates meeting the configured unsafe
definition and predicts all 96 as safe. Its factual-anchor error is low
(three-seed mean 0.166), but its action shuffle gap is only 0.0003.

Highest-error false-safe examples include:

| Scene | Candidate | Failure label | Relevant true outcome |
|---|---|---|---|
| `16e446eab82b5d45` | `b7eaaff488a34762` | false-safe | DAC=0, NC=0.5, TTC event time=2.0 s |
| `16e446eab82b5d45` | `f2819e03f83d9de8` | false-safe | DAC=0, TTC event time=2.2 s |
| `191f94dc85fd5899` | `d26a2defff6d4435` | false-safe | dynamic collision, NC=0 |
| `c6e502d2e3845682` | `668ccbafd871394b` | false-safe | dynamic collision, NC=0 |
| `dcc3937e2e45545b` | `7d550514b786b8bd` | false-safe | DAC=0 |

The complete deterministic top-50 list is in
`action_collapse_artifacts/false_safe_examples.jsonl`.

## Structured future learns scene priors but not useful action dependence

The scene-action future tube improves the class-balanced factual objective over
the fit-only mean prior (0.708 versus 1.265), yet scene-only is slightly better
(0.699). Action shuffling changes map error by only -0.000046, and Effect
Alignment is 0.107. Moreover, unweighted map MAE is worse than the per-cell mean
prior (0.235 versus 0.196). This is a deliberately retained negative result:
sparse-risk weighting makes the target learnable under its training objective,
but it does not make the factual model action-grounded.

## Log-replay / IDM conflicts

Most evaluated pairs agree, but 75 pairs are low-confidence and are changed to
`ambiguous`. Several are precisely the safety-boundary cases that should not
receive strong AEE supervision:

| Scene | Candidate pair | Geometry distance | LR order | IDM order | Conflict |
|---|---|---:|---:|---:|---|
| `a5e8ec7df7c253e4` | `5660029b1b2f1c51` / `787d6f43a0986301` | 0.012 | 1 | 0 | hard relation disagrees |
| `2b5b074e74e350fb` | `8a4af7a495e9b25f` / `f98748666d1eb110` | 0.033 | 1 | 0 | hard relation disagrees |
| `014725c44c265d3e` | `03f9087234d7f877` / `dc446051c508fd84` | 0.058 | -1 | 0 | hard relation disagrees |
| `c4b99ac30f3d56e6` | `4c53ce7dca029af0` / `8c0c78a1c1d1659e` | 0.096 | -1 | 0 | hard relation disagrees |

These examples explain why identifiability confidence is retained even though
aggregate agreement is high. They are pseudo-counterfactual traffic-assumption
conflicts, not observations of the true response to an unexecuted action.

## Global separation and AEE cases

Not available: global separation and AEE are Phase 6 and were not implemented
or trained before the Gate-2 stopping point. Therefore this delivery cannot
show either global over-separation or an AEE success case. The 2,095 cached
safety-boundary pairs are the fixed evaluation set for that next comparison.

## Better world metric but worse planning

Not available: no world loss was connected to Qwen+DiT and no planning pilot was
run. The baseline action path and official evaluators remain untouched, so no
planning transfer or world/action gradient-conflict claim is made.
