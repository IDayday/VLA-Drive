# Implemented local interaction algorithm

The formal algorithm begins with public Qwen3-VL-2B-Instruct and a freshly initialized original Flow-Matching DiT. All driving-specific weights are trained within this campaign and the recipe is recorded. It does not load the earlier private VLA foundation. This finite7284scene campaign is not equivalent to the larger DriveDreamer-Policy training recipe.

## Current observations and shared foundation

The allowed input is currentF0/L0/R0, calibrated after the actual crop/resize, navigation and allowed ego state/history. Training labels live in separate records. Deployment uses no future labels, map, LiDAR, logged neighbor speed or track IDs.

`tools/local_interaction_mask_v2/foundation.py` constructs the public baseline and shared current-world reader. The reader has64scene and64agent queries, internal dimension256; its input is the allowed current Qwen visual features. The original image/text/history/action prefix remains intact. Continuous scene/agent embeddings are appended **after the native action-token prefix** under the original causal attention. Current class/box heads read the post-Qwen agent states. The append-tail design differs from the earlier pre-action world-token proposal: the later local planning adapter carries world information to action conditions explicitly.

Foundation training updates fresh originalDiT/history/Reader/current heads, common rank8Q/V language-LoRA and driving-token embeddings. Public vision and original Qwen tensors stay frozen; the unused legacy motion head is frozen. A frozen Qwen operation still carries gradients from current-world losses into trainable Reader inputs and language adapters. Immutable public visual features can be cached because their encoder is frozen. The final24pass foundation is shared by all four planning controls, includingA0; A0 is therefore not the paper's action-only training ablation.

The selected current-head refinement subsequently updates only the existing current head. Its input features and upstream weights are frozen, and future labels are erased. Formal inference uses QwenbaseBF16, small language-LoRA arithmeticFP32 and originalDiTFP32. This explicit numerical repair is recorded in cache identity and parity reports.

## Predicted local graph

`observability.py` measures projected-box legality, area, pixel height, truncation, support views and geometry. These are uncalibrated evidence proxies. No independent2D detector or occlusion model was supplied for formal experiments; a plausible projection does not prove a real detected object.

`local_graph.py` separates existence, observational reliability and current relevance. The fixed selector is `configs/local_interaction_mask_v2/selector_v1.json`. Current navigation/speed define a conservative corridor; current predicted headings and unknown-speed envelopes define following, merging and crossing candidates. Neighbor speeds remain explicitly unknown. No frozen-plan proposal or GT future is used to select nodes.

| Group | Role |
|---|---|
| A | Reliable predicted vehicle/pedestrian/bicycle hypotheses selected by direct relation or bounded second hop. Eligible for graph trajectory tasks. |
| B | Plausible relevant but unreliable, static/unknown or capacity-excluded hypotheses. Their current features remain risk context; they are never declared confirmed free space or given reliable future conditions. |
| C | Weakly related hypotheses outside the explicit local trajectory graph. The original image/scene path remains. |
| D | Invalid, very low-existence, duplicate or padding hypotheses. Excluded from explicit actor attention, graph loss, graph pooling and graph planner memory. |

The relation model retains at most8direct and16total non-ego nodes, without requiring full capacity. Eligible second-hop nodes must relate to a retained direct neighbor. Selected neighbor-neighbor edges use current following/crossing rules. `edge_mask[i,j]=True` means queryi can read keyj. Confirmed geometric relations communicate both ways; directional features are hypotheses, not causal directions. Self-edges make valid rows safe; inactive rows and values are cleared before computation. Shared visual context remains available, so graph disconnection is not a claim that the VLM cannot see an outside object.

The nearest control uses the same observational eligibility and16node upper limit, selects nearest eligible actors in the fixed range and uses nearest relations. Its actual node/target populations can differ; scientific comparison therefore reports both full coverage and common matched track cohorts.

## Label association and masked trajectory flow

`local_targets.py` performs the original full64slot Hungarian assignment once, then accepts only same-class associations with current center distance below2m. It maps originalslot→localslot→GTtrack and reads that track's future points in ego(t0). Rejected associations remain unsupervised; there is no second forced nearest assignment. Future-valid bits do not change deployed nodes, relations or mask draws.

`local_masks.py` independently samples50%all-hidden,25%ego-hidden and25%one eligible neighbor-hidden. A missing eligible neighbor falls back to all-hidden. A selected neighbor without accepted future labels produces an explicitly counted empty task; it is not resampled. Active, predictable, hidden, condition-valid and label-valid masks remain distinct.

The graph flow uses dimension128,4heads,2layers and8futurexy points at0.5s intervals, with10sampling integration steps. It predicts current-centered residuals with20m ego and5m neighbor scales. Ego and neighbor regression sums have separate valid-coordinate denominators. Effective supervised coordinates differ betweenALL andMASK despite identical presentations. The conditional sampler clears unknown values before encoding and clamps known points at initialization and every integration step. Main deployment always hides all future ground truth.

Graph/task/ego-FM RNG streams are separate. A stable per-scene/sample/seed full source-slot noise bank is gathered by original identities, so local packing and batch order do not silently replace actor noise. The protocol is versioned; historical scores with a different noise rule do not enter the new main table.

## Original DiT planning transfer

`local_planner.py::LocalPlanningBridge` implements equal17,315,969trainable-parameter CURRENT/ALL/MASK bridges with the same initialization. CURRENT projects current actor features. ALL/MASK add their own frozen all-hidden graph hidden states to the same projected current features. Every arm also receives the same current scene/B-risk context. A projection followed by gated cross-attention adds this memory to the original action conditions. The full819,503,620parameter original action head stays frozen, and its10step sampler produces the executed ego trajectory.

The graph's ego prediction is saved separately. It never replaces the executed DiT trajectory or becomes a GT-conditioned planner input. No new scorer, RL, BEV provider, teacher-current-box graph or future-image generator is part of this main algorithm. Bbox supervision establishes current perception; future graph loss establishes a trajectory-prediction task, not identified counterfactual dynamics.

P2 trains only each949,450parameter graph. P3 freezes Qwen, LoRA, driving tokens, Reader/current heads, originalDiT and graph, training only the equal bridges. These frozen stages permit current-feature caching. Cache identity binds the foundation, current head, source, language numerics, selector, graph/noise rules, observations and calibration. Online prediction uses the same payload packing and bridge functions; all three final trained checkpoints must pass the real online/cache check before evaluation.

## Evidence boundaries

The full training audit retains only16.71%of the supported/relevant motion audit targets; expected valid neighbor-task supervision is10.82%. False-positive and unmatched dense graphs are retained in the audit. Formal results must expose this limitation and cannot equate low localADE with broad actor prediction quality.

The registered training-holdout schedule determines graph16→32and planner8→16extensions. All four models are then locked before any development/NavtestPDMS. GPU trajectory export and official CPU scoring run asynchronously with complete scene denominators. Related-vs-equalweak future-condition removal is a privileged diagnostic and may be out of distribution; sensitivity does not prove causation. Final scientific conclusions require the full Navtest paired results and log-cluster uncertainty, separately from the20engineering checks in `ENGINEERING_EVIDENCE.md`.
