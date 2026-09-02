# Formal PlanReg-WM training

Both BaseInit and Driving-VQA Init train the same model for 27 full exposures
of 103,288 trainval examples. Internal validation is disabled, Navtest never
selects an epoch, and epoch 27 is the fixed report checkpoint. Recovery-only
checkpoints are saved at epochs 5, 10, 15, 20, 25, and 27 plus `last.ckpt`.

## Student path

The model is InternVL3-2B with one front camera, 16 read-only planning
registers inside every InternViT block, thumbnail-query tile aggregation, and
fresh rank-32 Q/V LoRA in all 24 vision blocks. The 48 Q/V adapters contain 96
A/B linear layers and 3,145,728 trainable parameters. K and every base
InternViT/LLM parameter remain frozen.

The frozen semantic path detaches patch features before the LLM and evaluates
the LLM under `no_grad`; a trainable Q-Former maps LLM hidden states to 16
semantic tokens. A one-layer, eight-head cross-attention uses normalized
planning tokens as Q and semantic tokens only as K/V. The output is
`LN(planning + sigmoid(gate) * semantic_context)`, with initial gate
probability 0.20. The generator and scorer consume the same `[B,16,256]`
tokens.

The generator retains 64 queries, eight poses, and four refinement stages.
The scorer remains the exact independent four-layer DrivoR adaptation pinned
to `valeoai/DrivoR@fc6e5aa144bbcb5a046e22c18f1bd5cf3af8634a`: detached
`[8,3]` proposals are flattened/re-embedded, six independent heads are used,
TTC target 2 is masked, and log-space aggregation weights are 1/1/0/5/5/2.
Scorer loss reaches scene/register features but never proposal coordinates.

## Online world model

World-model loss is enabled in both formal versions from optimizer step zero.
It predicts normalized future register states at 0.5, 1.5, and 3.0 seconds
from current registers and K=1 GT trajectory conditioning. Horizon weights are
1.0/0.7/0.4; absolute cosine and delta SmoothL1 weights are 1.0 and 0.5.
The global WM coefficient starts at 0.01 and ramps to 0.10 over the first 10%
of optimizer steps. Invalid horizons are omitted from the weighted denominator.

EMA reference momenta are 0.996 to 0.9999 at global batch 16, with cosine
scheduling. At another global batch, each endpoint is
`m_actual = m_reference ** (actual_global_batch / 16)`. EMA is initialized only
after VLM and shared trainable initialization restore, updated after optimizer
steps, and never used at inference.

## Optimizer

The AdamW peak LRs below are defined at global batch 32 and scale by
`sqrt(actual_global_batch/32)` subject to caps:

| Logical group | GB32 peak LR | Cap |
|---|---:|---:|
| planning adapter | 2e-4 | 3e-4 |
| semantic fusion | 2e-4 | 3e-4 |
| action generator | 2e-4 | 3e-4 |
| scorer | 2e-4 | 3e-4 |
| future predictor | 2e-4 | 3e-4 |
| semantic Q-Former | 1e-4 | 1.5e-4 |
| vision Q/V LoRA | 4e-5 | 5e-5 |

Matrix weights use weight decay 0.01. Biases, normalization parameters,
register/query/embedding tensors, semantic/tile gates, and every LoRA A/B use
zero decay. AdamW uses betas 0.9/0.999 and epsilon 1e-8. The step scheduler is
5% linear warmup from 1% of peak followed by cosine decay to 10% of peak.
Gradient clipping is global norm 1.0; accumulation is one.

Deployment export strips EMA, predictor, optimizer, scheduler, and callback
state, then requires an exact topology load without legacy LoRA folding or
scaling. Evaluation uses BF16 VLM plus FP32 action/scorer, not “full FP32”, and
uses only the current frame.

Multi-trajectory consequence modeling, candidate-specific future targets, RGB
prediction, ranking loss, and trajectory self-refinement are not implemented.
