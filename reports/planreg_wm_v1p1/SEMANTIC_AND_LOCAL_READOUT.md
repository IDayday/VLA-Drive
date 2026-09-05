# Semantic and local visual readout

## Source changes, not performance claims

New models use semantic-query std=0.02. Legacy mode defaults to 1e-6; loading or
resuming does not reinitialize queries or registers. Sixteen Q-Former output
tokens and planning-primary semantic cross-attention are unchanged.

Backbone returns `semantic_token_valid_mask` from the actual LLM attention mask.
The Q-Former inverts valid=True to `key_padding_mask=True` for padding. All valid
multimodal tokens remain visible. All-valid masks use the old no-mask path;
arbitrary appended padding or changed padding values cannot affect outputs.
The frozen semantic route still detaches patches before the no-grad LLM, while
the trainable Q-Former receives the resulting hidden states.

The actual tokenizer system prompt now describes **one current front-camera
image**, ego-position history and command, not eight camera/history images.
The tokenizer call and cached tokens use the same versioned prompt. Cache records
carry the actual system-prompt hash; V1.1 rejects missing/mismatched hashes.
This wording issue was not established as a PDMS root cause.

## `global_local_8_8`

| Output slots | Query | Keys | Values |
|---|---|---|---|
| 0..7 global | 8 learned queries | thumbnail registers | thumbnail visual registers |
| 8..15 local | 8 learned queries + global-summary conditioning | all crop×register vectors + normalized tile-position MLP | pure visual crop×register vectors |

Each group has its own LayerNorm; concatenate to `[B,16,256]`. There is no extra
scene self-attention by default, no zero-initialized total crop gate, no extra
camera, and no increase in scorer memory length. Registers stay inside InternViT;
only patches enter pixel shuffle/MLP/LLM. Teacher and student have the same readout
topology and the FP32 EMA mapping includes its trained parameters.

`thumbnail_query_attention` is retained for replay. New readout output parity with
it is neither required nor claimed: this is a design change. Future target slots
use the same 8+8 grouping but are **not tracked object identities**.

Tests cover shape, crop+metadata permutation invariance, geometric mismatching,
local-content response and scorer-to-local-visual gradients. Position/query
identity alone, with constant visual values, does not pass the visual-rank test.
There is no enforced token orthogonality or maximum-rank requirement.

## New real-model observations

The four-update, two-training-scene smoke recorded semantic slot-centered RMS
about 0.0402, energy effective rank about 10.64, cross-scene content RMS about
0.0472, and cross-scene slot-content RMS about 0.0048. These measurements subtract
each slot's across-scene mean; they are not merely query-ID variance. Fusion
attention was still nearly uniform (entropy about 2.7724).

These are limited initialization/numerical observations, not evidence of improved
driving. The 8+8 planning readout can still have low centered effective rank early
in training; do not relabel geometric separation as learned scene understanding.
See JSON artifacts for exact values and the longer update run.

No new full Navtest local-readout comparison was run. A task-level comparison must
use normally observed inputs and official selected PDMS, Oracle@64, regret and
catastrophic misselection—not only changed indices or more diffuse attention.
