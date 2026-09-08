# V2 gradient routing

| Loss / input | Visual Q/V LoRA | Internal registers / neck | Language LoRA / soft queries | Generator / shared head | Scorer | WM predictor | EMA |
|---|---|---|---|---|---|---|---|
| Four-stage physical/scaled trajectory loss | yes, rich memory and language patch path | yes | yes | yes | no | no | no |
| Exact scorer loss | yes | yes | yes | **no direct coordinate/decoder gradient** | yes | no | no |
| TF + differentiable RO | yes via online z0 and semantic context | yes via online z0 | yes through current-only semantic context | no | no | yes | no |
| Logged future motion/image/target | data only | teacher is detached | never an LLM input | never an input | never an input | conditioning / detached target | no-grad forward |

Frozen InternViT base, language FFN, embeddings, lm_head and frozen mlp1 do not update. Their differentiable operations are not wrapped in no_grad in the student language path. The student does not detach patches before the language model.

All trainable tensors are FP32 and belong exactly once to one AdamW group. Matrix WD is 0.01; bias/norm/query/register/gate/LoRA/embedding WD is zero. Q/V LoRA zero-B initialization legitimately gives zero first-step A gradients; the actual 32-step smoke checks parameter changes, not just finite gradients.

EMA parameters are frozen and permanently eval. Persistent masters are buffers protected against dtype casts. The successful-optimizer-step hook—not a batch-end callback—updates masters and forward copies. CPU GradScaler tests exercise an actual skipped optimizer step and accumulation.

`global_valid_mean` reports a globally normalized scalar and applies `world_size/global_count` to each local numerator's gradient, compensating DDP's average. Empty ranks participate in the same collectives and retain a zero-valued differentiable graph. This normalization applies to V2 regression/WM only; the fixed-source scorer loss is unchanged.

The low-frequency same-batch diagnostic uses `autograd.grad(..., retain_graph=True)`, never writes `optimizer.grad`, and compares planning and weighted WM on identical visual parameter sets. InternViT's remote-code reentrant checkpoint was adapted per encoder instance to non-reentrant checkpointing to support this diagnostic. Frozen dropout behavior and source block operation order are retained. No global monkey patch to PyTorch is used.

Teacher forcing and rollout share the exact predictor instance. Rollout intermediate states remain differentiable; future feature changes cannot change its outputs at fixed current inputs/actions. Missing real intermediate images invalidate TF's real-prefix dependencies but are not substituted into RO.
