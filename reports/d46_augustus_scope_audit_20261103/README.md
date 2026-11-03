# AUGUstus scope audit: dense trunk versus LoRA

Read-only audit, 2026-11-03. No actor instantiation, GPU, scoring, optimizer,
registration or live-source changes. [Machine-readable facts](SCOPE_AUDIT.json)
and [pinned source hashes](SOURCE_RECEIPTS.json) bind upstream EDF3422 and local
SCI82. Recommendations are proposals, not running experiments or score gains.

The official pinned default uses **dense text/action training, with the visual
tower frozen during RL**: inherited `lora_rank=0`, SFT `freeze_llm=false`, the RL
environment defaults `TRANSFUSION_FREEZE_VIT=true`, and the FSDP optimizer takes
the built module's parameters. Custom original model configuration/environment
can override this; this is not a claim about an unseen author launch. Sources:
[official SFT defaults](https://github.com/haha-yuki-haha/AUGUstus/blob/edf3422304e4b2987e839a9212cf1176782e7380/AUGUstus_SFT/configs/navsim_qwen3p5.yaml#L34),
[LoRA default](https://github.com/haha-yuki-haha/AUGUstus/blob/edf3422304e4b2987e839a9212cf1176782e7380/AUGUstus_GRPO/verl/trainer/config/model/hf_model.yaml#L43),
[construction/freezing](https://github.com/haha-yuki-haha/AUGUstus/blob/edf3422304e4b2987e839a9212cf1176782e7380/AUGUstus_GRPO/recipe/transfusion/transfusion_policy.py#L202),
[optimizer](https://github.com/haha-yuki-haha/AUGUstus/blob/edf3422304e4b2987e839a9212cf1176782e7380/AUGUstus_GRPO/verl/workers/engine/fsdp/transformer_impl.py#L451).

| Function | Official Qwen3.5 model path | Existing local D equivalent |
|---|---|---|
| Shared text blocks | `qwen_model.model.language_model.layers.*` | `action_expert.decision.qwen_model.language_model.layers.*` |
| Text final norm | `qwen_model.model.language_model.norm.*` | `action_expert.decision.qwen_model.language_model.norm.*` |
| Input/output action interface | `action_encoder.*`, `action_decoder.*` | `action_expert.decision.action_input.*`, `action_expert.decision.action_output.*` |
| Time/position | `time_embed.*`, `traj_pos_embed.*`, native AdaRMSNorm | `action_expert.decision.time_embedding.*`, `time_films.*`, `action_position` |
| Extra local current interfaces | No claimed one-to-one correspondence | `p_embeddings`, `role_embedding.weight`, `ego_input.*` |
| Vision | `qwen_model.model.visual.*` | `action_expert.decision.qwen_model.visual.*` |

For **AUG_ACTION_FULL**, recommend all 28 local text blocks including dense
attention, FFN and norms, final norm and the existing action/time/current
interfaces. For **AUG_ACTION_LORA**, replace only the text-core scope with the
existing attention q/k/v/o LoRA and keep the same action-interface decision in
both arms. Current D0 trains 6,422,528 attention-LoRA parameters; it is not full
text training. Actual full unique parameter count remains unmeasured here.

Two integration boundaries are essential:

- The existing provider caches pre-language `encoded.inputs_embeds`. Keep token
  embeddings frozen in the minimal adaptation, or explicitly reconstruct their
  current gradient path from raw token IDs every inner. Do not count detached
  cached embeddings as trained. State this adaptation boundary in the protocol.
- Existing D44 reference construction shares frozen base Parameter objects via
  a deepcopy memo. FULL must allocate independent current trainables and verify
  their identities are disjoint from the frozen reference; directly unfreezing
  the old shared base would corrupt the reference.

Current D0/FPO is not official JaT: local Gaussian logratio/KL uses **sum24**,
K10/shift5/W3 and independent initial candidate noise. Official code uses
coordinate means, configured **lambda_t**, 3 RL steps, and the paper specifies
shared group initial noise. Local FPO uses an energy-ratio PG proxy. These
differences require explicit new protocols, not a scope-only execution migration.
[Official ratio implementation](https://github.com/haha-yuki-haha/AUGUstus/blob/edf3422304e4b2987e839a9212cf1176782e7380/AUGUstus_GRPO/recipe/transfusion/algorithms/jit_grpo.py#L118),
[stage1 configuration](https://github.com/haha-yuki-haha/AUGUstus/blob/edf3422304e4b2987e839a9212cf1176782e7380/AUGUstus_GRPO/recipe/transfusion/config/stage1.yaml#L235).

The paper's unified action/CoT system and its action-only ablation differ. This
local proposal retains D's IL46 action policy and does not restore missing CoT
weights or claim the paper's full-system results. Both proposed arms start
independently from IL46 with new Adam0, with at most two formal trains; planned
32+16 resources and any future gains require actual qualification/deployment.
[Paper, algorithms and ablations](https://arxiv.org/html/2610.16985v1).
