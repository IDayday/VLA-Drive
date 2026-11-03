"""Prototype protocol, never an amendment to the live policy-only contract."""
from starVLA.rl.qwen_world_compare.contracts import fingerprint

EXPERIMENT='D4_FRESH_AUX'
SCHEMA='d46_current_only_fresh_R16_reward_readout_v1'
PROTOCOL=dict(schema=SCHEMA,experiment_id=EXPERIMENT,algorithm_base='D4',
    parent='strict complete verified IL46 deployment policy, never original auxiliary heads',
    initialization='private CPU default generator42, restored after construction; no CUDA RNG use',
    initializer_draw_order=['R16 Normal(0,.02)','LayerNorm+Linear bridge','EvaluationReadout'],
    new_modules=['decision.r_embeddings','readouts.bridge','readouts.evaluation'],
    missing_IL_modules_recovered=False,reference='pure original pruned IL46, copied before fresh modules',
    kept='every existing C/P/action/FiLM/role tensor unchanged; zero RL LoRA',
    trainable='registered textLoRA + fresh R16 + bridge/evaluation only',
    R_learning_rate=2e-5,head_learning_rate=2e-5,
    inactive='unused parameters have gradNone and no Adamstate; never add modules after DDP construction',
    objective='unchanged native joint24 GaussianSafetyEP PPO/adaptiveKL plus original D4 post_feedback.loss',
    labels='existing scored TRAIN candidates factors/score/pairs/native event classes only',
    deployment='original single B1 K10shift5 A-only ODE; prune all fresh R/readouts',
    deployment_payload='strict separate policy closure: discard BOTH action_expert.decision.r_embeddings AND action_expert.readouts.*; legacy D44 exporter is not sufficient',
    formal_admission='prototype only, requires independently registered fresh-aux route and actual fullD46 GPU qualification',
    D6='initializer/scope reusable later; D6_FRESH_AUX not implemented or qualified here')
PROTOCOL_SHA256=fingerprint(PROTOCOL)
