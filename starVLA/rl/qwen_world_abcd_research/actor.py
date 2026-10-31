"""Architecture LoRA with independent reference modules and shared frozen bases."""
import copy

from starVLA.rl.qwen_world_compare.policy_adapter import FourFrameworkRecipeActor
from starVLA.rl.qwen_world_compare.rl_scope import install_lora


class ResearchActor(FourFrameworkRecipeActor):
    def __init__(self, model, cfg, conditions=None):
        backend = copy.deepcopy(cfg)
        if model.architecture_id == 'D' and backend['grpo']['sample']['randn_clip'] is None:
            # The inherited sampler stores a float bound and calls clamp. An
            # infinite internal bound is exactly the identity, not a truncated
            # Gaussian; semantic/config/checkpoint identity remains JSON null.
            backend['grpo']['sample']['randn_clip'] = float('inf')
        super().__init__(model, backend, conditions=conditions)

    def _configure_action_grpo(self, grpo_cfg):
        self.eval().requires_grad_(False)
        # Separate module graphs permit current-only LoRA installation. Frozen
        # parameter storage is immutable and can be shared without a
        # second 2B Qwen allocation. No adapter or trainable tensor is shared.
        memo = {id(p): p for p in self.action_expert.parameters()}
        self.ref_action_expert = copy.deepcopy(self.action_expert, memo).eval().requires_grad_(False)
        if (grpo_cfg.get('finetune') or 'lora') != 'lora':
            raise ValueError('ABCD research uses explicit attention LoRA')
        lora = dict(grpo_cfg.get('lora', {}))
        if not lora.get('enabled', True):
            raise ValueError('LoRA must remain enabled')
        self.action_finetune_mode, self.action_lora_enabled = 'lora', True
        self.lora_r, self.lora_alpha = int(lora.get('r', 16)), float(lora.get('alpha', 32))
        self.lora_dropout = float(lora.get('dropout', 0))
        self.lora_target_modules = self._resolve_action_lora_target_modules(list(lora.get('target_modules', ['q','k','v','o'])))
        self.scope_manifest = install_lora(self.action_expert, paths=self.lora_target_modules,
            r=self.lora_r, alpha=self.lora_alpha, dropout=self.lora_dropout)
