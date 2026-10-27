"""Two explicitly authorized trainable-scope experiments, not recipe aliases."""
import copy
from pathlib import Path
import yaml

from .rl_runtime import expected_cleanx_recipe

MEMORY = 'ATTN_LORA_MEMORY'
TIME = 'TIME_MOD_LORA'
SCOPES = (MEMORY, TIME)
SCOPE_SCHEMA = 'world_postcot_cleanx_scope_RL_v1'
MEMORY_OBSERVATION = 'world_postcot_trainable_memory_observation_v1'


def time_paths(head):
    """Actual linear maps, including the final adaptive output normalization.

    These projections consume time+ego in this actor. They are not ordinary
    LayerNorm affine parameters and are not pure-time AdaRMS from pi0.5.
    """
    from torch import nn
    paths=['time_embedding.0','time_embedding.2']
    paths += [f'blocks.{i}.modulation.1' for i in range(len(head.blocks))]
    paths += ['output_modulation.1']
    for path in paths:
        if not isinstance(head.get_submodule(path),nn.Linear):
            raise ValueError('Expected an actual unwrapped time/modulation Linear: '+path)
    return paths


def expected_scope_recipe(scope):
    if scope not in SCOPES:raise ValueError('Unknown scope experiment')
    cfg=copy.deepcopy(expected_cleanx_recipe())
    cfg.update(adaptation=scope, reference_observation='permanently frozen original IL40k Qwen and memory',
        observation_gradient='frozen Qwen features; current memory rebuilt each inner' if scope==MEMORY else 'frozen Qwen and memory',
        memory_learning_rate=5e-6 if scope==MEMORY else None,
        cache_lifetime='one frozen Qwen feature tree per scene; no current-memory graph across inner' if scope==MEMORY else cfg['context_lifetime'])
    if scope==MEMORY:
        cfg['context_lifetime']='Only frozen Qwen features and IL reference memory persist; current memory and KV recomputed each inner'
    if scope==TIME:
        cfg['deployment_lora']='preserve exact time/modulation LoRA wrappers for BF16 parity; merging is not default'
        cfg['grpo']['lora'].update(r=32,alpha=32.,target_modules=['time_modulation_exact_paths'])
        cfg['time_module_mapping']=dict(shared_time_mlp=True,block_adaln_projection=True,
            final_adaln_projection=True,ego_mlp_trainable=False,
            modulation_input='time_embedding(sigma)+frozen_ego_embedding(ego)',
            paper_rank=32,alpha32='explicit engineering choice; paper Table11 states rank32',
            architecture_adaptation='AdaLayerNorm, fused time+ego; not pure-time AdaRMS')
    return cfg


def load_scope_recipe(path):
    cfg=yaml.safe_load(Path(path).read_text())
    if cfg!=expected_scope_recipe(cfg.get('adaptation')):
        raise ValueError('Scope recipe changed outside its complete registered contract')
    return cfg
