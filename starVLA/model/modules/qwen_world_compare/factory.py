"""Immutable resolved recipes and independent generic-initialization modules."""
import copy
from pathlib import Path
import torch
import yaml
from .contracts import ARCHITECTURES,LOSS_WEIGHTS,SCHEMA
from .model import FourFrameworkVLA

def load_config(path):
    cfg=yaml.safe_load(Path(path).read_text())
    if cfg['schema']!=SCHEMA or cfg['architecture_id']not in ARCHITECTURES:raise ValueError('Wrong four-framework schema/architecture')
    if cfg['seed']!=42 or cfg['evaluation_seed']!=2101:raise ValueError('Fixed single seed protocol changed')
    il=cfg['il']
    if (il['max_updates']!=100000 or il['global_batch']!=32 or il['interface_updates']!=1000
        or il['fm_repeat']!=1 or il['post_scenes_per_update']!=8 or il['wa_queries']!=128
        or il['save_every']!=1000 or il['dev_every']!=5000 or il['loss_weights']!=LOSS_WEIGHTS):
        raise ValueError('The four complete common IL budgets/objectives changed')
    if cfg['candidate_refresh']is not False:raise ValueError('Per-arm candidate refresh is forbidden')
    if cfg['deployment']!=dict(prediction_type='x0',action_shape=[8,3],sampling='single_candidate_cleanx_ODE',steps=10,shift=5.,scorer='NAVSIM_v1.1_canonical_independent_reference_pair',seed=2101):raise ValueError('Deployment protocol changed')
    if cfg['execution']['micro_batch']<1 or 32%(8*cfg['execution']['micro_batch']):raise ValueError('Topology cannot preserve global32')
    return cfg

def build_model(cfg,*,checkpoint_config=None,train_heads=True):
    # Component seeds keep A/C's shared world and current scene weights equal
    # despite the extra world-KV projections in C. No driving weights loaded.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model=FourFrameworkVLA(cfg['architecture_id'],cfg['qwen_path'],config=cfg['model'],checkpoint_config=checkpoint_config,train_heads=train_heads)
        if cfg['architecture_id']!='D':
            from ..qwen_world_postcot.memory import SceneMemoryProvider
            torch.manual_seed(142);model.scene_provider=SceneMemoryProvider(**cfg['model']['scene'])
        if cfg['architecture_id']in('A','C'):
            from .world_stream import WorldStream
            torch.manual_seed(242);model.decision.world_stream=WorldStream(**cfg['model']['decision']['world_config'])
    return model
