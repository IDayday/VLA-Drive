"""Resolved, versioned recipes; prior fixed G8/r16 validators are unchanged."""
import copy
from pathlib import Path
import yaml
from starVLA.rl.qwen_world_compare.runtime import expected_recipe

ARMS=('D0','D0_CLIP020','D0_CANONICAL_CREDIT','D1','D2','D3_FPO','D3_AWM','D4','D5','D6')


def source_config(cfg):
    out=copy.deepcopy(expected_recipe('D'))
    out['grpo']['sample'].update(group_size=cfg['group'],randn_clip=float('inf'),final_action_clip=None)
    out['grpo']['lora'].update(r=cfg['text_rank'],alpha=cfg['text_rank']*2)
    return out


def defaults(arm):
    if arm not in ARMS:raise ValueError('Unknown D44 algorithm')
    return dict(schema='d44_native_rl_v1',arm=arm,parent='D_IL44K',seed=42,evaluation_seed=2101,
        global_scenes=32 if arm=='D4'else 64,group=16,inner_updates=2,
        learning_rate=3e-6 if arm=='D3_AWM'else 1e-6 if arm=='D4'else 2e-6,
        text_rank=32 if arm=='D5'else 16,film_last_layers=8 if arm in('D5','D6')else 0,
        train_action_io=arm=='D5',post_feedback=arm in('D4','D6'),
        clip=.2 if arm=='D0_CLIP020'else .01 if arm=='D3_FPO'else .1,
        beta=.001,target_kl=.015,entropy=0.,
        target_epochs=4 if arm in('D0','D0_CLIP020','D0_CANONICAL_CREDIT')else 12 if arm=='D6'else 6,
        maximum_epochs=20,reference_coefficient=1. if arm=='D3_AWM'else .001,
        rescue_start_epoch=.5,post_start_epoch=1. if arm=='D4'else 4.,
        positive_weight=.05,last_weight=.02,post_weight=.05,nominal_seed=2101,
        mc_samples=4,save_every=100,scoring_workers=2,condition_cache_scenes=8,
        warmup_epochs=.08,hold_until_epoch=.60,cosine_end_epoch=4.,min_lr_ratio=.05,
        data='/mnt/project/qwen-world-rl-optimization-20261030/data/NAVTRAIN',
        population_manifest='/mnt/project/qwen-world-rl-optimization-20261030/data/manifests/FULL_103288.json',
        parent_checkpoint='/mnt/project/d44-native-rl-20261103/IL44_PARENT',
        canonical_reward_scale='raw_0_1',action_shape=[8,3],steps=10,shift=5.,
        stochastic_steps=[7,8,9],innovation_clip=None,coordinate_reduction='sum24',
        dtype='FP32_parameters_states_probability_BF16_forward',checkpoint_policy='source',probability_layout='sequential',
        sampler_selection='fixed_K10_shift5_no_scorer',training_uses_navtest=False,
        stage_boundaries=[1.,4.,8.,12.],historical_teacher_capacity=8192)


def read(path):
    cfg=yaml.safe_load(Path(path).read_text())
    if cfg.get('schema')!='d44_native_rl_v1' or cfg.get('arm')not in ARMS:
        raise ValueError('D44 recipe schema mismatch')
    if cfg['global_scenes'] not in (32,64,128)or cfg['group']not in(8,16,32)or cfg['inner_updates']not in(2,4):
        raise ValueError('Unregistered batch/group/inner combination')
    if cfg['innovation_clip']is not None or cfg['coordinate_reduction']!='sum24':raise ValueError('Gaussian probability contract changed')
    if cfg['seed']!=42 or cfg['evaluation_seed']!=2101 or cfg['training_uses_navtest']:raise ValueError('Data/seed contract changed')
    if cfg.get('probability_layout','sequential')!='sequential':raise ValueError('Packed W3 probability layout is not qualified for training')
    if(cfg['steps'],cfg['shift'],cfg['action_shape'],cfg['stochastic_steps'],cfg['entropy'])!=(10,5.,[8,3],[7,8,9],0.):
        raise ValueError('Unimplemented sampler/action/entropy variation cannot be passed as metadata')
    if cfg['arm']=='D6':
        from .stages import stage_at
        stage_at(0,tuple(cfg['stage_boundaries']))
        if cfg['film_last_layers']!=8 or cfg['text_rank']!=16 or not cfg['post_feedback']:raise ValueError('D6 modules must be preinstalled before training')
    return cfg
