"""Exact S6 science; original performance V1 protocols remain the default."""
import copy
import math

from starVLA.rl.qwen_world_performance.contracts import ROOT,SCHEMA as BASE_SCHEMA,EVALUATION_EPOCHS as BASE_EVALUATIONS,validate_recipe as validate_base
from starVLA.rl.qwen_world_performance.nominal_improvement import QUALITY_CONFIG
from .calibration import PROTOCOL as CALIBRATION_PROTOCOL
from starVLA.rl.qwen_world_performance.adaptive_group import PROTOCOL as EXTRA_PROTOCOL
from starVLA.rl.qwen_world_postcot.contracts import fingerprint

ARM='S6_VERIFIED_QUALITY'
SCHEMA='qwen_world_verified_quality_rl_v1'
EVALUATION_EPOCHS=[.0625,.125]+list(BASE_EVALUATIONS)
SCHEDULE=dict(type='warmup_hold_cosine_floor',clock='valid_scene_exposures',warmup_epochs=.08,
    hold_until_epoch=.75,cosine_decay_end_epoch=4.,minimum_lr_ratio=.05)
V1_SPEC=dict(start_epoch=.5,anchor_budget_share=.05,min_improvement=.02,strict_high_min_pdms=.9,
    sigma_min=.1,sigma_max=.9,mc_samples=2,cfm_share=.75,ode_last_share=.25,
    calibration_slots=100,gradient_diagnostic_slots=500,adaptive_extra_G=True)
NOMINAL_SPEC=dict(V1_SPEC,schema=QUALITY_CONFIG['schema'],start_epoch=0.,
    strict_high_nominal_anchor_budget_share=.02,allow_strict_high_nominal_teacher=True,
    repair_gain_margin=.02,safe_high_gain_margin=.005)
NOMINAL_SPEC.pop('min_improvement')
PROTOCOL=dict(schema='world_verified_quality_fresh_IL70_B64G16M4_v2',arm=ARM,
    nominal=copy.deepcopy(QUALITY_CONFIG),calibration=copy.deepcopy(CALIBRATION_PROTOCOL),
    adaptive_extra=copy.deepcopy(EXTRA_PROTOCOL),B=64,G=16,M=4,
    peak_action_lr=5e-5,peak_memory_lr=5e-6,schedule=copy.deepcopy(SCHEDULE),
    evaluation_epochs=copy.deepcopy(EVALUATION_EPOCHS),minimum_epochs=3.,initial_target_epochs=6.,optional_epoch_cap=20.,
    initialization='same complete IL70, fresh Adam0/RNG42; never import an RL checkpoint',
    teacher='actual same-scene/current-version strict NC/DAC/TTC/Comfort1, PDMS>=.9; repair gain>=.02, already-strict-high B1 gain>=.005',
    budget='replace original beta50 IL share: .05 when B1 not strict-high, .02 when B1 strict-high; .75CFM+.25actualODElast',
    sampling='first full TRAIN epoch uniform; existing own-run TRAIN-only priority thereafter; never NAVTEST tokens/labels')
QUALITY_PROTOCOL_SHA256=fingerprint(PROTOCOL)


def build_recipe(basecfg,arm=ARM):
    if arm!=ARM:raise ValueError('Only exact S6 arm is implemented')
    validate_base(basecfg);cfg=copy.deepcopy(basecfg);r=cfg['research'];p=r['performance']
    if p['group_size']!=16:raise ValueError('S6 requires original G16')
    r.update(schema=SCHEMA,arm=ARM,family='verified_quality',verified_quality=copy.deepcopy(PROTOCOL))
    p.update(algorithm='verified_quality',scenes_per_collection=64,group_size=16,inner_updates=4,
        nominal_rescue=copy.deepcopy(NOMINAL_SPEC),schedule=copy.deepcopy(SCHEDULE),minimum_epochs=3.,initial_target_epochs=6.,extension_epochs=[8,12,20],
        evaluation_epochs=copy.deepcopy(EVALUATION_EPOCHS),
        stages=[dict(name='main',start_epoch=0.,B=64,action_lr=5e-5,memory_lr=5e-6,rescue=True,history=False)])
    for key in ('awfm','fpo','history'):p.pop(key,None)
    windows=math.ceil(p['population']/64)*20
    cfg.update(learning_rate=5e-5,memory_learning_rate=5e-6,gradient_accumulation_steps=8,max_steps=windows*4,
        logical_budget=dict(fresh_global_rollouts=windows,scenes=windows*64,candidates=windows*64*16))
    cfg['grpo']['train']['num_inner_epochs']=4
    return validate_recipe(cfg)


def validate_recipe(cfg):
    r=cfg['research'];p=r['performance']
    if (r.get('schema')!=SCHEMA or r.get('arm')!=ARM or r.get('family')!='verified_quality'
            or r.get('verified_quality')!=PROTOCOL or r.get('campaign_root')!=str(ROOT)):
        raise ValueError('Exact independent S6 identity/protocol required')
    if (p.get('algorithm')!='verified_quality' or p['nominal_rescue']!=NOMINAL_SPEC or p['schedule']!=SCHEDULE
            or (p['scenes_per_collection'],p['group_size'],p['inner_updates'])!=(64,16,4)
            or p['stages']!=[dict(name='main',start_epoch=0.,B=64,action_lr=5e-5,memory_lr=5e-6,rescue=True,history=False)]
            or cfg['learning_rate']!=5e-5 or cfg['memory_learning_rate']!=5e-6 or cfg['gradient_accumulation_steps']!=8
            or p['priority']['start_epoch']!=1. or r['data_policy']!='uniform_full'
            or p['evaluation_epochs']!=EVALUATION_EPOCHS or p['minimum_epochs']!=3. or p['initial_target_epochs']!=6. or p['extension_epochs']!=[8,12,20]):
        raise ValueError('Changed S6 batch/schedule/teacher gate or original first-epoch population')
    windows=math.ceil(p['population']/64)*20
    if cfg['max_steps']!=windows*4 or cfg['logical_budget']!=dict(fresh_global_rollouts=windows,scenes=windows*64,candidates=windows*64*16):
        raise ValueError('S6 valid exposure/allocated padding budget changed')
    original=copy.deepcopy(cfg);original['research'].update(schema=BASE_SCHEMA)
    for k in ('family','verified_quality'):original['research'].pop(k)
    original['research']['performance'].update(algorithm='nominal_rescue',nominal_rescue=copy.deepcopy(V1_SPEC))
    validate_base(original)
    return cfg
