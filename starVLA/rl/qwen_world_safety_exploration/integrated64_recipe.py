"""One explicit fresh-IL70 experiment combining the four requested changes."""
import copy

from .group_runtime import CAMPAIGN, validate_recipe
from .retention_recipe import CONFIG as ORIGINAL_RETENTION
from .integrated64_schedule import PROTOCOL as SCHEDULE

ROOT = CAMPAIGN / 'integrated_training_20261102'
ARM = 'INTEGRATED_SAFETY_EP_B64_G16_RETENTION_WKC'
ROLE = 'integrated64'
SCHEMA = 'world_integratedSafetyEP_B64_G16_Retention_WKC_v1'
CREDIT_SCHEMA = 'world_SafetyEP_multiscene64_G16_group_credit_v1'
RETENTION_CONFIG = dict(copy.deepcopy(ORIGINAL_RETENTION), admission_start_update=250)
CONFIG = dict(schema=SCHEMA, scenes_per_rank=8, group_size=16, world_size=8,
              inner_updates=4, normalization_candidates=1024,
              optimizer_horizon_updates=12500, source_scene_clock_multiplier=8,
              source_candidate_clock_multiplier=16,
              retention=RETENTION_CONFIG, schedule=SCHEDULE)
EVALUATE_UPDATES = [250, 625, 1250] + list(range(1500, 12501, 250))
INITIAL_CAP = 6250
PROTOCOL = dict(**copy.deepcopy(CONFIG),
    behavior='64 distinct fresh contexts; all1024 candidates share one theta_old; four inner updates',
    normalization='group mean within each G16; one population std over world8*scene8*G16, held for four inners',
    backward='eight equal scene microbackwards/rank; DDP no_sync surrounds first7 forward+backward; one clip/Adam/scheduler',
    retention_loss='unchanged fresh SafetyPG + beta50*((1-.25*g)*freshILmeanMSE+.25*g*historicalRLmeanMSE) per scene',
    witnesses='TRAIN-only real nominal B1 ODE; strict NC/DAC/TTC/Comfort all1 and canonical>=.9; prior-window history only',
    queries='1024 freshPG/window plus64 freshB1 and at most64 historyB1 after admission; report all separately',
    clocks='scene exposure multiplier8; freshPG candidate multiplier16; ceil(u/4) windows include explicit pending material',
    schedule_description='scene-clock warm125, keep through625, cosine to2500, floor.1 to fixed12500; caps never reset LR',
    initialization='fresh same formal IL70, Adam0/RNG42; user confirmed no bestRL10k checkpoint fork',
    evaluation='same canonical NAVTEST12146; true single B1 K10 seed2101; no best-of-G selection')


def budget(updates):
    if type(updates) is not int or not 0 <= updates <= 12500:
        raise ValueError('Integrated budget update outside fixed horizon')
    windows = (updates + 3) // 4
    scenes, candidates = windows * 64, windows * 1024
    return dict(optimizer_updates=updates, next_inner=updates % 4,
        collected_windows=windows, scene_exposures=scenes, candidate_queries=candidates,
        nominal_scene_reference_updates=8*updates,
        nominal_candidate_reference_updates=16*updates,
        scene_surplus_at_pending_boundary=scenes-16*updates,
        candidate_surplus_at_pending_boundary=candidates-256*updates,
        strict_equal_nominal_source_budget=updates % 4 == 0,
        additional_official_query_accounting_required=updates > 250,
        query_scope='candidate_queries is fresh sourcePG only; actual TRAIN B1 verification must be added separately',
        matching='scene/candidate clocks are different; pending surplus and history cost prevent a same-compute claim')


def validate_integrated_recipe(cfg):
    r = cfg.get('research', {})
    if (r.get('campaign_root') != str(ROOT) or r.get('arm') != ARM
            or r.get('standard_role') != ROLE or r.get('integrated64') != CONFIG
            or r.get('structured_advantage') != CREDIT_SCHEMA
            or r.get('data_policy') != 'uniform_full'
            or cfg.get('evaluate_updates') != EVALUATE_UPDATES):
        raise ValueError('Only explicit same-IL70 integrated B64/G16/Retention/WKC recipe')
    forbidden = ('family', 'batch32', 'verified_retention', 'combined_consolidation_retention',
                 'consolidation_training', 'standardized_training', 'revival_training',
                 'curriculum', 'endpoint_policy', 'precision_training', 'mixed_exploration',
                 'success_memory', 'rescue_auxiliary', 'preference_auxiliary', 'fpo')
    if any(r.get(k) is not None for k in forbidden):
        raise ValueError('No additional mechanism in the four-factor integrated recipe')
    if (cfg.get('max_steps') != 12500 or cfg.get('lr_warmup_steps') != 125
            or cfg.get('lr_scheduler_type') != 'cosine'
            or cfg.get('gradient_accumulation_steps') != 8
            or cfg.get('save_every_optimizer_updates') != 125
            or cfg['grpo']['sample']['group_size'] != 16
            or cfg['grpo']['train']['rollout_buffer_batches'] != 8
            or cfg['grpo']['train']['num_inner_epochs'] != 4
            or cfg['logical_budget'] != dict(fresh_global_rollouts=3125,
                                            scenes=200000, candidates=3200000)):
        raise ValueError('Integrated actual scene/group/optimizer accounting changed')
    view = copy.deepcopy(cfg)
    view['research']['campaign_root'] = str(CAMPAIGN)
    view.update(max_steps=50000, lr_warmup_steps=100, lr_scheduler_type='constant',
                gradient_accumulation_steps=1,
                logical_budget=dict(fresh_global_rollouts=12500,
                                    scenes=100000, candidates=1600000))
    view['grpo']['train']['rollout_buffer_batches'] = 1
    validate_recipe(view)
    return cfg


def build_recipe(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    r = cfg['research']
    for key in ('standardized_training', 'standard_role', 'revival_training'):
        r.pop(key, None)
    r.update(campaign_root=str(ROOT), arm=ARM, standard_role=ROLE,
             integrated64=copy.deepcopy(CONFIG), structured_advantage=CREDIT_SCHEMA,
             stage_gate_updates=[u for u in EVALUATE_UPDATES if u <= INITIAL_CAP])
    cfg.update(max_steps=12500, lr_warmup_steps=125, lr_scheduler_type='cosine',
               gradient_accumulation_steps=8, save_every_optimizer_updates=125,
               evaluate_updates=list(EVALUATE_UPDATES),
               logical_budget=dict(fresh_global_rollouts=3125,
                                   scenes=200000, candidates=3200000))
    cfg['grpo']['sample']['group_size'] = 16
    cfg['grpo']['train']['rollout_buffer_batches'] = 8
    return validate_integrated_recipe(cfg)
