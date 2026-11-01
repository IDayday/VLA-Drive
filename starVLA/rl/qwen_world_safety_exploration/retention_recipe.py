"""Fresh-IL70, independently registered TRAIN regression-gated RL mean anchor."""
import copy
from .group_runtime import CAMPAIGN,validate_recipe
from .credit import STRUCTURED_SCHEMA
from .standardized_recipe import PROTOCOL as SCHEDULE

ROOT=CAMPAIGN/'improvement_training_20261101'
ARM='IMPROVEMENT_VERIFIED_RETENTION_G8'
SCHEMA='world_verified_RL_nominal_mean_retention_v1'
CONFIG=dict(schema=SCHEMA,capacity=256,admission_start_update=2000,history_scenes_per_rank=1,
 history_share=.25,min_score=.9,regression_margin=.01,upgrade_margin=1e-6,
 ODE_seed=2101,bank_seed=42,reservoir='uniform_over_verified_admissions')
PROTOCOL=dict(**CONFIG,quality='official NC=DAC=TTC=Comfort1 and canonical>=.9; TRAIN only',
 gate='same-root current trueB1 ODE fails strict4 OR score < storedscore-.01; freeze gate/target for4inners',
 target='stored historical currentLoRA+scene/predictiveMemory sourceW3 nominalmeans on verified ODE chain, NOT an endpoint target or density',
 loss='freshSafetyPG+50*((1-.25*gate)*freshILmeanMSE+.25*gate*historicalRLmeanMSE); original [.2,1,1] weights',
 ODE='actual B1 raw carrier, root seed_for(native_qwen_sourceflow_ODE,2101,token); cleanxODEK10shift5; no SDE innovations',
 privacy='newrun TRAIN stream only; no NAV labels, no old-run teachers; history never enters PG/reward/std',
 state='private rank RNG reservoir+uniform historical selection; bank/native RNG/targets/quality gates sealed at committed collection',
 feature_cache='hostLRU2 frozenQwen+ILfeatures only; currentMemory rebuilt each probability/anchor invocation',
 cost='beforeu2000 no extraODE; afterwards freshB1 plus atmost1 historicalB1 perrank/collection, extra queries counted separately')


def validate_retention_recipe(cfg):
 r=cfg.get('research',{})
 if (r.get('campaign_root')!=str(ROOT)or r.get('arm')!=ARM or r.get('standard_role')!='verified_retention' or r.get('family')is not None
     or r.get('verified_retention')!=CONFIG or r.get('structured_advantage')!=STRUCTURED_SCHEMA
     or r.get('data_policy')!='uniform_full' or cfg.get('max_steps')!=50000 or cfg.get('lr_warmup_steps')!=1000
     or cfg.get('lr_scheduler_type')!='cosine' or cfg.get('save_every_optimizer_updates')!=500
     or cfg['grpo']['sample']['group_size']!=8):raise ValueError('Only registered freshIL70/H/W3/G8 verified-retention recipe')
 for k in ('batch32','consolidation_training','standardized_training','revival_training','curriculum','endpoint_policy',
           'success_memory','rescue_auxiliary','preference_auxiliary','fpo'):
  if r.get(k)is not None:raise ValueError('Verified retention must not bundle another algorithm: '+k)
 old=copy.deepcopy(cfg);old['research']['campaign_root']=str(CAMPAIGN)
 old.update(lr_warmup_steps=100,lr_scheduler_type='constant');validate_recipe(old)
 return cfg


def build_recipe(base):
 c=copy.deepcopy(base);r=c['research'];r.pop('standardized_training',None)
 r.update(campaign_root=str(ROOT),arm=ARM,standard_role='verified_retention',verified_retention=copy.deepcopy(CONFIG),stage_gate_updates=[2000,5000,10000,12000,14000,16000,18000,20000])
 c.update(max_steps=50000,lr_warmup_steps=1000,lr_scheduler_type='cosine',save_every_optimizer_updates=500,
  evaluate_updates=[2000,5000,10000]+list(range(12000,50001,2000)))
 return validate_retention_recipe(c)
