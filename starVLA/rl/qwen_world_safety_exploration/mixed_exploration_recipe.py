"""Fixed four-late/four-earlier W3 mixture, independent fresh-IL70 experiment."""
import copy
from .group_runtime import CAMPAIGN,validate_recipe
from .credit import STRUCTURED_SCHEMA
ROOT=CAMPAIGN/'improvement_training_20261101'
ARM='IMPROVEMENT_MIXED_NOISE_G8'
SCHEMA='world_fixed4late4early_W3_sourcePPO_v1'
CONFIG=dict(schema=SCHEMA,group_size=8,world_size=8,late_candidates=4,early_candidates=4,
 late_steps=[7,8,9],early_steps=[3,6,8],innovation_slots_per_candidate=3)
ROWS=[[7,8,9]]*4+[[3,6,8]]*4
PROTOCOL=dict(**CONFIG,steps_per_candidate=ROWS,probability='source relative-clamped dimension-mean24 proxy for each actual candidate/ordinal-slot transition; NOT exact joint density',
 sampling='oneG8 initialroot draw+threeG8 orderedinnovation draws; eta.1 rigorous source means/std; branchstrata fixed independently of reward',
 ordinal_slots=['late7_early3','late8_early6','late9_early8'],anchor='same beta50 original [.2,1,1] weighted nominalmeanMSE, over actual row-specific states/times; still3slots/candidate',
 advantages='unchanged SafetyEP on all8 mixed candidates and global64 rawA std once; no perbranch centering/filtering',
 deployment='unchanged singlecandidate original cleanxODE10shift5; no reward-based branchselection at inference',
 cost='same8scored candidates/rank/collection,10batch8 velocityforwards and3ordinal gradientforwards perinner; extra grouping/gather bookkeeping only')

def validate_mixed_recipe(cfg):
 r=cfg.get('research',{})
 if (r.get('campaign_root')!=str(ROOT)or r.get('arm')!=ARM or r.get('standard_role')!='mixed_noise'
     or r.get('mixed_exploration')!=CONFIG or r.get('family')is not None
     or r.get('structured_advantage')!=STRUCTURED_SCHEMA or r.get('data_policy')!='uniform_full'
     or cfg.get('max_steps')!=50000 or cfg.get('lr_warmup_steps')!=1000 or cfg.get('lr_scheduler_type')!='cosine'
     or cfg.get('save_every_optimizer_updates')!=500 or cfg['grpo']['sample']['group_size']!=8):
  raise ValueError('Only fresh H fixed4late4early mixedW3/SafetyEP sourcePPO recipe')
 if any(r.get(k)is not None for k in ('batch32','verified_retention','consolidation_training','standardized_training','revival_training','curriculum','endpoint_policy','precision_training','success_memory','rescue_auxiliary','fpo')):raise ValueError('Mixed exploration is a separate kernel change')
 old=copy.deepcopy(cfg);old['research']['campaign_root']=str(CAMPAIGN);old.update(lr_warmup_steps=100,lr_scheduler_type='constant');validate_recipe(old);return cfg

def build_recipe(base):
 c=copy.deepcopy(base);r=c['research'];r.pop('standardized_training',None)
 r.update(campaign_root=str(ROOT),arm=ARM,standard_role='mixed_noise',mixed_exploration=copy.deepcopy(CONFIG),stage_gate_updates=[2000,5000,10000,12000,14000,16000,18000,20000])
 c.update(max_steps=50000,lr_warmup_steps=1000,lr_scheduler_type='cosine',save_every_optimizer_updates=500,evaluate_updates=[2000,5000,10000]+list(range(12000,50001,2000)))
 return validate_mixed_recipe(c)
