"""Fresh long-horizon retests of existing methods; no algorithm rewrite."""
import copy

from .group_runtime import CAMPAIGN,validate_recipe
from .standardized_recipe import PROTOCOL
from .credit import STRUCTURED_SCHEMA
from .dense_credit import SCHEMA as DENSE
from .saturated_credit import SCHEMA as SATURATED
from .curriculum import DEFAULT_CONFIG

ROOT=CAMPAIGN/'revival_training_20261101'
SCHEMA='world_legacy_algorithms_fixed50k_revival_v1'
ARMS=('REVIVAL_G16','REVIVAL_DENSE_G8','REVIVAL_SATURATED_G8','REVIVAL_ACTIVE_DATA_G8')
ROLES=('g16','dense','saturated','active_data')


def validate_revival_recipe(cfg):
 r=cfg.get('research',{});arm=r.get('arm')
 if (r.get('campaign_root')!=str(ROOT) or r.get('revival_training')!=SCHEMA
     or r.get('standardized_training')!=PROTOCOL or arm not in ARMS
     or cfg.get('max_steps')!=50000 or cfg.get('lr_warmup_steps')!=1000
     or cfg.get('lr_scheduler_type')!='cosine' or cfg.get('save_every_optimizer_updates')!=500
     or (cfg.get('learning_rate'),cfg.get('memory_learning_rate'))!=(5e-5,5e-6)):
  raise ValueError('Only independent revival G/root, fixed50k warm1000 floor.1 and original W3peakLR')
 i=ARMS.index(arm);g=16 if i==0 else 8
 if (r.get('standard_role')!=ROLES[i] or cfg.get('grpo',{}).get('sample',{}).get('group_size')!=g
     or r.get('structured_advantage')!=(STRUCTURED_SCHEMA,DENSE,SATURATED,STRUCTURED_SCHEMA)[i]
     or r.get('data_policy')!=('online_priority' if i==3 else 'uniform_full')
     or r.get('curriculum')!=(DEFAULT_CONFIG if i==3 else None)
     or r.get('endpoint_policy') is not None or r.get('endpoint_learning_rate_scale') is not None):
  raise ValueError('Revival role/G/originalcredit/originaldata/curriculum must match exactly')
 old=copy.deepcopy(cfg);old['research']['campaign_root']=str(CAMPAIGN)
 old.update(lr_scheduler_type='constant',lr_warmup_steps=100)
 validate_recipe(old)
 return cfg
