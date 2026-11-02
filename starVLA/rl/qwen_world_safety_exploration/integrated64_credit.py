"""Original SafetyEP math on64fresh scenes/G16, one1024 population scale."""
import copy
import torch
from .credit import GroupCredit,canonical_components,protocol as original_protocol
from .integrated64_recipe import SCHEMA,CREDIT_SCHEMA


def protocol():
 p=original_protocol(16);p.update(schema=CREDIT_SCHEMA,scenes_per_rank=8,normalization_population_groups=64,normalization_population_candidates=1024,
  normalization='onepopulation std of64freshscene/G16 originalSafetyrawA; zero groupsincluded; no historicalmaterial',reduction='mean8equal scene losses perrank, thenDDP rankmean; fixed4inner')
 return p


@torch.no_grad()
def finalize_window(window):
 if window.get('schema')!=SCHEMA or 'advantages'in window or len(window['samples'])!=8:raise ValueError('Finalize8freshG16groups exactlyonce')
 samples=window['samples'];mats=[canonical_components(s['records'],16,device=s['rewards'].device)for s in samples]
 if any(not torch.equal(s['rewards'],m['canonical_rewards'])or'advantages'in s for s,m in zip(samples,mats)):raise ValueError('Original fresh canonical rewards changed')
 rewards=torch.cat([s['rewards']for s in samples]);components=torch.cat([m['components']for m in mats]);stats=GroupCredit(16).advantages(rewards,components)
 if stats['statistics']['normalization_population_candidates']!=1024:raise ValueError('Integrated normalization requires actual8ranks×8scenes×G16')
 meta=dict(stats['statistics'],structured_advantage_schema=CREDIT_SCHEMA);parts=('feasible','cost','A_safe','A_EP','A_raw')
 window.update(rewards=rewards,canonical_rewards=rewards.clone(),advantages=stats['advantages'],adv_mask=stats['adv_mask'],global_std=stats['global_std'],
  raw_group_std=stats['raw_group_std'],normalization_std=stats['normalization_std'],advantage_statistics=meta,structured_advantage=dict(protocol=protocol(),components=components,**{k:stats[k]for k in parts}))
 for b,s in enumerate(samples):
  sl=slice(b*16,(b+1)*16)
  s.update(canonical_rewards=s['rewards'].clone(),advantages=stats['advantages'][sl],adv_mask=stats['adv_mask'][sl],global_std=stats['global_std'],
   raw_group_std=stats['raw_group_std'][b:b+1],normalization_std=stats['normalization_std'],advantage_statistics=copy.deepcopy(meta),
   structured_advantage=dict(protocol=protocol(),components=components[b:b+1],**{k:stats[k][b:b+1]for k in parts}))
 return window
