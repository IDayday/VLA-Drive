"""Once-finalized world32scene/G8 Safety credit; leaves retain sourcePPO loss."""
import copy
import torch
from .credit import GroupCredit,canonical_components,protocol as source_protocol
from .batch32_recipe import CREDIT_SCHEMA


def protocol():
 p=source_protocol(8);p.update(schema=CREDIT_SCHEMA,scenes_per_rank=4,normalization_population_groups=32,
  normalization_population_candidates=256,normalization='population std over all32freshscene/G8 originalSafetyrawA once; no active-only filtering',
  reduction='equal mean of4scene local source losses, then rankmean; reuse unchanged advantages for4inners')
 return p


@torch.no_grad()
def finalize_window(window):
 if 'advantages' in window or window.get('schema')!='world_multiscene32_G8_sourcePPO_query_clock_v1':raise ValueError('Fresh Batch32 window finalized exactly once')
 samples=window['samples']
 if len(samples)!=4 or any('advantages' in s for s in samples):raise ValueError('Exactly4 fresh unfinalized groups required')
 mats=[canonical_components(s['records'],8,device=s['rewards'].device) for s in samples]
 for s,m in zip(samples,mats):
  if not torch.equal(s['rewards'],m['canonical_rewards']):raise ValueError('Original canonical rewards changed')
 rewards=torch.cat([s['rewards'] for s in samples]);components=torch.cat([m['components'] for m in mats])
 stats=GroupCredit(8).advantages(rewards,components)
 if stats['statistics']['normalization_population_candidates']!=256:raise ValueError('Formal Batch32 normalization requires eight ranks, all256 candidates')
 proto=protocol();meta=dict(stats['statistics'],structured_advantage_schema=CREDIT_SCHEMA)
 window.update(rewards=rewards,canonical_rewards=rewards.clone(),advantages=stats['advantages'],adv_mask=stats['adv_mask'],
  global_std=stats['global_std'],raw_group_std=stats['raw_group_std'],normalization_std=stats['normalization_std'],advantage_statistics=meta,
  structured_advantage=dict(protocol=proto,components=components,**{k:stats[k] for k in ('feasible','cost','A_safe','A_EP','A_raw')}))
 for b,s in enumerate(samples):
  sl=slice(b*8,(b+1)*8)
  s.update(canonical_rewards=s['rewards'].clone(),advantages=stats['advantages'][sl],adv_mask=stats['adv_mask'][sl],
   global_std=stats['global_std'],raw_group_std=stats['raw_group_std'][b:b+1],normalization_std=stats['normalization_std'],advantage_statistics=copy.deepcopy(meta),
   structured_advantage=dict(protocol=copy.deepcopy(proto),components=components[b:b+1],**{k:stats[k][b:b+1] for k in ('feasible','cost','A_safe','A_EP','A_raw')}))
 return window
