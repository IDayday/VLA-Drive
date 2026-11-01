"""Actual production collection99 CPU replay, no weights/scorer/GPU execution."""
import copy
import json
from pathlib import Path

import pytest
import torch

from starVLA.rl.qwen_world_safety_exploration.runtime import runtime_for_config
from starVLA.rl.qwen_world_safety_exploration.credit import raw_advantage_parts
from starVLA.rl.qwen_world_safety_exploration.combined_credit import decision,tensor_digest

RUN=Path('/mnt/project/qwen-world-safety-exploration-20261031/standard_training_20261101/runs/STANDARD_COMBINED_G8')


def capture():
 identity=json.loads((RUN/'identity.json').read_text());runtime=runtime_for_config(RUN/'resolved_recipe.yaml')
 runtime.credit=runtime.make_credit(identity['recipe']);rs=[];states=[]
 for rank in range(8):
  p=next((RUN/'failures').glob(f'48bfa05850524d4ca31cd1b66296c45d/rank{rank}-*/partial_rollout.pt'))
  rs.append(torch.load(p,map_location='cpu',weights_only=False))
  states.append(json.loads((p.parent/'evidence.json').read_text())['state'])
 return identity,runtime,rs,states


def test_actual_gpu_reduction_residual_preserved_and_discrete_stats_verified_exactly():
 ident,runtime,rs,states=capture();before=[tensor_digest({k:r[k] for k in ('chain','old_logp','advantages','adv_mask','combined_credit')}) for r in rs]
 cpu=raw_advantage_parts(rs[0]['structured_advantage']['components'])['A_raw']
 actual=rs[0]['combined_credit']['original_raw_A'];assert actual.ne(0).sum()==8 and cpu.ne(0).sum()==3
 assert (actual-cpu).abs().max()<2e-16
 assert rs[0]['advantage_statistics']['combined_true_nonzero_adv_fraction']==1.
 ss=[runtime.validate_boundary(s,r,ident) for s,r in zip(states,rs)];runtime.validate_ranks(ss)
 assert before==[tensor_digest({k:r[k] for k in ('chain','old_logp','advantages','adv_mask','combined_credit')}) for r in rs]
 for field,value in [('combined_true_nonzero_adv_fraction',.375),('combined_adv_cap5_candidates',1)]:
  bad=copy.deepcopy(ss);bad[0]['advantage_statistics'][field]=value
  with pytest.raises(ValueError):runtime.validate_ranks(bad)
 bad=copy.deepcopy(ss);bad[0]['advantages'][0]+=.01
 with pytest.raises(ValueError):runtime.validate_ranks(bad)


def test_actual_branch_recompute_uses_saved_signs_and_structural_dense_saturation_zeros():
 ident,runtime,rs,states=capture()
 for r in rs:
  info=r['combined_credit'];components=r['structured_advantage']['components']
  recomputed=decision(r['records'],r['rewards'],components,info['original_raw_A'])
  assert tensor_digest(info)==tensor_digest(recomputed)
 # The structural zero gates do no mean-of-unequal-values reduction: Dense has
 # constantSafetycost and nofeasibleEP; Sat has allEP1 ->zeroEPoffsets exactly.
 for g in (8,16):
  dense=torch.ones(1,g,4,dtype=torch.float64);dense[:,:,1]=0
  sat=torch.ones(1,g,4,dtype=torch.float64)
  for x in (dense,sat):assert raw_advantage_parts(x)['A_raw'].eq(0).all()
