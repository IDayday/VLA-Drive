"""Revival strict recipe parity and actual eight-rank native legacy paths."""
import copy
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

from test_qwen_world_safety_curriculum import manifest
from test_world_safety_exploration_dense import population as dense_population
from test_world_safety_exploration_saturated import population as saturated_population
from test_world_safety_exploration_groups import components,records
from world_postcot.test_scope_rl import TinyModel
from starVLA.rl.qwen_world_postcot.scope_actor import ScopeRecipeActor
from starVLA.rl.qwen_world_postcot.scope_runtime import ScopeRuntime
from starVLA.rl.qwen_world_postcot.scope_contracts import SCOPE_SCHEMA
from starVLA.rl.qwen_world_safety_exploration import revival_recipe as R
from starVLA.rl.qwen_world_safety_exploration.runtime import runtime_for_config
from starVLA.rl.qwen_world_safety_exploration.conditions import GroupMemoryConditions
from starVLA.rl.qwen_world_safety_exploration.group_runtime import GroupRuntime
from starVLA.rl.qwen_world_safety_exploration.curriculum import OnlinePriorityStream
from starVLA.rl.qwen_world_safety_exploration.recovery import tensor_digest
from starVLA.rl.qwen_world_research_data.data import SceneStream
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss
from starVLA.rl.simwam_recipe import checkpoint
from starVLA.rl.simwam_recipe.contracts import seed_for

TEMPLATES=R.ROOT.parent/'configs/revival_training_20261101_templates'
def cfg(arm):return yaml.safe_load((TEMPLATES/(arm+'.yaml')).read_text())


def test_strict_revival_templates_are_old_algorithms_with_only_new_schedule_identity():
 names=('SAFETY_G16','SAFETY_DAC_DENSE_G8','SAFETY_SATURATED_G8','SAFETY_ACTIVE_DATA_G8')
 for arm,oldname in zip(R.ARMS,names):
  actual=cfg(arm);old=yaml.safe_load((R.ROOT.parent/'configs'/(oldname+'.yaml')).read_text());before=copy.deepcopy(actual)
  assert R.validate_revival_recipe(actual) is actual and actual==before
  assert actual['grpo']==old['grpo'] and actual['logical_budget']==old['logical_budget']
  assert actual['research'].get('curriculum')==old['research'].get('curriculum')
  rt=runtime_for_config(TEMPLATES/(arm+'.yaml'));credit=rt.make_credit(actual)
  assert credit.group_size==actual['grpo']['sample']['group_size'] and credit.protocol['schema']==old['research']['structured_advantage']
  for key,value in [('campaign_root',str(R.ROOT.parent)),('revival_training','wrong'),('standard_role','wrong')]:
   bad=copy.deepcopy(actual);bad['research'][key]=value
   with pytest.raises(ValueError):R.validate_revival_recipe(bad)
  bad=copy.deepcopy(actual);bad['grpo']['sample']['noise_level']=.03
  with pytest.raises(ValueError):R.validate_revival_recipe(bad)


class Data:
 sha='revival_CPU_current_features'
 def __init__(self,tokens):self.tokens=tokens
 def observation(self,token):return dict(images={},navigation='forward',proprio=[1.,0.,0.,0.,1.,0.,0.,0.])


def setup(arm,manifest):
 c=cfg(arm);runtime=runtime_for_config(TEMPLATES/(arm+'.yaml'));runtime.group_size=c['grpo']['sample']['group_size'];runtime.credit=runtime.make_credit(c)
 base=SceneStream(manifest,8)
 if arm==R.ARMS[3]:
  with patch.object(GroupRuntime,'make_scene_stream',return_value=base):runtime.make_scene_stream(c,None,8)
 else:runtime.stream=base
 torch.manual_seed(42);model=TinyModel()
 actor=ScopeRecipeActor(model.actor,c,scene_provider=model.scene_provider,predictive_provider=model.predictive_provider)
 actor._loss=ResearchLoss(c)
 cond=GroupMemoryConditions(model,Data(base.tokens),'unit_IL70',actor,group_size=runtime.group_size,processor=lambda observations:dict(current=True))
 object.__setattr__(actor,'conditions',cond)
 with patch.object(GroupRuntime,'make_actor',return_value=(actor,None,None)):actor,_,_=runtime.make_actor(c,None,torch.device('cpu'))
 opt=ScopeRuntime().make_optimizer(actor,c);actor._loss.optimizer=opt;actor._loss.learning_rate=c['learning_rate'];sched=actor._loss._build_scheduler('cosine',50000,1000)
 ident=dict(world=8,sha256='revival_CPU',schema=SCOPE_SCHEMA,recipe=c,projected_condition_identity=cond.sha,score_protocol=dict(sha256='mock'))
 return runtime,actor,cond,opt,sched,ident


class ScoreMock:
 protocol=dict(sha256='mock')
 def __init__(self,arm,rank,g):
  if arm==R.ARMS[1]:self.records=dense_population(g)[1][rank]
  elif arm==R.ARMS[2]:self.records=saturated_population(g)[1][rank]
  else:
   x=components(g)[0] if arm==R.ARMS[3] else components(g)[rank]
   self.records=records(x)
 def score(self,job):
  token,log,actions=job;rr=copy.deepcopy(self.records)
  for j,r in enumerate(rr):r.update(token=token,log=log,protocol_sha256='mock',cache=dict(sha256='mockcache'),requested_trajectory=actions[j].copy())
  return rr


def native(rank,init,root,manifest):
 torch.set_num_threads(1);dist.init_process_group('gloo',init_method='file://'+init,rank=rank,world_size=8,timeout=timedelta(seconds=240))
 try:
  for arm in R.ARMS:
   runtime,actor,cond,opt,sched,ident=setup(arm,manifest);g=runtime.group_size
   generator=torch.Generator().manual_seed(seed_for('simwam_source_rank_rollout',42,rank));service=ScoreMock(arm,rank,g)
   st=dict(updates=0,collections=0,next_inner=0,scene_cursor=0,raw_scene_exposures=0,new_candidates=0,phase='READY')
   runtime.bind_training_state(st,None,ident);count=12 if arm==R.ARMS[3] else 1
   hooks=dict(boundary_validator=runtime.validate_boundary,rank_validator=runtime.validate_ranks)
   folder=Path(root)/arm
   with patch.object(torch.cuda,'is_available',lambda:False):
    for col in range(1,count+1):
     token,pos=runtime.stream.at(st['scene_cursor'],rank)
     r=runtime.collect(actor,cond.get(token,torch.device('cpu')),token,runtime.stream.logs[token],service,generator,
      variant='R_CANONICAL',version=dict(run=ident['sha256'],update=st['updates'],adapter_sha256='same'))
     r['stream_position']=pos;runtime.finalize_advantages(r)
     st.update(collections=col,scene_cursor=8*col,raw_scene_exposures=8*col,new_candidates=8*g*col,next_inner=0,phase='ROLLOUT_COMMITTED')
     runtime.on_collection_committed(r,st)
     for inner in range(4):
      if col==count:
       before=tensor_digest(opt.state_dict());old_sched=copy.deepcopy(sched.state_dict());rng=generator.get_state().clone()
       oldbank=copy.deepcopy(st.get('curriculum'));digest=tensor_digest({k:v for k,v in r.items() if k!='records'})
       path=checkpoint.save(folder,actor,opt,sched,generator,st,r,ident,**hooks)
       runtime,actor,cond,opt,sched,otherid=setup(arm,manifest);assert otherid==ident
       hooks=dict(boundary_validator=runtime.validate_boundary,rank_validator=runtime.validate_ranks)
       st,r=checkpoint.restore(path,actor,opt,sched,generator,ident,**hooks)
       assert tensor_digest(opt.state_dict())==before and sched.state_dict()==old_sched and torch.equal(generator.get_state(),rng)
       assert st.get('curriculum')==oldbank and tensor_digest({k:v for k,v in r.items() if k!='records'})==digest
       runtime.bind_training_state(st,r,ident)
      value,metrics=actor(r);value.backward();params=[p for p in actor.parameters() if p.requires_grad]
      assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
      for p in params:dist.all_reduce(p.grad);p.grad.div_(8)
      torch.nn.utils.clip_grad_norm_(params,1.);opt.step();sched.step();opt.zero_grad(set_to_none=True)
      st['updates']+=1;st['next_inner']=inner+1;st['phase']='INNER_COMMITTED'
     st.update(next_inner=0,phase='READY')
    path=checkpoint.save(folder,actor,opt,sched,generator,st,None,ident,**hooks)
    loaded,none=checkpoint.restore(path,actor,opt,sched,generator,ident,**hooks)
    assert loaded==st and none is None
   if arm==R.ARMS[3]:
    assert r['stream_position']['selection_source']=='priority'
    assert st['curriculum']['stats']['priority_exposures']==1 and st['curriculum']['uniform_draws']==11
   if rank==0:
    (folder/'CPU_PASS.json').write_text(json.dumps(dict(arm=arm,world=8,group=g,updates=st['updates'],boundaries=5,
     final_LRs=[q['lr'] for q in opt.param_groups],private_curriculum_preserved=arm==R.ARMS[3],
     curriculum_summary=runtime.public_training_state(st).get('curriculum'),mock='Actual ScopeRecipeActor/memory with tiny dimensions; algebraic canonical scorer/frozen Qwen fixture'),default=list))
 finally:dist.destroy_process_group()


def test_actual_eight_rank_native_G16_Dense_Sat_Active_priority_recovery(tmp_path,manifest):
 torch.set_num_threads(1)
 mp.start_processes(native,args=(str(tmp_path/'gloo'),str(tmp_path/'native'),str(manifest)),nprocs=8,join=True,start_method='fork')
 for arm in R.ARMS:assert (tmp_path/'native'/arm/'CPU_PASS.json').is_file()
