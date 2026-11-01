"""Fixed-update scheduler, disjoint credit, realScope/native CPU boundaries."""
import copy
import io
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

from test_world_safety_exploration_groups import records,rollout,ToyModel,state
from test_world_safety_exploration_inversion import population as inversion_population
from test_world_safety_exploration_dense import flags
from world_postcot.test_scope_rl import make
from starVLA.rl.qwen_world_postcot.scope_contracts import MEMORY,SCOPE_SCHEMA
from starVLA.rl.qwen_world_postcot.scope_runtime import ScopeRuntime
from starVLA.rl.qwen_world_safety_exploration import combined_credit as C
from starVLA.rl.qwen_world_safety_exploration import standardized_recipe as S
from starVLA.rl.qwen_world_safety_exploration.group_runtime import GroupRuntime
from starVLA.rl.qwen_world_safety_exploration.runtime import runtime_for_config
from starVLA.rl.qwen_world_safety_exploration.recovery import tensor_digest
from starVLA.rl.qwen_world_safety_exploration.credit import GroupCredit,canonical_components,raw_advantage_parts
from starVLA.rl.qwen_world_safety_exploration.endpoint_runtime import EndpointLoss,EndpointRuntimeMixin
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss
from starVLA.rl.simwam_recipe import checkpoint
from starVLA.rl.simwam_recipe.contracts import seed_for

TEMPLATES=S.ROOT.parent/'configs/standard_training_20261101_templates'
def config(arm):return yaml.safe_load((TEMPLATES/(arm+'.yaml')).read_text())


def scalar_optimizer(peaks):
 p=[torch.nn.Parameter(torch.tensor(.3)),torch.nn.Parameter(torch.tensor(.7))]
 opt=torch.optim.AdamW([dict(params=[p[0]],lr=peaks[0],name='action_lora'),dict(params=[p[1]],lr=peaks[1],name='scene_predictive_memory')])
 return p,opt


def test_scheduler_endpoints_fixed_horizon_and_exact_stage_resume():
 assert S.multiplier(0)==.001 and S.multiplier(1000)==1 and S.multiplier(50000)==.1
 assert S.multiplier(999)==pytest.approx(.999001)
 assert S.multiplier(1001)<1 and S.multiplier(30000)>S.multiplier(49999)>.1
 for peak in ([5e-5,5e-6],[1e-5,1e-6]):
  params,opt=scalar_optimizer(peak);sched=S.FixedHorizonCosineLR(opt)
  assert sched.last_epoch==0 and [g['lr'] for g in opt.param_groups]==[p*.001 for p in peak]
  for u in range(1,10002):
   for p in params:p.grad=torch.ones_like(p)
   opt.step();sched.step()
   assert [g['lr'] for g in opt.param_groups]==[p*S.multiplier(u) for p in peak]
   if u in (2000,5000,10000):
    stream=io.BytesIO();torch.save(dict(opt=opt.state_dict(),sched=sched.state_dict()),stream);stream.seek(0)
    saved=torch.load(stream,weights_only=False);pp,oo=scalar_optimizer(peak);ss=S.FixedHorizonCosineLR(oo)
    oo.load_state_dict(saved['opt']);ss.load_state_dict(saved['sched'])
    assert tensor_digest(oo.state_dict())==tensor_digest(opt.state_dict()) and ss.state_dict()==sched.state_dict()
    for p in pp:p.grad=torch.ones_like(p)
    oo.step();ss.step();assert ss.last_epoch==u+1 and [g['lr'] for g in oo.param_groups]==[p*S.multiplier(u+1) for p in peak]
  for key,value in [('horizon_updates',10000),('min_lr_ratio',.01)]:
   bad=copy.deepcopy(sched.state_dict());bad['fixed_protocol'][key]=value
   with pytest.raises(ValueError):sched.load_state_dict(bad)
  bad=copy.deepcopy(sched.state_dict());bad['_last_lr'][1]=bad['_last_lr'][0]
  with pytest.raises(ValueError):sched.load_state_dict(bad)
 with pytest.raises(ValueError):S.multiplier(50001)
 for arm in S.ARMS:
  c=config(arm);before=copy.deepcopy(c);assert S.validate_standardized_recipe(c) is c and c==before
  runtime_for_config(TEMPLATES/(arm+'.yaml'))
  for field,value in [('max_steps',10000),('lr_warmup_steps',100),('lr_scheduler_type','constant')]:
   bad=copy.deepcopy(c);bad[field]=value
   with pytest.raises(ValueError):S.validate_standardized_recipe(bad)


def population(fallback=False):
 x=torch.ones(8,8,4,dtype=torch.float64);x[:,:,3]=1 if fallback else torch.linspace(.8,1,8)
 x[0,:,1]=0
 recs=[records(xx) for xx in x]
 for rec,f in zip(recs[0],flags(8)):rec[C.D.FIELD]=f.numpy()
 if not fallback:
  xx,rr,*_=inversion_population(8);x[2]=xx[4];recs[2]=rr[4]
 x[1]=1;recs[1]=records(x[1])
 for r in recs[3:]:
  for a in r:
   m=a['metrics'];m['comfort']=1.;m['score']=(5*m['ego_progress']+7)/12;a['readouts']['canonical_PDMS']=m['score']
 rewards=torch.cat([canonical_components(r,8)['canonical_rewards'] for r in recs]);material=torch.cat((rewards.double().unsqueeze(-1),x),-1)
 infos=[C.decision(recs[i],rewards[i:i+1],x[i:i+1],raw_advantage_parts(x[i:i+1])['A_raw']) for i in range(8)]
 allnew=torch.cat([i['updated_raw_A'] for i in infos]);return x,recs,rewards,material,infos,allnew


def test_combination_disjoint_base_scale_fallback_unrelated_loss_gradient_and_native_material(monkeypatch):
 for fallback in (False,True):
  x,recs,rewards,material,infos,allnew=population(fallback);credit=C.CombinedCredit(8)
  assert [i['branch'] for i in infos[:3]]==(['dense','saturated','saturated'] if fallback else ['dense','saturated','inversion'])
  for j in range(8):
   b=GroupCredit(8).advantages(rewards[j:j+1],x[j:j+1],all_material=material)
   got=credit.advantages(rewards[j:j+1],x[j:j+1],info=infos[j],all_material=material,all_new_raw=allnew)
   expected=allnew.std(unbiased=False) if fallback else b['normalization_std']
   assert torch.equal(got['normalization_std'],expected)
   torch.testing.assert_close(got['A_raw'].sum(),b['A_raw'].sum(),atol=1e-14,rtol=0)
   torch.testing.assert_close(got['A_safe']+got['A_EP'],got['A_raw'],atol=1e-14,rtol=0)
   if not fallback and infos[j]['branch']=='none':
    assert torch.equal(got['advantages'],b['advantages']) and torch.equal(got['adv_mask'],b['adv_mask'])
    rr=rollout(j,8,recs[j]);rr.update(advantages=got['advantages'],adv_mask=got['adv_mask'],advantage_statistics=got['statistics'])
    model=ToyModel(8);other=copy.deepcopy(model)
    v,_=C.CombinedLoss(config('STANDARD_COMBINED_G8'))._policy_loss(model,rr)
    w,_=ResearchLoss(config('STANDARD_COMBINED_G8'))._policy_loss(other,dict(rr,advantages=b['advantages'],adv_mask=b['adv_mask']))
    v.backward();w.backward();assert torch.equal(v,w) and torch.equal(model.p.grad,other.p.grad)
   assert tensor_digest(C.restore(C.plain(infos[j])))==tensor_digest(infos[j])
  calls=[]
  monkeypatch.setattr(C.dist,'is_initialized',lambda:True);monkeypatch.setattr(C.dist,'get_world_size',lambda:8)
  def gather(out,value):
   full=material if value.ndim==3 else allnew;calls.append(value.ndim)
   for j in range(8):out[j].copy_(full[j:j+1])
  monkeypatch.setattr(C.dist,'all_gather',gather)
  rs=[credit.finalize(rollout(j,8,recs[j])) for j in range(8)]
  assert len(calls)==8*(2 if fallback else 1)
  ident=dict(world=8,sha256='test',schema=SCOPE_SCHEMA,recipe=config('STANDARD_COMBINED_G8'),projected_condition_identity='condition')
  runtime=runtime_for_config(TEMPLATES/'STANDARD_COMBINED_G8.yaml');runtime.credit=credit
  ss=[runtime.validate_boundary(state(8,1),r,ident) for r in rs];runtime.validate_ranks(ss)
  bad=copy.deepcopy(rs[0]);bad['combined_credit']['delta_safe'][0,0]+=.01
  with pytest.raises(ValueError):runtime.validate_boundary(state(8,1),bad,ident)
  bad=copy.deepcopy(ss);bad[1]['normalization_std']+=.01
  with pytest.raises(ValueError):runtime.validate_ranks(bad)
  monkeypatch.undo()


class MockService:
 protocol=dict(sha256='standard_CPU_mock')
 def __init__(self,rank,fallback):self.records=population(fallback)[1][rank]
 def score(self,job):
  token,log,requested=job;out=copy.deepcopy(self.records)
  for j,r in enumerate(out):r.update(token=token,log=log,requested_trajectory=requested[j].copy(),cache=dict(sha256='mockcache'),protocol_sha256=self.protocol['sha256'])
  return out


def actual_actor(runtime,cfg):
 actor,cond,_,_=make(MEMORY);actor._loss=ResearchLoss(cfg)
 with patch.object(GroupRuntime,'make_actor',return_value=(actor,None,None)):
  actor,_,_=runtime.make_actor(cfg,SimpleNamespace(),torch.device('cpu'))
 return actor,cond


def native(rank,init,root,arm):
 torch.set_num_threads(1);dist.init_process_group('gloo',init_method='file://'+init,rank=rank,world_size=8,timeout=timedelta(seconds=180))
 try:
  for fallback in ([False,True] if arm=='STANDARD_COMBINED_G8' else [False]):
   runtime=runtime_for_config(TEMPLATES/(arm+'.yaml'));cfg=config(arm);runtime.group_size=8;runtime.credit=runtime.make_credit(cfg)
   actor,cond=actual_actor(runtime,cfg)
   if arm==S.ARMS[3]:assert isinstance(actor._loss,EndpointLoss)
   generator=torch.Generator().manual_seed(seed_for('simwam_source_rank_rollout',42,rank));service=MockService(rank,fallback)
   r=runtime.collect(actor,cond.get('train','cpu'),'train','mocklog',service,generator,variant='R_CANONICAL',version=dict(run='test',update=0,adapter_sha256='same'))
   runtime.finalize_advantages(r)
   ident=dict(world=8,sha256='test',schema=SCOPE_SCHEMA,recipe=cfg,projected_condition_identity=cond.sha,score_protocol=service.protocol)
   opt=ScopeRuntime().make_optimizer(actor,cfg);actor._loss.optimizer=opt;actor._loss.learning_rate=cfg['learning_rate'];sched=actor._loss._build_scheduler('cosine',50000,1000)
   hooks=dict(boundary_validator=runtime.validate_boundary,rank_validator=runtime.validate_ranks);folder=Path(root)/str(fallback)
   digest=tensor_digest({k:v for k,v in r.items() if k!='records'})
   with patch.object(torch.cuda,'is_available',lambda:False):
    for u in range(4):
     before=tensor_digest(opt.state_dict());sch=copy.deepcopy(sched.state_dict());rng=generator.get_state().clone()
     p=checkpoint.save(folder,actor,opt,sched,generator,state(8,u),r,ident,**hooks)
     actor,cond=actual_actor(runtime,cfg);opt=ScopeRuntime().make_optimizer(actor,cfg);actor._loss.optimizer=opt
     sched=actor._loss._build_scheduler('cosine',50000,1000)
     st,r=checkpoint.restore(p,actor,opt,sched,generator,ident,**hooks)
     assert st==state(8,u) and tensor_digest(opt.state_dict())==before and sched.state_dict()==sch and torch.equal(generator.get_state(),rng)
     assert tensor_digest({k:v for k,v in r.items() if k!='records'})==digest
     total,metrics=actor(r);total.backward();params=[p for p in actor.parameters() if p.requires_grad]
     assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
     if arm==S.ARMS[3]:
      ratio=runtime.policy_ratio(actor,r)
      if u==0:assert torch.equal(ratio,torch.ones_like(ratio))
     for p in params:dist.all_reduce(p.grad);p.grad.div_(8)
     torch.nn.utils.clip_grad_norm_(params,1);opt.step();sched.step();opt.zero_grad(set_to_none=True)
    ready=state(8,0);ready.update(updates=4,next_inner=0,phase='READY')
    path=checkpoint.save(folder,actor,opt,sched,generator,ready,None,ident,**hooks)
    loaded,none=checkpoint.restore(path,actor,opt,sched,generator,ident,**hooks)
    assert loaded==ready and none is None
   if rank==0:(folder/'CPU_PASS.json').write_text(json.dumps(dict(arm=arm,fallback=fallback,world=8,boundaries=['pending0','pending1','pending2','pending3','READY4'],scheduler=sched.state_dict(),fixture='actual ScopeRecipeActor with tiny dimensions; mock Qwen/scorer')))
 finally:dist.destroy_process_group()


@pytest.mark.parametrize('arm',['STANDARD_COMBINED_G8','STANDARD_ENDPOINT_LR02_G8'])
def test_real_scope_eight_rank_pending_fourinner_fixed_schedule_native_restore(tmp_path,arm):
 torch.set_num_threads(1)
 mp.start_processes(native,args=(str(tmp_path/'gloo'),str(tmp_path/'native'),arm),nprocs=8,join=True,start_method='fork')
 assert (tmp_path/'native/False/CPU_PASS.json').is_file()
