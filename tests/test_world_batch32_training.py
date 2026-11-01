"""Actual multi-scene sampling, gradient mean, strict native four-inner replay."""
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
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
from world_postcot.test_scope_rl import TinyModel
from test_world_safety_exploration_groups import records,components
from starVLA.rl.qwen_world_postcot.scope_actor import ScopeRecipeActor
from starVLA.rl.qwen_world_postcot.scope_contracts import SCOPE_SCHEMA
from starVLA.rl.qwen_world_safety_exploration.conditions import GroupMemoryConditions
from starVLA.rl.qwen_world_safety_exploration.batch32_runtime import Batch32Runtime,Batch32Loss,FrozenFeatureWindow
from starVLA.rl.qwen_world_safety_exploration import batch32_recipe as B,batch32_recovery as V
from starVLA.rl.qwen_world_safety_exploration.credit import GroupCredit,canonical_components
from starVLA.rl.simwam_recipe import checkpoint
from starVLA.rl.simwam_recipe.contracts import seed_for

TEMPLATE=B.ROOT.parent/'configs/improvement_training_templates'/f'{B.ARM}.yaml'
def cfg():return yaml.safe_load(TEMPLATE.read_text())

def test_recipe_query_clock_full_window_math_and_exact_scheduler_resume():
 c=cfg();before=copy.deepcopy(c);assert B.validate_batch32_recipe(c)is c and c==before
 assert B.multiplier(0)==.001 and B.multiplier(250)==1 and B.multiplier(12500)==.1
 from starVLA.rl.qwen_world_safety_exploration.standardized_recipe import multiplier
 for u in (1,249,250,251,500,1250,2500,5000,12500):assert B.multiplier(u)==multiplier(4*u)
 p=[torch.nn.Parameter(torch.tensor(.2)),torch.nn.Parameter(torch.tensor(.7))]
 opt=torch.optim.AdamW([dict(params=[p[0]],lr=5e-5,name='action_lora'),dict(params=[p[1]],lr=5e-6,name='scene_predictive_memory')]);sched=B.Batch32CosineLR(opt)
 for u in range(1,502):
  for a in p:a.grad=torch.ones_like(a)
  opt.step();sched.step()
 assert sched.last_epoch==501
 stream=io.BytesIO();torch.save((opt.state_dict(),sched.state_dict()),stream);stream.seek(0);oo,ss=torch.load(stream,weights_only=False)
 q=[torch.nn.Parameter(a.detach().clone())for a in p];other=torch.optim.AdamW([dict(params=[q[0]],lr=5e-5,name='action_lora'),dict(params=[q[1]],lr=5e-6,name='scene_predictive_memory')]);sch=B.Batch32CosineLR(other)
 other.load_state_dict(oo);sch.load_state_dict(ss)
 assert V.material_digest(other.state_dict())==V.material_digest(opt.state_dict())and sch.state_dict()==sched.state_dict()
 for a in q:a.grad=torch.ones_like(a)
 other.step();sch.step();assert sch.last_epoch==502 and other.param_groups[0]['lr']==5e-5*B.multiplier(502)
 bad=copy.deepcopy(ss);bad['batch32_protocol']['scenes_per_rank']=1
 with pytest.raises(ValueError):sch.load_state_dict(bad)
 for key,val in [('gradient_accumulation_steps',1),('lr_warmup_steps',1000),('max_steps',50000)]:
  bad=copy.deepcopy(c);bad[key]=val
  with pytest.raises(ValueError):B.validate_batch32_recipe(bad)
 #32 genuinely different scene groups, not four copies normalized independently.
 x=components(8).repeat(4,1,1);x[8:16,:,3]*=.1
 rewards=torch.cat([canonical_components(records(y),8)['canonical_rewards']for y in x]);material=torch.cat((rewards.double().unsqueeze(-1),x),-1)
 expected=GroupCredit(8).advantages(rewards,x,all_material=material)
 assert expected['statistics']['normalization_population_candidates']==256
 raw=expected['A_raw'];torch.testing.assert_close(expected['advantages'],(raw/(raw.std(unbiased=False)+1e-8)).reshape(-1).float(),atol=0,rtol=0)


class Data:
 tokens=[f's{i}'for i in range(128)];sha='batch32_CPU_train'
 def observation(self,token):return dict(images={},navigation='forward',proprio=[1.+int(token[1:])*.001,0.,0.,0.,1.,0.,0.,0.])

class Stream:
 logs={t:'log'+str(int(t[1:])//8)for t in Data.tokens}
 def at(self,cursor,rank):
  token=Data.tokens[cursor+rank];return token,dict(cursor=cursor,rank=rank,token=token)

class Service:
 protocol=dict(sha256='batch32_CPU_mock_canonical')
 def score(self,job):
  token,log,requested=job;out=records(components(8)[int(token[1:])%8])
  for j,r in enumerate(out):r.update(token=token,log=log,requested_trajectory=requested[j].copy(),cache=dict(sha256='mockcache'),protocol_sha256=self.protocol['sha256'])
  return out


def actor_runtime(c):
 torch.manual_seed(42);model=TinyModel();actor=ScopeRecipeActor(model.actor,c,scene_provider=model.scene_provider,predictive_provider=model.predictive_provider)
 actor._loss=Batch32Loss(c)
 base=GroupMemoryConditions(model,Data(),'IL70_CPU_fixture',actor,group_size=8,processor=lambda observations:dict(current=True))
 conditions=FrozenFeatureWindow(base);object.__setattr__(actor,'conditions',conditions)
 runtime=Batch32Runtime();runtime.cfg=c;runtime.actor=actor;runtime.conditions=conditions;runtime.group_size=8;runtime.credit=runtime.make_credit(c);runtime.stream=Stream()
 return actor,runtime,model

def state(u):return dict(updates=u,collections=1,next_inner=u%4,scene_cursor=32,raw_scene_exposures=32,new_candidates=256,
 phase='READY'if u==4 else 'ROLLOUT_COMMITTED'if u==0 else 'INNER_COMMITTED',RL_updates_local=u,anchor_only_updates_local=0,
 seconds_rollout=0.,seconds_scoring_wait=0.,seconds_scoring_cpu=0.,seconds_old_probability=0.,seconds_update=0.)


def test_real_registered_stream_binding_and_four_token_collection_entry():
 c=cfg();runtime=Batch32Runtime();runtime.cfg=c
 args=SimpleNamespace(manifest='/mnt/project/qwen-world-rl-optimization-20261030/data/manifests/FULL_103288.json')
 stream=runtime.make_scene_stream(c,args,8)
 assert runtime.stream is stream and len(stream.tokens)==103288
 runtime.conditions=SimpleNamespace(get=lambda token,device:dict(token=token))
 runtime.bind_training_state(dict(scene_cursor=64),None,{})
 token,pos=stream.at(64,0);generator=torch.Generator().manual_seed(42);calls=[]
 def lowlevel(actor,condition,t,log,service,gen,**kwargs):
  calls.append((t,log,condition,kwargs['version']))
  assert gen is generator and kwargs['group_size']==8
  return dict(cond_bg=condition,chain=torch.zeros(8,11,8,3),old_logp=torch.zeros(8,3),token=t,log=log,
   rewards=torch.zeros(1,8),records=[],seconds=dict(rollout=0.,scoring_wait=0.,old_probability=0.,scoring_cpu=0.))
 version=dict(run='entry_CPU',update=8,adapter_sha256='same')
 with patch('starVLA.rl.qwen_world_safety_exploration.batch32_runtime.collect_group',side_effect=lowlevel):
  got=runtime.collect(SimpleNamespace(device=torch.device('cpu')),dict(token=token),token,stream.logs[token],Service(),generator,variant='R_CANONICAL',version=version)
 expected=[stream.at(64+8*b,0)for b in range(4)]
 assert got['tokens']==[t for t,p in expected]and got['window_positions']==[p for t,p in expected]
 assert len(set(got['tokens']))==4 and all(v==version for t,l,c,v in calls)and len(calls)==4


def native(rank,init,root):
 torch.set_num_threads(1);dist.init_process_group('gloo',init_method='file://'+init,rank=rank,world_size=8,timeout=timedelta(seconds=180))
 try:
  c=cfg();actor,runtime,model=actor_runtime(c);generator=torch.Generator().manual_seed(seed_for('simwam_source_rank_rollout',42,rank))
  identity=dict(world=8,sha256='batch32_test',schema=SCOPE_SCHEMA,recipe=c,projected_condition_identity=runtime.conditions.sha,score_protocol=Service.protocol)
  fresh=state(0);fresh.update(collections=0,scene_cursor=0,raw_scene_exposures=0,new_candidates=0,phase='READY')
  runtime.bind_training_state(fresh,None,identity);token,pos=runtime.stream.at(0,rank)
  r=runtime.collect(actor,runtime.conditions.get(token,torch.device('cpu')),token,runtime.stream.logs[token],Service(),generator,variant='R_CANONICAL',version=dict(run=identity['sha256'],update=0,adapter_sha256='same'))
  r['stream_position']=pos;runtime.finalize_advantages(r);runtime.commit_collection_accounting(r,fresh,8)
  fresh['phase']='ROLLOUT_COMMITTED'
  assert fresh['scene_cursor']==32 and fresh['new_candidates']==256 and r['advantages'].numel()==32
  runtime.validate_ranks(checkpoint.gather(runtime.validate_boundary(fresh,r,identity)))
  assert model.qwen.calls==4 and len(runtime.conditions.cache)==4
  #Independent explicit scene-mean gradient on the genuine ScopeActor.
  reference,rr,othermodel=actor_runtime(c)
  total=sum(reference(s)[0]/4 for s in r['samples']);total.backward()
  opt=runtime.make_optimizer(actor,c);actor._loss.optimizer=opt;sched=actor._loss._build_scheduler('cosine',12500,250)
  ddp=DDP(actor,broadcast_buffers=False,find_unused_parameters=True,static_graph=False,bucket_cap_mb=25)
  opt.zero_grad(set_to_none=True);_,_,audit=runtime.backward_update(ddp,actor,r,state(0))
  assert audit['fresh_error']==0 and actor.current_memory_builds==12 #4sampling+4oldp+4currentloss
  for p,q in zip((p for p in actor.parameters()if p.requires_grad),(p for p in reference.parameters()if p.requires_grad)):
   dist.all_reduce(q.grad);q.grad.div_(8)
   torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=3e-5)
  opt.zero_grad(set_to_none=True)
  hooks=dict(boundary_validator=runtime.validate_boundary,rank_validator=runtime.validate_ranks)
  digest=V.material_digest(r);path=Path(root)
  with patch.object(torch.cuda,'is_available',lambda:False):
   for u in range(4):
    old=V.material_digest(opt.state_dict());sch=copy.deepcopy(sched.state_dict());rng=generator.get_state().clone()
    weights=V.material_digest({n:p for n,p in actor.named_parameters()if p.requires_grad})
    cp=checkpoint.save(path,actor,opt,sched,generator,state(u),r,identity,**hooks)
    # Actual fresh actor + empty frozen-feature cache; restore all four conditions.
    actor,runtime,model=actor_runtime(c);opt=runtime.make_optimizer(actor,c);actor._loss.optimizer=opt;sched=actor._loss._build_scheduler('cosine',12500,250)
    hooks=dict(boundary_validator=runtime.validate_boundary,rank_validator=runtime.validate_ranks)
    st,r=checkpoint.restore(cp,actor,opt,sched,generator,identity,**hooks)
    assert st==state(u)and V.material_digest(opt.state_dict())==old and sched.state_dict()==sch and torch.equal(generator.get_state(),rng)
    assert V.material_digest({n:p for n,p in actor.named_parameters()if p.requires_grad})==weights
    assert torch.equal(generator.get_state(),r['generator_states'][-1])
    assert V.material_digest(r)==digest
    ddp=DDP(actor,broadcast_buffers=False,find_unused_parameters=True,static_graph=False,bucket_cap_mb=25)
    opt.zero_grad(set_to_none=True);builds=actor.current_memory_builds;runtime.backward_update(ddp,actor,r,st)
    assert actor.current_memory_builds-builds==4 and model.qwen.calls==4
    assert all(p.grad is not None and torch.isfinite(p.grad).all()for p in actor.parameters()if p.requires_grad)
    torch.nn.utils.clip_grad_norm_([p for p in actor.parameters()if p.requires_grad],1);opt.step();sched.step()
   cp=checkpoint.save(path,actor,opt,sched,generator,state(4),None,identity,**hooks)
   st,none=checkpoint.restore(cp,actor,opt,sched,generator,identity,**hooks);assert st==state(4)and none is None
  #Saved-material contamination is rejected before any restore is applied.
  bad=copy.deepcopy(r);bad['samples'][1]['advantages'][0]+=.1
  with pytest.raises(ValueError):runtime.validate_boundary(state(3),bad,identity)
  bad=copy.deepcopy(r);bad['generator_states'][1][0]^=1
  with pytest.raises(ValueError):runtime.validate_boundary(state(3),bad,identity)
  if rank==0:(path/'CPU_PASS.json').write_text(json.dumps(dict(world=8,scenes=32,candidates=256,boundaries=['pending0','pending1','pending2','pending3','READY4'],scheduler=sched.state_dict(),DDP='actual8rank four no_sync microforwards/backwards, equal explicit scene-mean gradient',fixture='actual ScopeRecipeActor tiny dimensions; fake frozen Qwen and deterministic mocked scoring; NOT production52.7M qualification')))
 finally:dist.destroy_process_group()


def test_actual_eight_rank_four_scene_ddp_and_native_pending_fourinner_restore(tmp_path):
 torch.set_num_threads(1);mp.start_processes(native,args=(str(tmp_path/'gloo'),str(tmp_path/'native')),nprocs=8,join=True,start_method='fork')
 assert (tmp_path/'native/CPU_PASS.json').is_file()
