"""Independent early-decay schedule and realScope native4inner CPU recovery."""
import copy
import io
import json
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

from starVLA.rl.qwen_world_safety_exploration import consolidation_recipe as C
from starVLA.rl.qwen_world_safety_exploration.consolidation_runtime import consolidation_runtime
from starVLA.rl.qwen_world_safety_exploration.recovery import tensor_digest
from starVLA.rl.qwen_world_postcot.scope_runtime import ScopeRuntime
from starVLA.rl.qwen_world_postcot.scope_contracts import SCOPE_SCHEMA
from starVLA.rl.simwam_recipe import checkpoint
from starVLA.rl.simwam_recipe.contracts import seed_for
from test_world_safety_exploration_groups import state
from test_world_standardized_training import actual_actor, MockService, scalar_optimizer

BASE=Path('/mnt/project/qwen-world-safety-exploration-20261031/standard_training_20261101/configs/SAFETY_G8_CONTROL.yaml')
def config():return C.build_recipe(yaml.safe_load(BASE.read_text()))


def test_recipe_preserves_science_and_rejects_horizon_family_or_scope_change():
 base=yaml.safe_load(BASE.read_text());before=copy.deepcopy(base);cfg=config()
 assert base==before and C.validate_consolidation_recipe(cfg)is cfg
 assert cfg['grpo']==base['grpo'] and cfg['logical_budget']==base['logical_budget']
 assert cfg['max_steps']==50000 and cfg['research']['campaign_root']==str(C.ROOT)
 assert 'standardized_training'not in cfg['research']
 assert consolidation_runtime(cfg).make_credit(cfg).group_size==8
 for field,val in [('max_steps',20000),('lr_warmup_steps',100),('learning_rate',1e-5),('adaptation','TIME_MOD_LORA')]:
  bad=copy.deepcopy(cfg);bad[field]=val
  with pytest.raises(ValueError):C.validate_consolidation_recipe(bad)
 for key,val in [('cosine_decay_end_updates',50000),('horizon_updates',20000),('min_lr_ratio',.01)]:
  bad=copy.deepcopy(cfg);bad['research']['consolidation_training'][key]=val
  with pytest.raises(ValueError):C.validate_consolidation_recipe(bad)
 bad=copy.deepcopy(cfg);bad['research']['native_fork']={}
 with pytest.raises(ValueError):C.validate_consolidation_recipe(bad)


def test_two_group_schedule_warmup_decay_floor_and_real_adam_stage_restore():
 assert C.multiplier(0)==.001 and C.multiplier(999)==pytest.approx(.999001)
 assert C.multiplier(1000)==1 and C.multiplier(20000)==C.multiplier(30000)==C.multiplier(50000)==.1
 assert C.multiplier(1000)>C.multiplier(10000)>C.multiplier(19999)>.1
 for value in (-1,50001,True,1000.):
  with pytest.raises(ValueError):C.multiplier(value)
 ps,opt=scalar_optimizer([5e-5,5e-6]);sched=C.ConsolidationCosineLR(opt)
 assert sched.last_epoch==0 and opt.param_groups[0]['lr']==5e-5*.001
 for u in range(1,20003):
  for p in ps:p.grad=torch.ones_like(p)
  opt.step();sched.step()
  assert [g['lr']for g in opt.param_groups]==[v*C.multiplier(u)for v in(5e-5,5e-6)]
  if u in (500,1000,10000,20000):
   stream=io.BytesIO();torch.save(dict(opt=opt.state_dict(),sched=sched.state_dict()),stream);stream.seek(0);saved=torch.load(stream,weights_only=False)
   pp,oo=scalar_optimizer([5e-5,5e-6]);ss=C.ConsolidationCosineLR(oo);oo.load_state_dict(saved['opt']);ss.load_state_dict(saved['sched'])
   assert tensor_digest(oo.state_dict())==tensor_digest(opt.state_dict()) and ss.state_dict()==sched.state_dict()
   for p in pp:p.grad=torch.ones_like(p)
   oo.step();ss.step();assert ss.last_epoch==u+1
   assert [g['lr']for g in oo.param_groups]==[v*C.multiplier(u+1)for v in(5e-5,5e-6)]
 for k,val in [('horizon_updates',20000),('cosine_decay_end_updates',50000)]:
  bad=copy.deepcopy(sched.state_dict());bad['consolidation_protocol'][k]=val
  with pytest.raises(ValueError):sched.load_state_dict(bad)
 bad=copy.deepcopy(sched.state_dict());bad['_last_lr'][1]=bad['_last_lr'][0]
 with pytest.raises(ValueError):sched.load_state_dict(bad)
 bad=copy.deepcopy(sched.state_dict());bad['fixed_protocol']=bad.pop('consolidation_protocol')
 with pytest.raises(ValueError):sched.load_state_dict(bad)


def native(rank,init,root):
 torch.set_num_threads(1);dist.init_process_group('gloo',init_method='file://'+init,rank=rank,world_size=8,timeout=timedelta(seconds=180))
 try:
  cfg=config();runtime=consolidation_runtime(cfg);runtime.group_size=8;runtime.credit=runtime.make_credit(cfg)
  actor,cond=actual_actor(runtime,cfg);generator=torch.Generator().manual_seed(seed_for('simwam_source_rank_rollout',42,rank));service=MockService(rank,False)
  rollout=runtime.collect(actor,cond.get('train','cpu'),'train','mocklog',service,generator,variant='R_CANONICAL',version=dict(run='test',update=0,adapter_sha256='same'))
  runtime.finalize_advantages(rollout)
  ident=dict(world=8,sha256='test',schema=SCOPE_SCHEMA,recipe=cfg,projected_condition_identity=cond.sha,score_protocol=service.protocol)
  opt=ScopeRuntime().make_optimizer(actor,cfg);actor._loss.optimizer=opt;sched=actor._loss._build_scheduler('cosine',50000,1000)
  hooks=dict(boundary_validator=runtime.validate_boundary,rank_validator=runtime.validate_ranks)
  digest=tensor_digest({k:v for k,v in rollout.items()if k!='records'})
  with patch.object(torch.cuda,'is_available',lambda:False):
   for u in range(4):
    moments=tensor_digest(opt.state_dict());saved_sched=copy.deepcopy(sched.state_dict());saved_rng=generator.get_state().clone()
    path=checkpoint.save(root,actor,opt,sched,generator,state(8,u),rollout,ident,**hooks)
    actor,cond=actual_actor(runtime,cfg);opt=ScopeRuntime().make_optimizer(actor,cfg);actor._loss.optimizer=opt;sched=actor._loss._build_scheduler('cosine',50000,1000)
    st,rollout=checkpoint.restore(path,actor,opt,sched,generator,ident,**hooks)
    assert st==state(8,u) and tensor_digest(opt.state_dict())==moments and sched.state_dict()==saved_sched and torch.equal(generator.get_state(),saved_rng)
    assert tensor_digest({k:v for k,v in rollout.items()if k!='records'})==digest
    total,metrics=actor(rollout);total.backward();params=[p for p in actor.parameters()if p.requires_grad]
    assert all(p.grad is not None and torch.isfinite(p.grad).all()for p in params)
    for p in params:dist.all_reduce(p.grad);p.grad.div_(8)
    torch.nn.utils.clip_grad_norm_(params,1);opt.step();sched.step();opt.zero_grad(set_to_none=True)
   ready=state(8,0);ready.update(updates=4,next_inner=0,phase='READY')
   path=checkpoint.save(root,actor,opt,sched,generator,ready,None,ident,**hooks);loaded,none=checkpoint.restore(path,actor,opt,sched,generator,ident,**hooks)
   assert loaded==ready and none is None
  summary=runtime.validate_boundary(ready,None,ident,opt,sched)
  bad=copy.deepcopy(summary);bad['consolidation_training']['protocol']['cosine_decay_end_updates']=50000
  with pytest.raises(ValueError):runtime.validate_ranks([summary]*7+[bad])
  if rank==0:Path(root,'CPU_PASS.json').write_text(json.dumps(dict(world=8,boundaries=['pending0','pending1','pending2','pending3','READY4'],protocol=C.PROTOCOL,actual_fixture='Tiny realScope actor; mocked frozen observations/scorer; no modelweights/images/GPU')))
 finally:dist.destroy_process_group()


def test_real_scope_eight_rank_four_inner_native_recovery(tmp_path):
 torch.set_num_threads(1);mp.start_processes(native,args=(str(tmp_path/'gloo'),str(tmp_path/'native')),nprocs=8,join=True,start_method='fork')
 assert(tmp_path/'native/CPU_PASS.json').is_file()
