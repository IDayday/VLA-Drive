"""Orthogonal composition: unchanged retention loss, authentic early LR/native."""
import copy,json
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import pytest,torch,yaml
import torch.multiprocessing as mp
import test_world_verified_retention as R
from starVLA.rl.qwen_world_safety_exploration import combined_consolidation_retention_recipe as C
from starVLA.rl.qwen_world_safety_exploration.combined_consolidation_retention_runtime import CombinedRetentionRuntime,CombinedRetentionLoss
from starVLA.rl.qwen_world_safety_exploration.retention_runtime import RetentionLoss
from starVLA.rl.qwen_world_safety_exploration.group_runtime import GroupRuntime
from starVLA.rl.qwen_world_safety_exploration.consolidation_recipe import ConsolidationSchedulerLossMixin,multiplier,PROTOCOL as SCHEDULE
from starVLA.rl.qwen_world_safety_exploration import retention_bank as B,retention_material as M

ORIGINAL_ACTUAL=R.actual
BASE_CONFIG=R.config
TEMPLATE=C.ROOT.parent/'configs/improvement_training_templates'/f'{C.ARM}.yaml'
def config():return yaml.safe_load(TEMPLATE.read_text())

def actual(cfg):
 actor,old,model=ORIGINAL_ACTUAL(BASE_CONFIG())
 #Construct the real inherited make_actor path without loading productionQwen.
 base=old.conditions.base;object.__setattr__(actor,'conditions',base)
 runtime=CombinedRetentionRuntime();runtime.cfg=cfg;runtime.group_size=8;runtime.conditions=base
 with patch.object(GroupRuntime,'make_actor',return_value=(actor,None,None)):
  actor,_,_=runtime.make_actor(cfg,SimpleNamespace(),torch.device('cpu'))
 runtime.actor=actor;runtime.credit=runtime.make_credit(cfg);runtime.stream=old.stream
 return actor,runtime,model


def test_recipe_loss_gradient_identity_scheduler_endpoints_and_native_stage_clock():
 torch.set_num_threads(1);c=config();before=copy.deepcopy(c);assert C.validate_combined_recipe(c)is c and c==before
 from starVLA.rl.qwen_world_safety_exploration.runtime import runtime_for_config
 assert type(runtime_for_config(TEMPLATE))is CombinedRetentionRuntime
 assert CombinedRetentionLoss._policy_loss is RetentionLoss._policy_loss
 assert CombinedRetentionLoss._build_scheduler is ConsolidationSchedulerLossMixin._build_scheduler
 actor,rt,_=actual(c);service=R.Service();memory=B.new_bank('retention_test',0);memory['collections']=500
 teacher=M.probe(actor,rt.conditions,'t0',rt.stream.logs['t0'],service,R.version())
 rollout=R.primary(actor,rt,service,'t0',2000);rollout['verified_retention']=B.make_payload(memory,rollout,teacher,None,None)
 params=[p for p in actor.parameters()if p.requires_grad]
 for enabled in (False,True):
  if enabled:
   memory=B.transition(memory,rollout['verified_retention'])
   with torch.no_grad():next(actor.scene_provider.parameters()).add_(.01)
   service.regress_token='t0';rollout=R.primary(actor,rt,service,'t1',2004)
   current=M.probe(actor,rt.conditions,'t0',rt.stream.logs['t0'],service,R.version(2004));fresh=M.probe(actor,rt.conditions,'t1',rt.stream.logs['t1'],service,R.version(2004));index,_=B.selection(memory,'t1')
   rollout['verified_retention']=B.make_payload(memory,rollout,fresh,current,index)
  assert rollout['verified_retention']['enabled']==enabled
  old,_=RetentionLoss(c)._policy_loss(actor,rollout);new,_=CombinedRetentionLoss(c)._policy_loss(actor,rollout)
  oldgrad=torch.autograd.grad(old,params);newgrad=torch.autograd.grad(new,params)
  assert torch.equal(old,new)and all(torch.equal(a,b)for a,b in zip(oldgrad,newgrad))
 opt=rt.make_optimizer(actor,c);actor._loss.optimizer=opt;sched=actor._loss._build_scheduler('cosine',50000,1000)
 assert sched.consolidation_protocol==SCHEDULE
 for u in (0,1000,5000,10000,20000,30000,50000):
  s=sched.state_dict();s.update(last_epoch=u,_step_count=u+1,_last_lr=[p*multiplier(u)for p in sched.base_lrs]);sched.load_state_dict(s)
  assert sched.get_last_lr()==[p*multiplier(u)for p in sched.base_lrs]
 assert multiplier(20000)==multiplier(50000)==.1
 bad=copy.deepcopy(c);bad['research']['combined_consolidation_retention']['schedule']['cosine_decay_end_updates']=50000
 with pytest.raises(ValueError):C.validate_combined_recipe(bad)
 bad=copy.deepcopy(sched.state_dict());bad['consolidation_protocol']['min_lr_ratio']=.2
 with pytest.raises(ValueError):sched.load_state_dict(bad)


def initialize(actor,rt,c,u):
 opt=rt.make_optimizer(actor,c);actor._loss.optimizer=opt;sched=actor._loss._build_scheduler('cosine',50000,1000)
 for group in opt.param_groups:
  group['lr']=group['initial_lr']*multiplier(u)
  for p in group['params']:opt.state[p]=dict(step=torch.tensor(float(u)),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
 s=sched.state_dict();s.update(last_epoch=u,_step_count=u+1,_last_lr=[g['lr']for g in opt.param_groups]);sched.load_state_dict(s)
 return opt,sched


def native(rank,init,root):
 #Reuse the existing real8rank Scope/retention/checkpoint driver, replacing only
 #the config/factory and schedule initialization; no production monkeypatching.
 with patch.object(R,'config',config),patch.object(R,'actual',actual),patch.object(R,'initialize_counter_fixture',initialize):R.native(rank,init,root)
 if rank==0:
  p=Path(root)/'CPU_PASS.json';j=json.loads(p.read_text());j.update(combination=C.CONFIG,actual_factory='CombinedRetentionRuntime',actual_scheduler='ConsolidationCosineLR',retention_loss_implementation_unchanged=True);p.write_text(json.dumps(j,indent=2))


def test_actual8rank_native_combined_pending_fourinner_retention_and_early_scheduler(tmp_path):
 torch.set_num_threads(1);mp.start_processes(native,args=(str(tmp_path/'gloo'),str(tmp_path/'native')),nprocs=8,join=True,start_method='spawn')
 assert(tmp_path/'native/CPU_PASS.json').is_file()
