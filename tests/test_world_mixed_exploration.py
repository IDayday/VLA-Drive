"""True per-row kernels, actualBF16 means and strict eight-rank native replay."""
import copy,json
from pathlib import Path
from datetime import timedelta
from unittest.mock import patch
import pytest,torch,yaml
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from world_postcot.test_scope_rl import TinyModel
from test_world_safety_exploration_groups import records,components,state
from starVLA.rl.qwen_world_postcot.scope_actor import ScopeRecipeActor
from starVLA.rl.qwen_world_postcot.scope_contracts import SCOPE_SCHEMA
from starVLA.rl.qwen_world_safety_exploration.conditions import GroupMemoryConditions
from starVLA.rl.qwen_world_safety_exploration import mixed_exploration_recipe as C
from starVLA.rl.qwen_world_safety_exploration.mixed_exploration_actor import MixedScopeActor,MixedLoss
from starVLA.rl.qwen_world_safety_exploration.mixed_exploration_runtime import MixedRuntime
from starVLA.rl.qwen_world_safety_exploration.recovery import tensor_digest
from starVLA.rl.qwen_world_research_algorithms.policy import policy_gradient_objective
from starVLA.rl.simwam_recipe import checkpoint

TEMPLATE=C.CAMPAIGN/'configs/improvement_training_templates'/f'{C.ARM}.yaml'
def cfg():return yaml.safe_load(TEMPLATE.read_text())
class Data:
 tokens=['train'];sha='mixed_CPU_TRAIN'
 def observation(self,token):return dict(images={},navigation='forward',proprio=[1.,0.,0.,0.,1.,0.,0.,0.])
class Service:
 protocol=dict(sha256='mixed_CPU_mock')
 def __init__(self,rank=0):self.rank=rank
 def score(self,job):
  token,log,requested=job;out=records(components(8)[self.rank])
  for i,r in enumerate(out):r.update(token=token,log=log,requested_trajectory=requested[i].copy(),cache=dict(sha256='mockcache'),protocol_sha256=self.protocol['sha256'])
  return out

def actual(c,mixed=True):
 torch.manual_seed(42);model=TinyModel();klass=MixedScopeActor if mixed else ScopeRecipeActor
 actor=klass(model.actor,c,scene_provider=model.scene_provider,predictive_provider=model.predictive_provider)
 if mixed:actor._loss=MixedLoss(c)
 cond=GroupMemoryConditions(model,Data(),'IL70_CPUfixture',actor,group_size=8,processor=lambda obs:dict(current=True));object.__setattr__(actor,'conditions',cond)
 rt=MixedRuntime();rt.cfg=c;rt.actor=actor;rt.conditions=cond;rt.group_size=8;rt.credit=rt.make_credit(c)
 return actor,cond,rt


def test_actual_BF16_fixedstrata_sample_replay_std_loss_and_legacy_isolation():
 torch.set_num_threads(1);c=cfg();assert C.validate_mixed_recipe(c)is c
 actor,cond,rt=actual(c);base,bc,_=actual(c,False)
 g=torch.Generator().manual_seed(91);other=torch.Generator().manual_seed(91)
 r=rt.collect(actor,cond.get('train','cpu'),'train','mocklog',Service(),g,variant='R_CANONICAL',version=dict(run='test',update=0,adapter_sha256='same'))
 with torch.no_grad():oldchain,*_,oldindices=base.sample_action_chain(bc.get('train','cpu'),8,generator=other)
 assert isinstance(base,ScopeRecipeActor)and not isinstance(base,MixedScopeActor)and oldindices.tolist()==[7,8,9]
 assert torch.equal(g.get_state(),other.get_state())and torch.equal(r['chain'][:4],oldchain[:4])
 assert r['sampled_recomputed_mean_max_abs']==0 and torch.equal(r['mixed_sampling']['mean'],r['old_probability']['mean'])
 assert r['stoch_idx'].tolist()==C.ROWS and r['old_probability']['std'].shape==(8,3)
 fresh=actor.action_chain_logprobs(r['cond_bg'],r['chain'],r['timesteps'],r['deltas'],stoch_idx=r['stoch_idx'])
 assert torch.equal((fresh-r['old_logp']).exp(),torch.ones(8,3))
 for b in range(2):
  for q,k in enumerate(C.ROWS[b*4]):
   _,std=actor._sde_mean_std(r['chain'][b*4:b*4+4,k],torch.zeros(4,8,3),float(r['timesteps'][k])/1000,float(r['deltas'][k]))
   assert torch.equal(r['old_probability']['std'][b*4:b*4+4,q],torch.full((4,),std))
 rt.finalize_advantages(r)
 #One-process directed math test is not a formal normalization claim.
 total,metrics=actor(r)
 now,terms=actor.action_chain_logprobs(r['cond_bg'],r['chain'],r['timesteps'],r['deltas'],stoch_idx=r['stoch_idx'],return_terms=True)
 with torch.no_grad():_,ref=actor.action_chain_logprobs(r['cond_bg'],r['chain'],r['timesteps'],r['deltas'],stoch_idx=r['stoch_idx'],use_ref=True,return_terms=True)
 adv=r['advantages'][:,None].expand(8,3).clamp(-5,5)
 pg,_=policy_gradient_objective(now,r['old_logp'],adv,r['adv_mask'],clip_range=.005)
 w=actor.kl_weight_matrix(horizon=8,action_dim=3,device='cpu',dtype=torch.float32)
 anchor=(((terms['mean']-ref['mean'])**2*w).sum((2,3))/w.sum()).mean()
 assert torch.equal(total,pg+50*anchor)
 total.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all()for p in actor.parameters()if p.requires_grad)
 assert actor._probability_audit['current']['mean'].shape==(8,3,8,3)
 bad=copy.deepcopy(c);bad['research']['precision_training']={}
 with pytest.raises(ValueError):C.validate_mixed_recipe(bad)
 bad=copy.deepcopy(r);bad['stoch_idx'][4,0]=7
 with pytest.raises(ValueError):actor(bad)


def native(rank,init,root):
 torch.set_num_threads(1);dist.init_process_group('gloo',init_method='file://'+init,rank=rank,world_size=8,timeout=timedelta(seconds=180))
 try:
  c=cfg();actor,cond,rt=actual(c);gen=torch.Generator().manual_seed(100+rank)
  r=rt.collect(actor,cond.get('train','cpu'),'train','mocklog',Service(rank),gen,variant='R_CANONICAL',version=dict(run='test',update=0,adapter_sha256='same'));rt.finalize_advantages(r)
  ident=dict(world=8,sha256='test',schema=SCOPE_SCHEMA,recipe=c,projected_condition_identity=cond.sha,score_protocol=Service.protocol)
  opt=rt.make_optimizer(actor,c);actor._loss.optimizer=opt;sched=actor._loss._build_scheduler('cosine',50000,1000)
  path=Path(root);hooks=dict(boundary_validator=rt.validate_boundary,rank_validator=rt.validate_ranks);digest=tensor_digest({k:v for k,v in r.items()if k!='records'})
  for u in range(4):
   before=tensor_digest(opt.state_dict());sch=copy.deepcopy(sched.state_dict());rng=gen.get_state().clone();weights=tensor_digest({n:p for n,p in actor.named_parameters()if p.requires_grad})
   with patch.object(torch.cuda,'is_available',lambda:False):cp=checkpoint.save(path,actor,opt,sched,gen,state(8,u),r,ident,**hooks)
   actor,cond,rt=actual(c);opt=rt.make_optimizer(actor,c);actor._loss.optimizer=opt;sched=actor._loss._build_scheduler('cosine',50000,1000);hooks=dict(boundary_validator=rt.validate_boundary,rank_validator=rt.validate_ranks)
   with patch.object(torch.cuda,'is_available',lambda:False):st,r=checkpoint.restore(cp,actor,opt,sched,gen,ident,**hooks)
   assert st==state(8,u)and tensor_digest(opt.state_dict())==before and sched.state_dict()==sch and torch.equal(gen.get_state(),rng)
   assert weights==tensor_digest({n:p for n,p in actor.named_parameters()if p.requires_grad})and digest==tensor_digest({k:v for k,v in r.items()if k!='records'})
   ddp=DDP(actor,broadcast_buffers=False,find_unused_parameters=True,static_graph=False,bucket_cap_mb=25)
   opt.zero_grad(set_to_none=True);build=actor.current_memory_builds;loss,metrics=ddp(r)
   if u==0:
    rho=(actor._probability_audit['current']['source_clamped_logprob'].mean((-1,-2))-r['old_logp']).exp();assert torch.equal(rho,torch.ones(8,3))
   assert actor.current_memory_builds-build==1 and all(f'stepordinal{q}'in ' '.join(metrics)for q in range(3))
   loss.backward();params=[p for p in actor.parameters()if p.requires_grad];assert all(p.grad is not None and torch.isfinite(p.grad).all()for p in params)
   torch.nn.utils.clip_grad_norm_(params,1);opt.step();sched.step()
  ready=state(8,0);ready.update(updates=4,next_inner=0,phase='READY')
  with patch.object(torch.cuda,'is_available',lambda:False):cp=checkpoint.save(path,actor,opt,sched,gen,ready,None,ident,**hooks);got,none=checkpoint.restore(cp,actor,opt,sched,gen,ident,**hooks)
  assert got==ready and none is None
  bad=copy.deepcopy(r);bad['mixed_sampling']['raw_innovations'][0,0,0,0]+=.1
  with pytest.raises(ValueError):rt.validate_boundary(state(8,3),bad,ident)
  bad=copy.deepcopy(r);bad['old_probability']['std'][4,0]=bad['old_probability']['std'][0,0]
  with pytest.raises(ValueError):rt.validate_boundary(state(8,3),bad,ident)
  if rank==0:(path/'CPU_PASS.json').write_text(json.dumps(dict(world=8,matrix=C.ROWS,native=['pending0','pending1','pending2','pending3','READY4'],fresh_ratio_error=0,sampled_recomputed_BF16mean_error=0,queries_perrank=8,optimizer='AdamW two groups, fixed50k/warm1000cos',fixture='actualScopeRecipeActor tiny dimensions, FakeQwen/mockcanonical, actual8rankCPU DDP; not productionGPU/52.7M efficacy')))
 finally:dist.destroy_process_group()


def test_actual_eight_rank_DDP_native_matrix_fourinner_restore(tmp_path):
 torch.set_num_threads(1);mp.start_processes(native,args=(str(tmp_path/'gloo'),str(tmp_path/'native')),nprocs=8,join=True,start_method='spawn')
 assert(tmp_path/'native/CPU_PASS.json').is_file()
