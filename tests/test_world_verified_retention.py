"""Three targeted checks: trueB1/targets, loss parity, actual8rank native bank."""
import copy
import io
import json
from pathlib import Path
from datetime import timedelta
from unittest.mock import patch
import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from world_postcot.test_scope_rl import TinyModel
from test_world_safety_exploration_groups import components,records
from starVLA.model.modules.qwen_world_postcot.flow import CleanXFlowPolicy
from starVLA.rl.qwen_world_postcot.scope_actor import ScopeRecipeActor
from starVLA.rl.qwen_world_postcot.scope_contracts import SCOPE_SCHEMA
from starVLA.rl.qwen_world_safety_exploration.conditions import GroupMemoryConditions
from starVLA.rl.qwen_world_safety_exploration.group_runtime import collect_group
from starVLA.rl.qwen_world_safety_exploration import retention_recipe as C,retention_material as M,retention_bank as B
from starVLA.rl.qwen_world_safety_exploration.retention_runtime import RetentionRuntime,RetentionLoss
from starVLA.rl.qwen_world_safety_exploration.standardized_recipe import multiplier
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss
from starVLA.rl.simwam_recipe import checkpoint
from starVLA.rl.simwam_recipe.contracts import seed_for

TEMPLATE=C.CAMPAIGN/'configs/improvement_training_templates'/f'{C.ARM}.yaml'
def config():return yaml.safe_load(TEMPLATE.read_text())

class Data:
 tokens=[f't{i}'for i in range(512)];sha='retention_CPU_TRAIN'
 def observation(self,token):return dict(images={},navigation='forward',proprio=[1.+int(token[1:])*.001,0.,0.,0.,1.,0.,0.,0.])
class Stream:
 tokens=Data.tokens;logs={t:'log'+str(int(t[1:])//8)for t in tokens}
 def at(self,cursor,rank):return self.tokens[(cursor+rank)%len(self.tokens)],dict(cursor=cursor,rank=rank)
class Service:
 protocol=dict(sha256='retention_CPU_mock_canonical')
 def __init__(self):self.regress_token=None;self.counts=[]
 def score(self,job):
  token,log,requested=job;self.counts.append(len(requested))
  if len(requested)==8:out=records(components(8)[int(token[1:])%8])
  else:
   x=torch.ones(8,4,dtype=torch.float64);x[:,3]=.98
   if token==self.regress_token:x[:,2]=0
   out=records(x)[:1];out[0]['metrics']['comfort']=1.
   m=out[0]['metrics'];m['score']=(5*m['ego_progress']+5*m['time_to_collision_within_bound']+2)/12
   out[0]['readouts']['canonical_PDMS']=m['score']
  for j,r in enumerate(out):r.update(token=token,log=log,requested_trajectory=requested[j].copy(),cache=dict(sha256='mockcache'),protocol_sha256=self.protocol['sha256'])
  return out

def actual(c):
 torch.manual_seed(42);model=TinyModel();actor=ScopeRecipeActor(model.actor,c,scene_provider=model.scene_provider,predictive_provider=model.predictive_provider)
 actor._loss=RetentionLoss(c)
 base=GroupMemoryConditions(model,Data(),'IL70_fixture',actor,group_size=8,processor=lambda obs:dict(current=True))
 cond=M.FrozenPairFeatures(base);object.__setattr__(actor,'conditions',cond)
 runtime=RetentionRuntime();runtime.cfg=c;runtime.actor=actor;runtime.conditions=cond;runtime.group_size=8;runtime.credit=runtime.make_credit(c);runtime.stream=Stream()
 return actor,runtime,model

def version(u=2000):return dict(run='retention_test',update=u,adapter_sha256='CPUfixture')
def scalar_identity(c,rt):return dict(world=8,sha256='retention_test',schema=SCOPE_SCHEMA,recipe=c,projected_condition_identity=rt.conditions.sha,score_protocol=Service.protocol)

def test_true_B1_deployment_parity_source_nominal_targets_private_rng_and_reservoir():
 torch.set_num_threads(1);c=config();assert C.validate_retention_recipe(c)is c
 actor,rt,model=actual(c);service=Service();before=torch.get_rng_state().clone()
 witness=M.probe(actor,rt.conditions,'t0',rt.stream.logs['t0'],service,version())
 assert service.counts==[1]and torch.equal(before,torch.get_rng_state())
 raw=M.b1_condition(actor,rt.conditions.get('t0','cpu'));root=witness['chain'][:,0]
 with actor._context_call(raw)as cond:
  prepared=actor._prepare(cond,False)
  with torch.autocast('cpu',dtype=torch.bfloat16):expected=CleanXFlowPolicy(actor.action_expert).sample_ode(prepared,root)
 assert torch.equal(expected,witness['chain'][:,-1])
 assert all(b==1 for b in actor.prepared_context_source_batches)
 _,terms=actor.action_chain_logprobs(raw,witness['chain'],witness['timesteps'],witness['deltas'],stoch_idx=torch.tensor([7,8,9]),return_terms=True)
 assert torch.equal(terms['mean'],witness['nominal_mean'])
 allowed={t:rt.stream.logs[t]for t in rt.stream.tokens};M.validate_witness(witness,'retention_test',allowed,rt.conditions.sha,service.protocol['sha256'],require_good=True)
 bank=B.new_bank('retention_test',0);bank['collections']=500
 #Exercise the actual capacity256 boundary with detached synthetic scene IDs;
 #only reservoir/RNG algebra here, not claims of256 measured model successes.
 for i in range(256):
  e=copy.deepcopy(witness);e['token']=f'fixture{i}';e=M.sealed({k:v for k,v in e.items()if k!='witness_sha256'},'witness_sha256');bank['entries'].append(e)
 bank['verified_admissions']=256
 chosen,_=B.selection(bank,'t0');teacher=bank['entries'][chosen];current=copy.deepcopy(teacher)
 rr=dict(token='t0',behavior_version=version(2004));p=B.make_payload(bank,rr,witness,current,chosen)
 result=B.transition(bank,p);assert len(result['entries'])==256 and result['verified_admissions']==257
 assert result['reservoir_replacements']+result['reservoir_not_retained']==1
 assert M.digest(result)==M.digest(B.transition(bank,p))and torch.equal(before,torch.get_rng_state())
 bad=copy.deepcopy(p);bad['enabled']=not bad['enabled']
 with pytest.raises(ValueError):B.transition(bank,bad)
 bad=copy.deepcopy(c);bad['research']['verified_retention']['history_share']=.5
 with pytest.raises(ValueError):C.validate_retention_recipe(bad)


def primary(actor,rt,service,token,u):
 r=collect_group(actor,rt.conditions.get(token,actor.device),token,rt.stream.logs[token],service,torch.Generator().manual_seed(83),group_size=8,variant='R_CANONICAL',version=version(u))
 rt.credit.finalize(r);return r


def test_offgate_bitexact_source_loss_gradient_and_activated_anchor_budget_and_memory():
 torch.set_num_threads(1);c=config();actor,rt,model=actual(c);service=Service();r=primary(actor,rt,service,'t0',2000)
 bank=B.new_bank('retention_test',0);bank['collections']=500
 teacher=M.probe(actor,rt.conditions,'t0',rt.stream.logs['t0'],service,version())
 r['verified_retention']=B.make_payload(bank,r,teacher,None,None)
 control,cr,_=actual(c);control._loss=ResearchLoss(c)
 a,_=actor(r);b,_=control(r);a.backward();b.backward();assert torch.equal(a,b)
 for p,q in zip((p for p in actor.parameters()if p.requires_grad),(p for p in control.parameters()if p.requires_grad)):assert torch.equal(p.grad,q.grad)
 #A historical target from before a current-memory change; official scoring is
 #mocked, so this establishes gate/gradient wiring, not real physical regression.
 with torch.no_grad():
  for p in actor.scene_provider.parameters():p.add_(.01);break
 service.regress_token='t0';r=primary(actor,rt,service,'t1',2004)
 bank=B.transition(bank,B.make_payload(bank,dict(token='t0',behavior_version=version()),teacher,None,None))
 current=M.probe(actor,rt.conditions,'t0',rt.stream.logs['t0'],service,version(2004));fresh=M.probe(actor,rt.conditions,'t1',rt.stream.logs['t1'],service,version(2004))
 index,_=B.selection(bank,'t1');r['verified_retention']=B.make_payload(bank,r,fresh,current,index);assert r['verified_retention']['enabled']
 source,metrics=ResearchLoss(c)._policy_loss(actor,r);build=actor.current_memory_builds
 total,got=RetentionLoss(c)._policy_loss(actor,r)
 assert actor.current_memory_builds-build==2 and actor._probability_audit['current']['mean'].shape[0]==8
 mean=M.nominal_means(actor,teacher['condition'],teacher['chain'],teacher['timesteps'],teacher['deltas'])
 w=actor.kl_weight_matrix(horizon=8,action_dim=3,device=mean.device,dtype=mean.dtype)
 raw=(((mean-teacher['nominal_mean'])**2*w).sum((2,3))/w.sum()).mean()
 torch.testing.assert_close(total,source+.25*(50*raw-metrics['kl_loss']),atol=1e-7,rtol=1e-6)
 assert got['retention_effective_IL_share']==.75 and got['retention_effective_RL_share']==.25
 params=[(n,p)for n,p in actor.named_parameters()if p.requires_grad];grads=torch.autograd.grad(raw,[p for n,p in params],allow_unused=False)
 assert all(torch.isfinite(g).all()for g in grads)
 for prefix in ('action_expert.','scene_provider.','predictive_provider.'):assert sum(float(g.norm())for (n,p),g in zip(params,grads)if n.startswith(prefix))>0
 assert not teacher['nominal_mean'].requires_grad


def training_state(u,collections):
 i=u%4;return dict(updates=u,collections=collections,next_inner=i,scene_cursor=8*collections,raw_scene_exposures=8*collections,new_candidates=64*collections,
  phase='INNER_COMMITTED'if i else 'READY',RL_updates_local=u,anchor_only_updates_local=0,
  seconds_rollout=0.,seconds_scoring_wait=0.,seconds_scoring_cpu=0.,seconds_old_probability=0.,seconds_update=0.)


def initialize_counter_fixture(actor,rt,c,u):
 opt=rt.make_optimizer(actor,c);actor._loss.optimizer=opt;sched=actor._loss._build_scheduler('cosine',50000,1000)
 #CPU native fixture starts at the protocol admission boundary with synthetic
 #zero moments/clock, not2000 actual previous updates or production evidence.
 for group in opt.param_groups:
  group['lr']=group['initial_lr']*multiplier(u)
  for p in group['params']:opt.state[p]=dict(step=torch.tensor(float(u)),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
 s=sched.state_dict();s.update(last_epoch=u,_step_count=u+1,_last_lr=[g['lr']for g in opt.param_groups]);sched.load_state_dict(s)
 return opt,sched


def native(rank,init,folder):
 torch.set_num_threads(1);dist.init_process_group('gloo',init_method='file://'+init,rank=rank,world_size=8,timeout=timedelta(seconds=180))
 try:
  c=config();actor,rt,model=actual(c);ident=scalar_identity(c,rt);state=training_state(2000,500)
  state['retention']=B.new_bank(ident['sha256'],rank);state['retention']['collections']=500;rt.bind_training_state(state,None,ident)
  opt,sched=initialize_counter_fixture(actor,rt,c,2000);generator=torch.Generator().manual_seed(seed_for('simwam_source_rank_rollout',42,rank));service=Service()
  folder=Path(folder)
  for collection in range(2):
   if collection:service.regress_token=state['retention']['entries'][0]['token']
   token,position=rt.stream.at(state['scene_cursor'],rank)
   r=rt.collect(actor,rt.conditions.get(token,'cpu'),token,rt.stream.logs[token],service,generator,variant='R_CANONICAL',version=version(state['updates']))
   r['stream_position']=position;rt.finalize_advantages(r);state['collections']+=1;state['raw_scene_exposures']+=8;state['new_candidates']+=64;state['scene_cursor']+=8;state['phase']='ROLLOUT_COMMITTED';rt.on_collection_committed(r,state)
   assert r['verified_retention']['enabled']==bool(collection)
   hooks=dict(boundary_validator=rt.validate_boundary,rank_validator=rt.validate_ranks)
   for inner in range(4):
    st=copy.deepcopy(state);before=M.digest(opt.state_dict());sch=copy.deepcopy(sched.state_dict());rng=generator.get_state().clone();bank=M.digest(state['retention']);payload=M.digest(r)
    with patch.object(torch.cuda,'is_available',lambda:False):path=checkpoint.save(folder,actor,opt,sched,generator,state,r,ident,tag=f'c{collection}_u{state["updates"]}_inner{inner}',**hooks)
    actor,rt,model=actual(c);opt=rt.make_optimizer(actor,c);actor._loss.optimizer=opt;sched=actor._loss._build_scheduler('cosine',50000,1000);hooks=dict(boundary_validator=rt.validate_boundary,rank_validator=rt.validate_ranks)
    with patch.object(torch.cuda,'is_available',lambda:False):state,r=checkpoint.restore(path,actor,opt,sched,generator,ident,**hooks)
    rt.bind_training_state(state,r,ident)
    assert M.digest(state)==M.digest(st)and M.digest(opt.state_dict())==before and sched.state_dict()==sch and torch.equal(generator.get_state(),rng)
    assert M.digest(state['retention'])==bank and M.digest(r)==payload
    opt.zero_grad(set_to_none=True);total,metrics=actor(r);total.backward()
    params=[p for p in actor.parameters()if p.requires_grad]
    assert all(p.grad is not None and torch.isfinite(p.grad).all()for p in params)
    for p in params:dist.all_reduce(p.grad);p.grad.div_(8)
    torch.nn.utils.clip_grad_norm_(params,1);opt.step();sched.step()
    state['updates']+=1;state['next_inner']+=1;state['phase']='INNER_COMMITTED'
   state['next_inner']=0;state['phase']='READY'
   with patch.object(torch.cuda,'is_available',lambda:False):path=checkpoint.save(folder,actor,opt,sched,generator,state,None,ident,tag=f'c{collection}_READY',**hooks)
  assert service.counts==[8,1,8,1,1]
  bad=copy.deepcopy(r);bad['verified_retention']['teacher']['nominal_mean'][0,0,0,0]+=.1
  invalid=copy.deepcopy(state);invalid.update(updates=2007,next_inner=3,phase='INNER_COMMITTED')
  with pytest.raises(ValueError):rt.validate_boundary(invalid,bad,ident)
  bad=copy.deepcopy(state);bad['retention']['rng_state'][0]^=1
  #The full bank RNG is bound by the pending material's exact after digest.
  with pytest.raises(ValueError):rt.validate_boundary(dict(invalid,retention=bad['retention']),r,ident)
  if rank==0:(folder/'CPU_PASS.json').write_text(json.dumps(dict(world=8,gates=['pending0','pending1','pending2','pending3','READY'],collections=['offgate1','realB1_regression_gate1'],loss='actualScopeRecipeActor and unchangedSafetyA',fixture='tiny dimensions; mocked canonical scorer; synthetic pre2000 counter/moment origin; no production performance claim',main_queries_local=16,extra_B1_queries_local=3,restored='full bank/private RNG/Adam/schedule/fourinner material')))
 finally:dist.destroy_process_group()


def test_eight_rank_native_offgate_then_active_fourinner_bank_and_rng(tmp_path):
 torch.set_num_threads(1);mp.start_processes(native,args=(str(tmp_path/'gloo'),str(tmp_path/'native')),nprocs=8,join=True,start_method='spawn')
 assert(tmp_path/'native/CPU_PASS.json').is_file()
