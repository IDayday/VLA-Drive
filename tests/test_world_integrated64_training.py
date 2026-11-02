"""New actual64/G16 shape, DDP gradient mean and transactional native history."""
import copy,json
from pathlib import Path
from datetime import timedelta
from unittest.mock import patch
import pytest,torch,yaml
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from world_postcot.test_scope_rl import TinyModel
from test_world_safety_exploration_groups import components,records
from starVLA.rl.qwen_world_postcot.scope_actor import ScopeRecipeActor
from starVLA.rl.qwen_world_postcot.scope_contracts import SCOPE_SCHEMA
from starVLA.rl.qwen_world_safety_exploration.conditions import GroupMemoryConditions
from starVLA.rl.qwen_world_safety_exploration.credit import GroupCredit,canonical_components
from starVLA.rl.qwen_world_safety_exploration import integrated64_recipe as R,integrated64_schedule as S,integrated64_retention as H
from starVLA.rl.qwen_world_safety_exploration.integrated64_runtime import IntegratedRuntime,IntegratedLoss,FrozenWindowFeatures
from starVLA.rl.simwam_recipe import checkpoint
from starVLA.rl.simwam_recipe.contracts import seed_for


def config():return yaml.safe_load((R.ROOT/'configs'/f'{R.ARM}.yaml').read_text())


def test_actual1024_scale_and_pending_clocks_and_strict_recipe():
 c=config();assert R.validate_integrated_recipe(c)is c
 x=components(16).repeat(8,1,1);x[8:16,:,3]*=.13
 rewards=torch.cat([canonical_components(records(row),16)['canonical_rewards']for row in x]);material=torch.cat((rewards.double().unsqueeze(-1),x),-1)
 result=GroupCredit(16).advantages(rewards,x,all_material=material)
 assert result['statistics']['normalization_population_candidates']==1024
 assert result['statistics']['normalization_population_groups']==64
 torch.testing.assert_close(result['advantages'],(result['A_raw']/(result['A_raw'].std(unbiased=False)+1e-8)).reshape(-1).float(),atol=0,rtol=0)
 for u,scenes,candidates in ((250,4032,64512),(625,10048,160768),(6250,100032,1600512)):
  assert R.budget(u)['scene_exposures']==scenes and R.budget(u)['candidate_queries']==candidates
 bad=copy.deepcopy(c);bad['research']['integrated64']['retention']['history_share']=.5
 with pytest.raises(ValueError):R.validate_integrated_recipe(bad)

class Data:
 tokens=[f't{i}'for i in range(2048)];sha='integrated64_CPU_TRAIN'
 def observation(self,t):return dict(images={},navigation='forward',proprio=[1.+int(t[1:])*.001,0.,0.,0.,1.,0.,0.,0.])
class Stream:
 tokens=Data.tokens;logs={t:'log'+str(int(t[1:])//8)for t in tokens}
 def at(self,cursor,rank):return self.tokens[(cursor+rank)%len(self.tokens)],dict(cursor=cursor,rank=rank)
class Service:
 protocol=dict(sha256='integrated64_CPU_mock_canonical')
 def __init__(self):self.history=set();self.counts=[]
 def score(self,job):
  t,log,requested=job;self.counts.append(len(requested))
  if len(requested)==16:out=records(components(16)[int(t[1:])%8])
  else:
   x=torch.ones(16,4,dtype=torch.float64);x[:,3]=.98
   if t in self.history:x[:,2]=0
   out=records(x)[:1];m=out[0]['metrics'];m['comfort']=1.;m['score']=(5*m['ego_progress']+5*m['time_to_collision_within_bound']+2)/12;out[0]['readouts']['canonical_PDMS']=m['score']
  for j,r in enumerate(out):r.update(token=t,log=log,requested_trajectory=requested[j].copy(),cache=dict(sha256='CPUmockcache'),protocol_sha256=self.protocol['sha256'])
  return out

def actual(c):
 torch.manual_seed(42);model=TinyModel();a=ScopeRecipeActor(model.actor,c,scene_provider=model.scene_provider,predictive_provider=model.predictive_provider);a._loss=IntegratedLoss(c)
 base=GroupMemoryConditions(model,Data(),'IL70_integrated_CPU',a,group_size=16,processor=lambda obs:dict(current=True));conditions=FrozenWindowFeatures(base);object.__setattr__(a,'conditions',conditions)
 rt=IntegratedRuntime();rt.cfg=c;rt.actor=a;rt.conditions=conditions;rt.group_size=16;rt.credit=rt.make_credit(c);rt.stream=Stream()
 return a,rt,model

def state(u,c):
 return dict(updates=u,collections=c,next_inner=u%4,scene_cursor=64*c,raw_scene_exposures=64*c,new_candidates=1024*c,phase='READY'if u%4==0 else 'INNER_COMMITTED',RL_updates_local=u,anchor_only_updates_local=0,seconds_rollout=0.,seconds_scoring_wait=0.,seconds_scoring_cpu=0.,seconds_old_probability=0.,seconds_update=0.)
def optimizer(a,rt,c,u):
 o=rt.make_optimizer(a,c);a._loss.optimizer=o;s=a._loss._build_scheduler('cosine',12500,125)
 if u:
  for group in o.param_groups:
   group['lr']=group['initial_lr']*S.multiplier(u)
   for p in group['params']:o.state[p]=dict(step=torch.tensor(float(u)),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
  st=s.state_dict();st.update(last_epoch=u,_step_count=u+1,_last_lr=[g['lr']for g in o.param_groups]);s.load_state_dict(st)
 return o,s

def native(rank,init,folder):
 torch.set_num_threads(1);dist.init_process_group('gloo',init_method='file://'+init,rank=rank,world_size=8,timeout=timedelta(seconds=240))
 try:
  c=config();a,rt,m=actual(c);ident=dict(world=8,sha256='integrated64_CPU',schema=SCOPE_SCHEMA,recipe=c,projected_condition_identity=rt.conditions.sha,score_protocol=Service.protocol)
  st=state(248,62);st['integrated_retention']=H.new_bank(ident['sha256'],rank);st['integrated_retention'].update(windows=62,scenes=496)
  rt.bind_training_state(st,None,ident);o,s=optimizer(a,rt,c,248);g=torch.Generator().manual_seed(seed_for('simwam_source_rank_rollout',42,rank));service=Service();folder=Path(folder)
  for collection in range(3):
   if collection==2:service.history={e['token']for e in st['integrated_retention']['entries']}
   t,pos=rt.stream.at(st['scene_cursor'],rank);version=dict(run=ident['sha256'],update=st['updates'],adapter_sha256=f'CPUtheta{st["updates"]}')
   w=rt.collect(a,rt.conditions.get(t,'cpu'),t,rt.stream.logs[t],service,g,variant='R_CANONICAL',version=version);w['stream_position']=pos
   rt.validate_collection(w);rt.finalize_advantages(w);rt.commit_collection_accounting(w,st,8);st['phase']='ROLLOUT_COMMITTED';rt.on_collection_committed(w,st)
   assert w['chain'].shape==(8,16,11,8,3)and w['advantages'].numel()==128
   events=w['integrated_retention']['events'];assert sum(e['enabled']for e in events)==(8 if collection==2 else 0)
   assert sum(e['fresh_witness']is not None for e in events)==(8 if collection else 0)
   assert sum(e['teacher']is not None for e in events)==(8 if collection==2 else 0)
   rt.validate_ranks(checkpoint.gather(rt.validate_boundary(st,w,ident,o,s)))
   #Reference gradient uses all8 real Scope graphs; DDP implementation accumulates
   #one at atime and syncs once. No history gate in thisfirst window.
   if collection==0:
    ref,rr,mm=actual(c);sum(ref(leaf)[0]/8 for leaf in w['samples']).backward()
    ddp=DDP(a,broadcast_buffers=False,find_unused_parameters=True)
    o.zero_grad(set_to_none=True);_,_,audit=rt.backward_update(ddp,a,w,st);assert audit['fresh_error']==0
    for p,q in zip((p for p in a.parameters()if p.requires_grad),(p for p in ref.parameters()if p.requires_grad)):
     dist.all_reduce(q.grad);q.grad.div_(8);torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=3e-5)
    del ddp,ref,rr,mm;o.zero_grad(set_to_none=True)
   for inner in range(4):
    hooks=dict(boundary_validator=rt.validate_boundary,rank_validator=rt.validate_ranks)
    before=dict(state=H.M.digest(st),rollout=H.M.digest(w),optimizer=H.M.digest(o.state_dict()),scheduler=copy.deepcopy(s.state_dict()),weights=H.M.digest({n:p for n,p in a.named_parameters()if p.requires_grad}),rng=g.get_state().clone())
    with patch.object(torch.cuda,'is_available',lambda:False):cp=checkpoint.save(folder,a,o,s,g,st,w,ident,tag=f'c{collection}_u{st["updates"]}_i{inner}',**hooks)
    a,rt,m=actual(c);o,s=optimizer(a,rt,c,0);hooks=dict(boundary_validator=rt.validate_boundary,rank_validator=rt.validate_ranks)
    with patch.object(torch.cuda,'is_available',lambda:False):st,w=checkpoint.restore(cp,a,o,s,g,ident,**hooks)
    rt.bind_training_state(st,w,ident)
    assert H.M.digest(st)==before['state']and H.M.digest(w)==before['rollout']and H.M.digest(o.state_dict())==before['optimizer']and s.state_dict()==before['scheduler']and torch.equal(g.get_state(),before['rng'])
    assert H.M.digest({n:p for n,p in a.named_parameters()if p.requires_grad})==before['weights']
    ddp=DDP(a,broadcast_buffers=False,find_unused_parameters=True);o.zero_grad(set_to_none=True);builds=a.current_memory_builds
    loss,metrics,audit=rt.backward_update(ddp,a,w,st);assert a.current_memory_builds-builds==(16 if collection==2 else 8)
    assert metrics['retention_effective_IL_share']==(.75 if collection==2 else 1.)
    assert metrics['retention_history_ODE_queries']*8==sum(e['history_scored_now']for e in w['integrated_retention']['events'])
    params=[p for p in a.parameters()if p.requires_grad];assert all(p.grad is not None and torch.isfinite(p.grad).all()for p in params)
    torch.nn.utils.clip_grad_norm_(params,1);o.step();s.step();del ddp
    st['updates']+=1;st['next_inner']+=1;st['phase']='INNER_COMMITTED'
   st['next_inner']=0;st['phase']='READY'
   hooks=dict(boundary_validator=rt.validate_boundary,rank_validator=rt.validate_ranks)
   with patch.object(torch.cuda,'is_available',lambda:False):checkpoint.save(folder,a,o,s,g,st,None,ident,tag=f'c{collection}_READY',**hooks)
  assert st['integrated_retention']['fresh_ODE_queries_local']==16
  assert st['integrated_retention']['history_slots_selected']==8 and st['integrated_retention']['history_activated']==8
  assert st['integrated_retention']['history_ODE_queries_local']==sum(n==1 for n in service.counts)-16
  #625 is another pending evaluation boundary. Synthetic priorclock/moments are
  #declared explicitly; this tests actual native material, not625 measuredsteps.
  st.update(state(624,156));st['integrated_retention'].update(windows=156,scenes=1248);o,s=optimizer(a,rt,c,624);rt.bind_training_state(st,None,ident)
  t,pos=rt.stream.at(st['scene_cursor'],rank);w=rt.collect(a,rt.conditions.get(t,'cpu'),t,rt.stream.logs[t],service,g,variant='R_CANONICAL',version=dict(run=ident['sha256'],update=624,adapter_sha256='CPUtheta624'));w['stream_position']=pos
  rt.finalize_advantages(w);rt.commit_collection_accounting(w,st,8);st['phase']='ROLLOUT_COMMITTED';rt.on_collection_committed(w,st)
  o.zero_grad(set_to_none=True);ddp=DDP(a,broadcast_buffers=False,find_unused_parameters=True);rt.backward_update(ddp,a,w,st);torch.nn.utils.clip_grad_norm_([p for p in a.parameters()if p.requires_grad],1);o.step();s.step();del ddp
  st.update(updates=625,next_inner=1,phase='INNER_COMMITTED');hooks=dict(boundary_validator=rt.validate_boundary,rank_validator=rt.validate_ranks)
  with patch.object(torch.cuda,'is_available',lambda:False):cp=checkpoint.save(folder,a,o,s,g,st,w,ident,tag='pending625',**hooks);loaded,again=checkpoint.restore(cp,a,o,s,g,ident,**hooks)
  assert H.M.digest(loaded)==H.M.digest(st)and H.M.digest(again)==H.M.digest(w)
  bad=copy.deepcopy(w);bad['samples'][2]['advantages'][0]+=.1
  with pytest.raises(ValueError):rt.validate_boundary(st,bad,ident)
  bad=copy.deepcopy(w);bad['integrated_retention']['events'][0]['history_scored_now']=not bad['integrated_retention']['events'][0]['history_scored_now']
  with pytest.raises(ValueError):rt.validate_boundary(st,bad,ident)
  if rank==0:(folder/'CPU_PASS.json').write_text(json.dumps(dict(world=8,fresh_scenes=64,group_size=16,population=1024,actual_adam_steps=13,DDP='eight real microforward/backward graphs, first7 no_sync; equal explicit8scene mean gradient',native_boundaries=['pending0','pending1','pending2 includes250','pending3','READY','pending625'],restored=['all8chains/conditions/oldprob/advantages','Adam/weights/scheduler','mainRNG','shadowbank/privateRNG/events/querycount/gates'],history=['before250 noadmission','theta252 admits8 but prior-windowteachers unavailable','theta256 eight genuinehistory slots/regressiongates','cachedscore counts independently from8loss slots'],fixture='actual ScopeRecipeActor tiny32x2; fake frozenQwen and deterministic mocked scorer; synthetic248/624 previousclock andmoment origin; NOT production52.7M qualification or performance evidence')))
 finally:dist.destroy_process_group()


def test_eight_rank_real_scope_accumulation_shadow_history_and_native_pending(tmp_path):
 torch.set_num_threads(1);mp.start_processes(native,args=(str(tmp_path/'gloo'),str(tmp_path/'native')),nprocs=8,join=True,start_method='spawn')
 assert(tmp_path/'native/CPU_PASS.json').is_file()
