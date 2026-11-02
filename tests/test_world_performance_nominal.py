"""True Scope/B1 deployment arithmetic, teacher-only G16, calibrated budget."""
import copy,io
import numpy as np
import pytest,torch
from world_postcot.test_scope_rl import TinyModel
from test_world_safety_exploration_groups import cfg as group_cfg,records
from starVLA.rl.qwen_world_postcot.scope_actor import ScopeRecipeActor
from starVLA.rl.qwen_world_safety_exploration.conditions import GroupMemoryConditions
from starVLA.rl.qwen_world_safety_exploration.group_runtime import collect_group
from starVLA.rl.qwen_world_safety_exploration.credit import GroupCredit,canonical_components
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss
from starVLA.model.modules.qwen_world_postcot.flow import CleanXFlowPolicy
from starVLA.rl.qwen_world_performance import nominal_improvement as N,ode_target as O,adaptive_group as A
from starVLA.rl.simwam_recipe import checkpoint

class Data:
 tokens=['t0'];sha='CPU_TRAIN_fixture'
 def observation(self,t):return dict(images={},navigation='forward',proprio=[1.,0.,0.,0.,1.,0.,0.,0.])
class Service:
 protocol=dict(sha256='CPU_fixture_canonical')
 def __init__(self,primary_high=True,extra_high=True,nominal_good=False,nominal_ep=.98,primary_ep=(.8,1.)):
  self.primary_high=primary_high;self.extra_high=extra_high;self.nominal_good=nominal_good;self.nominal_ep=nominal_ep;self.primary_ep=primary_ep;self.group_calls=0;self.counts=[]
 def score(self,job):
  token,log,requested=job;n=len(requested);self.counts.append(n);x=torch.ones(n,4,dtype=torch.float64)
  if n==1:
   x[:,3]=self.nominal_ep
   if not self.nominal_good:x[:,2]=0
  else:
   self.group_calls+=1;x[:,3]=torch.linspace(*self.primary_ep,n,dtype=torch.float64)
   if not(self.primary_high if self.group_calls==1 else self.extra_high):x[:,0]=0
  out=records(x)
  for j,r in enumerate(out):
   m=r['metrics'];m['comfort']=np.float64(1);m['score']=np.float64(m['no_at_fault_collisions']*m['drivable_area_compliance']*(5*m['ego_progress']+5*m['time_to_collision_within_bound']+2)/12);r['readouts']['canonical_PDMS']=m['score']
   r.update(token=token,log=log,requested_trajectory=requested[j].copy(),cache=dict(sha256='fixture_cache'),protocol_sha256=self.protocol['sha256'])
  return out
class Capture(ResearchLoss):
 def record_policy_components(self,pg,anchor):self.pg_component=pg;self.weighted_anchor_component=anchor
class Loss(N.NominalLossMixin,Capture):pass

def actual(device='cpu'):
 torch.manual_seed(42);c=group_cfg(16);m=TinyModel().to(device);m.qwen.features=checkpoint.device_tree(m.qwen.features,torch.device(device));a=ScopeRecipeActor(m.actor,c,scene_provider=m.scene_provider,predictive_provider=m.predictive_provider);a._loss=Capture(c)
 conditions=GroupMemoryConditions(m,Data(),'IL70_CPUfixture',a,group_size=16,processor=lambda obs:dict(current=True));object.__setattr__(a,'conditions',conditions)
 #A changedcurrent trainableMemory, withthe realoriginalfrozenILreference intact.
 with torch.no_grad():next(a.scene_provider.parameters()).add_(.01)
 return a,conditions,c

def primary(a,conditions,service):
 v=dict(run='own_run_CPU',update=1200,adapter_sha256='same_theta')
 r=collect_group(a,conditions.get('t0',a.device),'t0','TRAIN_log',service,torch.Generator().manual_seed(42),group_size=16,variant='R_CANONICAL',version=v)
 mat=canonical_components(r['records'],16);pop=torch.cat((r['rewards'].double().unsqueeze(-1),mat['components']),-1).repeat(8,1,1)
 s=GroupCredit(16).advantages(r['rewards'],mat['components'],all_material=pop)
 r.update(advantages=s['advantages'],adv_mask=s['adv_mask'],global_std=s['global_std'],raw_group_std=s['raw_group_std'],advantage_statistics=s['statistics'])
 return r


def test_actual_B1_finalhead_current_memory_gradients_and_beta_budget():
 torch.set_num_threads(1);a,conditions,c=actual();service=Service();r=primary(a,conditions,service);rng=torch.get_rng_state().clone()
 nominal=O.probe_nominal(a,conditions,'t0','TRAIN_log',service,r['behavior_version']);assert torch.equal(rng,torch.get_rng_state())and service.counts==[16,1]
 raw=O.condition_for_batch(a,r['cond_bg'],1)
 with a._context_call(raw)as condition:
  prepared=a._prepare(condition,False)
  with torch.autocast('cpu',dtype=torch.bfloat16):deployment=CleanXFlowPolicy(a.action_expert).sample_ode(prepared,nominal['chain'][:,0])
 assert torch.equal(deployment,nominal['clean'])
 assert torch.equal(O.predict_clean_x(a,r['cond_bg'],nominal['last_state'],nominal['last_sigma']),nominal['clean'])
 saved=N.digest({k:r[k]for k in ('chain','rewards','old_logp','advantages','adv_mask')})
 N.attach_material(a,r,conditions,service,exposure_epoch=.5,rank=0,nominal=nominal)
 assert r['nominal_improvement']['enabled']and r['nominal_improvement']['extra']is None and service.counts==[16,1]
 N.validate_material(r,rank=0,expected_epoch=.5)
 total,_=a(r);pg=a._loss.pg_component;anchor=a._loss.weighted_anchor_component
 before=a.current_memory_builds;cfm,ode,_=N.compute_nominal_losses(a,r);assert a.current_memory_builds-before==2
 names=[(n,p)for n,p in a.named_parameters()if p.requires_grad]
 for objective in (cfm,ode):
  gradients=torch.autograd.grad(objective,[p for n,p in names],retain_graph=True,allow_unused=True)
  for prefix in ('action_expert.','scene_provider.','predictive_provider.'):
   assert sum(float(g.norm())for(n,p),g in zip(names,gradients)if n.startswith(prefix)and g is not None)>0
 state=N.new_calibration_state();names[0][1].grad=torch.ones_like(names[0][1])*.125;prior=names[0][1].grad.clone()
 normalized,metrics=N.calibrate_components(a,pg,anchor/50,cfm,ode,state)
 assert torch.equal(names[0][1].grad,prior)and metrics['nominal_gradient_norm_measured']==1
 expected=total+.05*(50*normalized['rescue']-anchor)
 a.zero_grad(set_to_none=True);a._loss=Loss(c);a._loss.auxiliary_state=N.new_calibration_state();actual_value,got=a(r)
 torch.testing.assert_close(actual_value,expected,atol=1e-6,rtol=1e-6)
 assert got['nominal_aux_IL_share']==.95 and got['nominal_aux_rescue_share']==.05
 assert a._probability_audit['current']['mean'].shape[0]==16
 actual_value.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all()for n,p in names)
 assert N.digest({k:r[k]for k in ('chain','rewards','old_logp','advantages','adv_mask')})==saved
 data=io.BytesIO();torch.save((r,state),data);data.seek(0);rr,ss=torch.load(data,weights_only=False);N.validate_material(rr,rank=0);N.validate_calibration_state(ss)
 bad=copy.deepcopy(rr);bad['nominal_improvement']['nominal']['last_state'][0,0,0]+=.1
 with pytest.raises(ValueError):N.validate_material(bad,rank=0)
 bad=copy.deepcopy(rr);bad['behavior_version']['run']='other_run'
 with pytest.raises(ValueError):N.validate_material(bad,rank=0)


def test_teacher_only_extra_trigger_margin_and_calibration_100_500_native_clock():
 torch.set_num_threads(1);a,conditions,c=actual();service=Service(primary_high=False);r=primary(a,conditions,service);saved=N.digest({k:r[k]for k in ('chain','rewards','old_logp','advantages','adv_mask')});global_rng=torch.get_rng_state().clone();audit=a._probability_audit
 N.attach_material(a,r,conditions,service,exposure_epoch=.5,rank=0)
 p=r['nominal_improvement'];assert p['enabled']and p['teacher']['origin']=='extra_G16'and service.counts==[16,1,16]
 assert torch.equal(global_rng,torch.get_rng_state())and a._probability_audit is audit
 N.validate_material(r,rank=0);assert N.digest({k:r[k]for k in ('chain','rewards','old_logp','advantages','adv_mask')})==saved
 #An existing strict-high candidate prohibits extra sampling even whenits gain
 #does not clear.02. A true B1 alreadystrict-high disables teacher supervision.
 for svc,reason in [(Service(nominal_good=True,nominal_ep=.736,primary_ep=(.76,.76)),'teacher_gain_below_margin'),(Service(nominal_good=True),'nominal_already_strict_high')]:
  rr=primary(a,conditions,svc);N.attach_material(a,rr,conditions,svc,exposure_epoch=.5,rank=0);assert not rr['nominal_improvement']['enabled']and rr['nominal_improvement']['skip_reason']==reason and svc.counts==[16,1];N.validate_material(rr,rank=0)
 svc=Service();rr=primary(a,conditions,svc);N.attach_material(a,rr,conditions,svc,exposure_epoch=.49,rank=0);assert svc.counts==[16];N.validate_material(rr,rank=0)
 model=torch.nn.Linear(1,1,bias=False);state=N.new_calibration_state();frozen=None;norm500=None
 for i in range(502):
  x=model.weight.sum();parts,metrics=N.calibrate_components(model,x.square(),3*x,(2+i*.01)*x,(5+i*.02)*x,state)
  if i==99:frozen=(state['scale_cfm'],state['scale_ode'])
  if i>=100:assert (state['scale_cfm'],state['scale_ode'])==frozen
  if i==499:norm500=copy.deepcopy(state['last_norms'])
  if i>=500:assert metrics['nominal_gradient_norm_measured']==0 and state['last_norms']==norm500
  if i in (36,99,498,499):
   data=io.BytesIO();torch.save(state,data);data.seek(0);state=torch.load(data,weights_only=False);N.validate_calibration_state(state)
 assert state['calibration_updates']==100 and state['measured_updates']==500 and state['effective_updates']==502
 bad=copy.deepcopy(state);bad['scale_cfm']=1e5
 with pytest.raises(ValueError):N.validate_calibration_state(bad)
