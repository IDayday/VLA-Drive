"""True-B1 ODE witnesses and detached nominal-W3 targets, never PG replay."""
import copy
import hashlib
import json
import math
import time
from collections import OrderedDict
import numpy as np
import torch
import torch.distributed as dist
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.model.modules.qwen_world_postcot.flow import clean_x_step,shift_sigma
from starVLA.rl.simwam_recipe import checkpoint
from starVLA.rl.simwam_recipe.contracts import seed_for
from .recovery import tensor_digest
from .retention_recipe import CONFIG,SCHEMA

STRICT=('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','comfort')
METRICS=STRICT+('ego_progress','score','driving_direction_compliance')


def digest(x):
 def native(v):
  if isinstance(v,np.ndarray):return torch.from_numpy(v)
  if isinstance(v,np.generic):return v.item()
  if isinstance(v,dict):return {k:native(a)for k,a in v.items()}
  if isinstance(v,(list,tuple)):return [native(a)for a in v]
  return v
 return tensor_digest(native(x))


def sealed(value,key):
 result=copy.deepcopy(value);result[key]=digest(value);return result


def verify_seal(value,key):
 if value.get(key)!=digest({k:v for k,v in value.items()if k!=key}):raise ValueError('Retention material digest changed: '+key)


class FrozenPairFeatures:
 def __init__(self,base):self.base=base;self.cache=OrderedDict()
 def __getattr__(self,name):return getattr(self.base,name)
 def materialize(self,descriptor,device):
  token=descriptor['token']
  if token in self.cache:self.base.token=token;self.base.cached=checkpoint.device_tree(self.cache.pop(token),device)
  value=self.base.materialize(descriptor,device);self.cache[token]=checkpoint.cpu_tree(self.base.cached)
  while len(self.cache)>2:self.cache.popitem(last=False)
  return value


def b1_condition(actor,descriptor):
 """Real batch1: only frozen carriers are sliced; current Memory rebuilds atB1."""
 x=actor.materialize(descriptor)
 return dict(x,context=x['base_memory'],valid=x['base_valid'],ego=x['base_ego'])


def nominal_means(actor,descriptor,chain,timesteps,deltas):
 """Original _sde_mean_std on detached ODE states, with no probability call."""
 if chain.shape!=(1,11,8,3):raise ValueError('Historical teacher is exactly one ODE chain')
 raw=b1_condition(actor,descriptor);means=[]
 with actor._context_call(raw)as cond:
  for k in (7,8,9):
   x=chain[:,k].detach();t=timesteps[k].detach().reshape(1).to(x)
   v=actor.action_velocity(x,t,cond,use_ref=False)
   mean,_=actor._sde_mean_std(x,v,float(timesteps[k])/actor.infer_action_scheduler.num_train_timesteps,float(deltas[k]))
   means.append(mean)
 return torch.stack(means,1)


@torch.no_grad()
def deployment_chain(actor,raw,root):
 """Exact deployed CleanXFlowPolicy update arithmetic, with all11 states saved."""
 schedule=shift_sigma(torch.linspace(1,0,11,dtype=torch.float32),5.).tolist()
 state=root.float();states=[state.clone()]
 with actor._context_call(raw)as cond:
  prepared=actor._prepare(cond,False)
  with torch.autocast(actor.device.type,dtype=torch.bfloat16):
   for current,nxt in zip(schedule[:-1],schedule[1:]):
    sigma=torch.full((1,),current,device=actor.device,dtype=torch.float32)
    prediction=actor.action_expert(state,sigma,prepared)
    state=prediction.float()if nxt==0 else clean_x_step(state,prediction,sigma,nxt,validate=False)
    states.append(state.clone())
 # Source W3 nominal replay retains its own original scheduler arithmetic.
 times,deltas=actor.infer_action_scheduler.build_inference_schedule(10,actor.device,torch.float32,shift_override=actor.grpo_infer_shift)
 return torch.stack(states,1),times,deltas


def validate_feedback(record,token,log,requested,protocol):
 if record.get('status')!='OK'or record.get('token')!=token or record.get('log')!=log or record.get('protocol_sha256')!=protocol:raise ValueError('B1 official feedback scene/protocol mismatch')
 m=record['metrics']
 if any(not math.isfinite(float(m[k]))or not 0<=float(m[k])<=1 for k in METRICS):raise ValueError('Invalid official B1 metrics')
 score=m[STRICT[0]]*m[STRICT[1]]*(5*m['ego_progress']+5*m[STRICT[2]]+2*m['comfort'])/12
 if abs(score-m['score'])>5e-15 or abs(record['readouts']['canonical_PDMS']-m['score'])>5e-15:raise ValueError('B1 canonical aggregation/readout mismatch')
 req=torch.as_tensor(record['requested_trajectory']).double().cpu()
 if req.shape!=(8,3)or not torch.equal(req,requested.detach().double().cpu()):raise ValueError('B1 feedback must match exactly the FP32-scored trajectory')
 if not isinstance(record.get('cache'),dict)or not record['cache'].get('sha256'):raise ValueError('Verified B1 requires real canonical cache provenance')
 return record


def good(record):return all(record['metrics'][k]==1 for k in STRICT)and record['metrics']['score']>=CONFIG['min_score']

def regression(teacher,current):
 return any(current['metrics'][k]!=1 for k in STRICT)or current['metrics']['score']<teacher['metrics']['score']-CONFIG['regression_margin']


@torch.no_grad()
def probe(actor,conditions,token,log,service,version):
 """One legal root, one genuine B1 forward chain, one canonical score."""
 started=time.monotonic()
 descriptor=conditions.get(token,actor.device);raw=b1_condition(actor,descriptor)
 root=torch.randn((1,8,3),generator=torch.Generator(device='cpu').manual_seed(seed_for('native_qwen_sourceflow_ODE',CONFIG['ODE_seed'],token)),dtype=torch.float32).to(actor.device)
 chain,times,deltas=deployment_chain(actor,raw,root)
 if chain.shape!=(1,11,8,3)or not torch.isfinite(chain).all():raise ValueError('True B1 cleanxODE10shift5 required')
 requested=ActionCodec.denorm_odo(chain[:,-1]).float()
 requested_cpu=requested.cpu();forward=time.monotonic()-started;started=time.monotonic()
 records=service.score((token,log,requested_cpu.numpy()));scoring=time.monotonic()-started
 if len(records)!=1:raise ValueError('B1 verification must score exactly one original trajectory')
 feedback=validate_feedback(records[0],token,log,requested[0],service.protocol['sha256'])
 started=time.monotonic();means=nominal_means(actor,descriptor,chain,times,deltas).detach().cpu()if good(feedback)else None
 forward+=time.monotonic()-started
 value=dict(schema=SCHEMA,token=token,log=log,condition=descriptor,source_version=copy.deepcopy(version),source_rank=dist.get_rank()if dist.is_initialized()else 0,
  chain=chain.detach().cpu(),timesteps=times.detach().cpu(),deltas=deltas.detach().cpu(),
  nominal_mean=means,root_sha256=digest(root),seconds=dict(forward=forward,scoring=scoring),
  requested=requested[0].detach().cpu(),feedback=checkpoint.cpu_tree(feedback),metrics=copy.deepcopy(feedback['metrics']),
  protocol_sha256=service.protocol['sha256'],single_scored_candidate=True,PG_replay=False)
 return sealed(value,'witness_sha256')


def validate_witness(e,run,allowed,condition_sha,protocol,*,require_good=False,expected_rank=None):
 verify_seal(e,'witness_sha256')
 if (e.get('schema')!=SCHEMA or allowed.get(e.get('token'))!=e.get('log')or e['source_version']['run']!=run
     or type(e['source_version']['update'])is not int or e['source_version']['update']<CONFIG['admission_start_update']
     or e.get('protocol_sha256')!=protocol or e.get('single_scored_candidate')is not True or e.get('PG_replay')is not False
     or e['condition']!=dict(schema='world_postcot_trainable_memory_observation_v1',token=e['token'],identity=condition_sha,repeats=8)):
  raise ValueError('Retention witness ancestry/TRAIN scope/singleB1 contract changed')
 if expected_rank is not None and e.get('source_rank')!=expected_rank:raise ValueError('Foreign-rank retention witness')
 if any(not math.isfinite(e['seconds'][k])or e['seconds'][k]<0 for k in ('forward','scoring')):raise ValueError('Invalid B1 cost accounting')
 for k,shape,dtype in [('chain',(1,11,8,3),torch.float32),('timesteps',(10,),torch.float32),('deltas',(10,),torch.float32),('requested',(8,3),torch.float32)]:
  x=e[k]
  if x.shape!=shape or x.dtype!=dtype or x.requires_grad or not torch.isfinite(x).all():raise ValueError('Invalid detached B1 '+k)
 root=torch.randn((1,8,3),generator=torch.Generator().manual_seed(seed_for('native_qwen_sourceflow_ODE',CONFIG['ODE_seed'],e['token'])),dtype=torch.float32)
 if not torch.equal(e['chain'][:,0].cpu(),root)or e['root_sha256']!=digest(root):raise ValueError('Retention ODE root is not the exact evaluation-seed B1 root')
 if not torch.equal(ActionCodec.denorm_odo(e['chain'][:,-1]).cpu()[0],e['requested'].cpu()):raise ValueError('Stored ODE endpoint/FP32 action mismatch')
 validate_feedback(e['feedback'],e['token'],e['log'],e['requested'],protocol)
 if e['metrics']!=e['feedback']['metrics']:raise ValueError('Stored witness metrics changed')
 if require_good and not good(e['feedback']):raise ValueError('Non-high-quality trajectory cannot become a historical target')
 mean=e['nominal_mean']
 if good(e['feedback']):
  if mean is None or mean.shape!=(1,3,8,3)or mean.dtype!=torch.float32 or mean.requires_grad or not torch.isfinite(mean).all():raise ValueError('Verified RL teacher needs detached native W3 means')
 elif mean is not None:raise ValueError('Failed quality witness must not provide a teacher target')
 return e
