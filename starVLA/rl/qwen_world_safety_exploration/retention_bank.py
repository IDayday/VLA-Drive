"""Recoverable private-RNG reservoir and exact committed collection transition."""
import copy
import math
import torch
from starVLA.rl.simwam_recipe.contracts import seed_for
from starVLA.rl.simwam_recipe import checkpoint
from .retention_recipe import CONFIG,SCHEMA
from .retention_material import digest,good,regression,sealed,verify_seal,validate_witness


def new_bank(run,rank):
 rng=torch.Generator().manual_seed(seed_for('world_verified_RL_retention_bank',CONFIG['bank_seed'],rank))
 return dict(schema=SCHEMA,config=copy.deepcopy(CONFIG),run=run,rank=rank,collections=0,entries=[],rng_state=rng.get_state(),
  verified_admissions=0,reservoir_replacements=0,reservoir_not_retained=0,upgrades=0,history_selected=0,history_activated=0,
  fresh_ODE_queries_local=0,history_ODE_queries_local=0,extra_forward_seconds_local=0.,extra_scoring_seconds_local=0.)


def selection(memory,fresh_token):
 ids=[i for i,e in enumerate(memory['entries'])if e['token']!=fresh_token]
 rng=torch.Generator();rng.set_state(memory['rng_state'].cpu())
 index=ids[int(torch.randint(len(ids),(1,),generator=rng))]if ids else None
 return index,rng


def transition(memory,p):
 """Pure replayable update. Old bank is selected before any fresh admission."""
 verify_seal(p,'material_sha256')if 'material_sha256'in p else None
 if (p['collection']!=memory['collections']+1 or p['memory_before_sha256']!=digest(memory)
     or p['rank']!=memory['rank']or p['source_version']['run']!=memory['run']):raise ValueError('Retention collection transition identity changed')
 index,rng=selection(memory,p['fresh_token'])
 if index!=p['selection_index']:raise ValueError('Private history-selection RNG/index changed')
 old=None if index is None else memory['entries'][index]
 if (old is None)!=(p['teacher']is None)or(old is not None and digest(old)!=digest(p['teacher'])):raise ValueError('Selected historical target is not the old bank member')
 if (old is None)!=(p['history_witness']is None):raise ValueError('Every selected historical member requires a real current B1 score')
 expected_gate=old is not None and regression(old,p['history_witness']['feedback'])
 if p['enabled']!=expected_gate:raise ValueError('Historical regression gate changed')
 m=copy.deepcopy(memory);entries=m['entries'];current=p['history_witness']
 if old is not None:
  m['history_selected']+=1;m['history_activated']+=int(expected_gate)
  if good(current['feedback'])and current['metrics']['score']>old['metrics']['score']+CONFIG['upgrade_margin']:
   entries[index]=copy.deepcopy(current);m['upgrades']+=1
 fresh=p['fresh_witness']
 if fresh is not None and good(fresh['feedback']):
  existing=next((i for i,e in enumerate(entries)if e['token']==fresh['token']),None)
  if existing is not None:
   if fresh['metrics']['score']>entries[existing]['metrics']['score']+CONFIG['upgrade_margin']:
    entries[existing]=copy.deepcopy(fresh);m['upgrades']+=1
  else:
   m['verified_admissions']+=1
   if len(entries)<CONFIG['capacity']:entries.append(copy.deepcopy(fresh))
   else:
    j=int(torch.randint(m['verified_admissions'],(1,),generator=rng))
    if j<CONFIG['capacity']:entries[j]=copy.deepcopy(fresh);m['reservoir_replacements']+=1
    else:m['reservoir_not_retained']+=1
 m['rng_state']=rng.get_state();m['collections']+=1
 m['fresh_ODE_queries_local']+=int(fresh is not None);m['history_ODE_queries_local']+=int(current is not None)
 for w in (fresh,current):
  if w is not None:
   m['extra_forward_seconds_local']+=w['seconds']['forward'];m['extra_scoring_seconds_local']+=w['seconds']['scoring']
 return checkpoint.cpu_tree(m)


def make_payload(memory,rollout,fresh,history,index):
 teacher=None if index is None else copy.deepcopy(memory['entries'][index])
 p=dict(schema=SCHEMA,config=copy.deepcopy(CONFIG),collection=memory['collections']+1,rank=memory['rank'],
  fresh_token=rollout['token'],source_version=copy.deepcopy(rollout['behavior_version']),selection_index=index,
  memory_before=copy.deepcopy(memory),memory_before_sha256=digest(memory),teacher=teacher,fresh_witness=fresh,history_witness=history,
  enabled=teacher is not None and regression(teacher,history['feedback']))
 p['memory_after_sha256']=digest(transition(memory,p))
 return sealed(p,'material_sha256')


def validate_memory(memory,*,run,rank,collections,allowed,condition_sha,protocol,maximum_update):
 if (memory.get('schema')!=SCHEMA or memory.get('config')!=CONFIG or memory.get('run')!=run or memory.get('rank')!=rank
     or memory.get('collections')!=collections or not 0<=len(memory.get('entries',[]))<=CONFIG['capacity']
     or len({e['token']for e in memory['entries']})!=len(memory['entries'])or memory['verified_admissions']<len(memory['entries'])):
  raise ValueError('Retention reservoir state/header/capacity changed')
 g=torch.Generator();g.set_state(memory['rng_state'].cpu())
 for e in memory['entries']:
  validate_witness(e,run,allowed,condition_sha,protocol,require_good=True,expected_rank=rank)
  if e['source_version']['update']>maximum_update or e['source_version']['update']%4:raise ValueError('Future or non-collection teacher version')
 for k in ('verified_admissions','reservoir_replacements','reservoir_not_retained','upgrades','history_selected','history_activated','fresh_ODE_queries_local','history_ODE_queries_local'):
  if type(memory[k])is not int or not 0<=memory[k]:raise ValueError('Invalid retention counter '+k)
 if (memory['fresh_ODE_queries_local']>collections or memory['history_ODE_queries_local']!=memory['history_selected']
     or memory['history_activated']>memory['history_selected']or memory['history_selected']>collections):raise ValueError('Retention actual-scoring counters inconsistent')
 if any(not math.isfinite(memory[k])or memory[k]<0 for k in ('extra_forward_seconds_local','extra_scoring_seconds_local')):raise ValueError('Invalid actual extra-probe timing')
 return memory
