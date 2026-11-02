"""True8scene/G16 native boundaries,1024credit and transactional history."""
import copy,math
import torch
import torch.distributed as dist
from . import recovery as R,integrated64_retention as history
from .credit import canonical_components,raw_advantage_parts,GroupCredit
from .retention_material import digest
from .integrated64_credit import protocol
from .integrated64_recipe import CONFIG,SCHEMA,CREDIT_SCHEMA,validate_integrated_recipe
from .integrated64_schedule import validate_scheduler_state,peaks


def validate_boundary(state,window,identity,optimizer=None,scheduler=None,*,stream=None,legacy=False):
 if legacy:raise ValueError('Integrated run needs its own complete native state')
 cfg=validate_integrated_recipe(identity['recipe']);u,c,i=(state[k]for k in ('updates','collections','next_inner'));pending=state.get('phase')in('ROLLOUT_COMMITTED','INNER_COMMITTED')
 if(identity['world']!=8 or type(u)is not int or not 0<=u<=12500 or not 0<=i<4 or i!=u%4 or c!=u//4+int(pending)
    or state.get('phase')not in('READY','ROLLOUT_COMMITTED','INNER_COMMITTED','COMPLETE')or(state['phase']=='COMPLETE'and u!=12500)
    or(state['phase']=='ROLLOUT_COMMITTED'and i!=0)or(state['phase']=='INNER_COMMITTED'and i==0)or pending!=(window is not None)
    or any(state[k]!=64*c for k in('scene_cursor','raw_scene_exposures'))or state['new_candidates']!=1024*c):raise ValueError('Integrated actual window/Adam/query boundary mismatch')
 if(optimizer is None)!=(scheduler is None):raise ValueError('Integrated Adam andscheduler validatedtogether')
 virtual=dict(state,scene_cursor=8*c,raw_scene_exposures=8*c,new_candidates=128*c)
 summary=dict(schema=SCHEMA,protocol=CONFIG,phase=state['phase'],updates=u,collections=c,next_inner=i,scene_cursor=state['scene_cursor'],run=identity['sha256'],pending=pending)
 if not pending:R.validate_structure(virtual,None,identity,optimizer,scheduler)
 if scheduler is not None:
  oo=optimizer.state_dict()if hasattr(optimizer,'state_dict')else optimizer;ss=scheduler.state_dict()if hasattr(scheduler,'state_dict')else scheduler
  expected=validate_scheduler_state(ss,peaks(cfg),u);groups=oo['param_groups']
  if(len(groups)!=2 or[g.get('name')for g in groups]!=['action_lora','scene_predictive_memory']or[g.get('initial_lr')for g in groups]!=peaks(cfg)
      or any(not math.isfinite(g['lr'])or abs(g['lr']-v)>1e-15 for g,v in zip(groups,expected))):raise ValueError('Integrated actual Adam/WKC LR mismatch')
  summary['scheduler']=copy.deepcopy(ss)
 bank=state['integrated_retention'];rank=dist.get_rank()if dist.is_initialized()else bank['rank'];summary['rank']=rank
 if stream is None:raise ValueError('Integrated recovery requires registeredTRAIN stream')
 allowed={t:stream.logs[t]for t in stream.tokens};scoring=identity['score_protocol']['sha256']
 history.validate_memory(bank,identity['sha256'],rank,c,allowed,identity['projected_condition_identity'],scoring,u-i)
 summary['integrated_retention']=dict(schema=history.BANK_SCHEMA,windows=bank['windows'],scenes=bank['scenes'],bank_sha256=digest(bank),bank_size=len(bank['entries']))
 if not pending:return summary
 if(window.get('schema')!=SCHEMA or window.get('protocol')!=CONFIG or window.get('group_size')!=16 or len(window.get('samples',[]))!=8
    or window.get('window_start_cursor')!=state['scene_cursor']-64 or window.get('rank')!=rank or len(set(window.get('tokens',[])))!=8):raise ValueError('Integrated native8scene identity changed')
 samples=window['samples'];components=[]
 if(window.get('cond_bg')!={'schema':'world_integrated64_conditions_v1','observations':[s['cond_bg']for s in samples]}
    or window.get('tokens')!=[s['token']for s in samples]or window.get('logs')!=[s['log']for s in samples]
    or window.get('token')!=samples[0]['token']or window.get('log')!=samples[0]['log']
    or len(window['records'])!=128 or digest(window['records'])!=digest([r for s in samples for r in s['records']])):raise ValueError('Integrated actualobservations/records detachedfromwindow')
 for b,s in enumerate(samples):
  R.validate_structure(virtual,s,identity,optimizer if b==0 else None,scheduler if b==0 else None)
  token,pos=stream.at(window['window_start_cursor']+8*b,rank)
  if(s['behavior_version']!=window['behavior_version']or s.get('protocol_sha256')!=scoring or s['token']!=token or s['log']!=stream.logs[token]
      or s.get('stream_position')!=pos or window['window_positions'][b]!=pos):raise ValueError('Integrated behavior/scoring/scene slot changed')
  if any(r.get('token')!=token or r.get('log')!=s['log']or r.get('protocol_sha256')!=scoring for r in s['records']):raise ValueError('Fresh canonicalrecords havewrong scene/protocol')
  mat=canonical_components(s['records'],16);want=dict(components=mat['components'],**raw_advantage_parts(mat['components']));saved=s['structured_advantage']
  if saved.get('protocol')!=protocol()or set(saved)!=set(want)|{'protocol'}:raise ValueError('Integrated Safety math protocol changed')
  for k,v in want.items():
   actual=saved[k]
   if(actual.requires_grad or actual.dtype!=v.dtype or actual.shape!=v.shape or not torch.allclose(actual.cpu(),v,atol=1e-14,rtol=0)
      or not torch.equal(window['structured_advantage'][k][b:b+1].cpu(),actual.cpu())):raise ValueError('Integrated originalSafety component changed '+k)
  if not torch.equal(s['rewards'].cpu(),mat['canonical_rewards'])or not torch.equal(s['canonical_rewards'].cpu(),mat['canonical_rewards']):raise ValueError('HistoricalB1 feedback entered canonicalPG rewards')
  raw=saved['A_raw'].cpu();scale=s['normalization_std'].cpu();adv=(raw/(scale+1e-8)).reshape(-1).float();mask=raw.ne(0).any(1,keepdim=True).expand_as(raw).reshape(-1).float()
  if(not torch.equal(s['adv_mask'].cpu(),mask)or not torch.allclose(s['advantages'].cpu(),adv,atol=1e-7,rtol=1e-6)
      or not torch.equal(window['advantages'][b*16:(b+1)*16].cpu(),s['advantages'].cpu())or not torch.equal(window['adv_mask'][b*16:(b+1)*16].cpu(),s['adv_mask'].cpu())):raise ValueError('Integrated finalizedA/mask changed')
  for k in('global_std','normalization_std'):
   if not torch.equal(window[k].cpu(),s[k].cpu()):raise ValueError('Different scale onmicrobatch')
  std=(mat['canonical_rewards'].double()-mat['canonical_rewards'].double()[:,:1]).std(1,unbiased=False,keepdim=True)
  if not torch.allclose(s['raw_group_std'].cpu(),std,atol=1e-12,rtol=0)or s['advantage_statistics']!=window['advantage_statistics']:raise ValueError('Integrated groupstd/populationmetadata changed')
  components.append(mat['components'])
 for k in('chain','old_logp'):
  if not torch.equal(window[k].cpu(),torch.stack([s[k].cpu()for s in samples])):raise ValueError('Window/leaf '+k+' mismatch')
 for k in('rewards','canonical_rewards','raw_group_std'):
  if not torch.equal(window[k].cpu(),torch.cat([s[k].cpu()for s in samples])):raise ValueError('Window/leaf '+k+' mismatch')
 meta=window['advantage_statistics'];scale=window['normalization_std']
 if(meta.get('structured_advantage_schema')!=CREDIT_SCHEMA or meta.get('normalization_population_candidates')!=1024 or meta.get('normalization_population_groups')!=64
    or scale.shape!=()or scale.dtype!=torch.float64 or scale.requires_grad or not torch.isfinite(scale)or scale<0 or window['structured_advantage']['protocol']!=protocol()or'reward_transform'in window):raise ValueError('Integrated actual1024 normalization identity changed')
 states=window['generator_states']
 if len(states)!=9:raise ValueError('Nine actual main samplingRNG boundariesrequired')
 for b,s in enumerate(samples):
  gen=torch.Generator();gen.set_state(states[b].cpu());root=torch.randn((16,8,3),generator=gen,dtype=torch.float32)
  if not torch.equal(root,s['chain'][:,0].cpu()):raise ValueError('ActualG16 firstlatent differsfromsaved RNG')
  for _ in range(3):torch.randn((16,8,3),generator=gen,dtype=torch.float32)
  if not torch.equal(gen.get_state(),states[b+1].cpu()):raise ValueError('Integrated mainSDE RNG transition changed')
 if window.get('stream_position')!=window['window_positions'][0]:raise ValueError('Integrated first streampositionmissing')
 material=window['integrated_retention'];M=history.M;M.verify_seal(material,'material_sha256')
 if material['source_version']!=window['behavior_version']or material['rank']!=rank or material['window_sequence']!=c:raise ValueError('Integrated history window/version changed')
 history.validate_memory(material['memory_before'],identity['sha256'],rank,c-1,allowed,identity['projected_condition_identity'],scoring,u-i)
 for b,event in enumerate(material['events']):
  if digest(event)!=digest(samples[b]['verified_retention'])or event['fresh_token']!=samples[b]['token']:raise ValueError('Microbatch historical target not its frozen scene transaction')
  if(event['fresh_witness']is not None)!=(event['source_version']['update']>=history.CONFIG['admission_start_update']):raise ValueError('Integrated admissionclock changed')
  for name in('fresh_witness','history_witness','teacher'):
   witness=event[name]
   if witness is not None:
    history.validate_witness(witness,identity['sha256'],allowed,identity['projected_condition_identity'],scoring,rank,require_good=name=='teacher')
    if name=='teacher':
     if witness['source_version']['update']>=u-i:raise ValueError('Samewindow entry masquerades as historicalRL')
    elif witness['source_version']!=window['behavior_version']:raise ValueError('CurrentB1 score comesfromanother theta')
  if event['fresh_witness']is not None and event['fresh_witness']['token']!=samples[b]['token']:raise ValueError('Wrongfresh B1 scene')
  if event['history_witness']is not None and event['history_witness']['token']!=event['teacher']['token']:raise ValueError('Wronghistory B1 scene')
 if digest(history.replay(material))!=digest(bank):raise ValueError('Integrated shadowbank not committed once')
 summary.update(tokens=window['tokens'],behavior=window['behavior_version'],rewards=window['rewards'].cpu().double().tolist(),components=torch.cat(components).tolist(),advantages=window['advantages'].cpu().reshape(8,16).tolist(),
  mask=window['adv_mask'].cpu().reshape(8,16).tolist(),normalization_std=float(scale),global_std=float(window['global_std']),original_raw_A=window['structured_advantage']['A_raw'].cpu().tolist(),digests=digest(window))
 return summary


def validate_ranks(summaries):
 if len(summaries)!=8 or[s['rank']for s in summaries]!=list(range(8)):raise ValueError('Integratedexact8ranks required')
 common=('schema','protocol','phase','updates','collections','next_inner','scene_cursor','run','pending','scheduler')
 for s in summaries:
  if {k:s.get(k)for k in common}!={k:summaries[0].get(k)for k in common}:raise ValueError('Integratedrank boundary/protocol disagree')
 if not summaries[0]['pending']:return summaries
 if len({t for s in summaries for t in s['tokens']})!=64:raise ValueError('Integrated window needs64different TRAIN scenes')
 if any(s['behavior']!=summaries[0]['behavior']for s in summaries):raise ValueError('Integrateddifferent collection theta acrossranks')
 rewards=torch.tensor([s['rewards']for s in summaries],dtype=torch.float64).reshape(64,16);components=torch.tensor([s['components']for s in summaries],dtype=torch.float64).reshape(64,16,4)
 expected=GroupCredit(16).advantages(rewards,components,all_material=torch.cat((rewards.unsqueeze(-1),components),-1))
 for rank,s in enumerate(summaries):
  raw=torch.tensor(s['original_raw_A'],dtype=torch.float64);adv=torch.tensor(s['advantages'],dtype=torch.float32).reshape(-1);mask=torch.tensor(s['mask'],dtype=torch.float32).reshape(-1)
  wantmask=raw.ne(0).any(1,keepdim=True).expand_as(raw).reshape(-1).float();want=(raw/(s['normalization_std']+1e-8)).reshape(-1).float()
  if(abs(s['normalization_std']-float(expected['normalization_std']))>1e-12 or abs(s['global_std']-float(expected['global_std']))>1e-12
     or not torch.equal(mask,wantmask)or not torch.allclose(adv,want,atol=1e-7,rtol=1e-6)
     or not torch.allclose(adv,expected['advantages'][rank*128:(rank+1)*128],atol=1e-7,rtol=1e-6)):raise ValueError('Integrated normalization not actualfresh1024')
 return summaries
