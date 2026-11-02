"""Focused LR/Adam tampering rejection and pre-plan stage/resume transition."""
import copy,io,json
from pathlib import Path
from types import SimpleNamespace
import pytest,torch,yaml
from starVLA.rl.qwen_world_performance.batch_runtime import PerformanceRuntime
from starVLA.rl.qwen_world_performance.sampling_priority import PerformanceSceneStream
from starVLA.rl.qwen_world_performance import nominal_improvement as N,persistent_history as H
from starVLA.rl.qwen_world_performance.recovery import validate_boundary
from starVLA.rl.qwen_world_performance.feedback import _memory

ROOT=Path('/mnt/project/qwen-world-performance-rl-20261103')

def fixture():
 c=yaml.safe_load((ROOT/'configs/QUAL_S5_PERFORMANCE_MAX.yaml').read_text());p=c['research']['performance'];actor=torch.nn.Linear(1,1);actor._loss=SimpleNamespace()
 opt=torch.optim.AdamW([dict(params=[actor.weight],lr=c['learning_rate'],name='action_lora'),dict(params=[actor.bias],lr=c['memory_learning_rate'],name='scene_predictive_memory')])
 rt=PerformanceRuntime();rt.spec=p;rt.cfg=c;rt.actor=actor;rt.group_size=16;rt.parent=dict(model_sha256=p['parent_model_sha256']);sched=rt.make_scheduler(opt,c)
 stream=PerformanceSceneStream(ROOT/'data/NAVTRAIN_SCENES.json',8,batch_size=p['stages'][0]['B'],run_identity='CPU_review')
 st=dict(updates=0,collections=0,next_inner=0,phase='READY',collection_ledger=[],scene_cursor=0,raw_scene_exposures=0,new_candidates=0,valid_scene_exposures=0,padded_scene_exposures=0,
  current_stage=p['stages'][0]['name'],performance_rank=0,extra_B1_queries=0,adaptive_G_queries=0,history_B1_queries=0,
  seconds_auxiliary=0.,evaluation_milestones=[],stage_transitions=[],
  performance_auxiliary=dict(nominal_improvement=N.new_calibration_state(),history=H.new_history_state('CPU_review',0)))
 stream.bind(st);ident=dict(recipe=c,sha256='CPU_review',world=8,data_stream_identity=stream.identity,performance_rank=0)
 return c,p,actor,opt,sched,rt,stream,st,ident


def test_ready_rejects_stage_peak_optimizer_LR_and_nonempty_Adam0():
 torch.set_num_threads(1);c,p,a,o,s,rt,stream,st,ident=fixture();validate_boundary(st,None,ident,o,s)
 assert s.stage==st['current_stage']==p['stages'][0]['name']
 oo=o.state_dict();ss=s.state_dict()
 bad=copy.deepcopy(oo);bad['param_groups'][0]['lr']=1.
 with pytest.raises(ValueError,match='learning rates'):validate_boundary(st,None,ident,bad,ss)
 bad=copy.deepcopy(ss);bad['stage']='wrongstage'
 with pytest.raises(ValueError,match='stage'):validate_boundary(st,None,ident,oo,bad)
 bad=copy.deepcopy(ss);bad['base_lrs']=[.001,.002];bad['_last_lr']=[x*.001 for x in bad['base_lrs']]
 matching=copy.deepcopy(oo)
 for g,lr in zip(matching['param_groups'],bad['_last_lr']):g['lr']=lr
 with pytest.raises(ValueError,match='peak learning'):validate_boundary(st,None,ident,matching,bad)
 bad=copy.deepcopy(ss);bad['population']+=8
 with pytest.raises(ValueError,match='stage'):validate_boundary(st,None,ident,oo,bad)
 bad=copy.deepcopy(oo);bad['state'][0]=dict(step=torch.tensor(0.),exp_avg=torch.ones_like(a.weight),exp_avg_sq=torch.ones_like(a.weight))
 with pytest.raises(ValueError,match='empty Adam'):validate_boundary(st,None,ident,bad,ss)


def test_stage_B64_to128_before_at_and_exact_scheduler_native_resume(tmp_path,monkeypatch):
 monkeypatch.setattr('starVLA.rl.qwen_world_performance.batch_runtime.dist.get_rank',lambda:0)
 torch.set_num_threads(1);c,p,a,o,s,rt,stream,st,ident=fixture();ledger=[]
 #One real complete token-plan pass (no images/scoring/model updates). Adam
 #counters/moments below are explicitly synthetic native-boundary fixtures.
 while stream._state['epoch']==0:
  plan=stream._plan();b=stream.batch_size;n=sum(x['valid']for x in plan['items']);stream.commit_window(None,plan['start_cursor']+b)
  ledger.append(dict(B=b,G=16,M=4,valid=n,start_update=4*len(ledger),stage='coverage'))
 st.update(scene_cursor=stream._state['cursor'],raw_scene_exposures=stream._state['cursor'],valid_scene_exposures=stream._state['valid_scene_exposures'],padded_scene_exposures=stream._state['padding_slots'],
  updates=4*len(ledger),collections=len(ledger),collection_ledger=ledger,new_candidates=stream._state['cursor']*16)
 u=st['updates'];s.set_exposures(st['valid_scene_exposures']);s.last_epoch=u
 for group in o.param_groups:
  for param in group['params']:o.state[param]=dict(step=torch.tensor(float(u)),exp_avg=torch.ones_like(param)*.007,exp_avg_sq=torch.ones_like(param)*.013)
 _memory(st);validate_boundary(st,None,ident,o,s)
 rt.stream=stream;rt.scheduler=s;rt.args=SimpleNamespace(output=str(tmp_path),resume=None);rt.bind_training_state(st,None,ident)
 assert rt.stage['name']=='coverage'and stream._state['selection_plan']is None
 #The trainer invokes this hook before the first stream.at of the newepoch.
 monkeypatch.setattr('starVLA.rl.qwen_world_performance.batch_runtime.dist.get_rank',lambda:0)
 rt.before_collection(a,st)
 receipt=json.loads((tmp_path/'stage_transitions/absorb/stage_transition_receipt.json').read_text())
 assert receipt['optimizer_state_preserved']and not receipt['optimizer_reset']and receipt['pinned_parent_sha']==rt.parent['model_sha256']
 assert st['current_stage']==rt.stage['name']=='absorb'and stream.batch_size==128
 assert s.stage=='absorb'and s.base_lrs==[3e-5,3e-6]
 assert stream._state['selection_plan']is None
 validate_boundary(st,None,ident,o,s)
 data=io.BytesIO();torch.save((st,o.state_dict(),s.state_dict()),data);data.seek(0);saved,oo,ss=torch.load(data,weights_only=False)
 c2,p2,a2,o2,s2,rt2,stream2,_,_=fixture();o2.load_state_dict(oo);s2.load_state_dict(ss);rt2.stream=stream2;rt2.scheduler=s2;rt2.args=SimpleNamespace(output=str(tmp_path),resume='CPU_counter_fixture')
 rt2.bind_training_state(saved,None,ident)
 assert rt2.stage['name']=='absorb'and stream2.batch_size==128 and s2.state_dict()==ss
 rt2.before_collection(a2,saved);assert s2.state_dict()==ss #Never skip/reset/duplicate the saved stage transition.
 _,pos=stream2.at(saved['scene_cursor'],0);assert pos['valid']and len(stream2._state['selection_plan']['items'])==128
 assert all(float(value['step'])==u and torch.all(value['exp_avg']==.007)for value in o2.state.values())
