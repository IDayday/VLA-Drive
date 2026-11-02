import copy,hashlib,json,threading
from pathlib import Path
import pytest,yaml
from tools.qwen_world_performance import continuous as C


def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,sort_keys=True))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def native(tmp):
 root=tmp/'study';rr=root/'runs/S1';cp=rr/'checkpoints/update000008_inner0';cp.mkdir(parents=True)
 cfg={'adaptation':'ATTN_LORA_MEMORY','science':'unchanged'};config=root/'config.yaml';config.write_text(yaml.safe_dump(cfg))
 body=dict(recipe=cfg,purpose='formal',world=8);identity=dict(body,sha256=C._fingerprint(body));dump(rr/'identity.json',identity);dump(cp/'identity.json',identity)
 rows=[dict(rank=r,run=identity['sha256'],updates=8,next_inner=0,phase='READY',valid_scene_exposures=512,collections=2)for r in range(8)]
 dump(cp/'transaction.json',dict(schema='simwam_checkpoint_transaction_v2',ranks=rows))
 for n in ['optimizer.pt']+[f'rank{r}.pt'for r in range(8)]:(cp/n).write_bytes(b'opaque native payload; never read by discovery')
 files={f.name:sha(f)for f in cp.iterdir()};dump(cp/'files.json',files);dump(cp/'COMPLETE',dict(schema=2,phase='READY',world=8,updates=8,next_inner=0,files_sha256=sha(cp/'files.json')))
 state=root/'owner/STATE.json';dump(state,{'runs':{'S1':{'status':'TRAINING','updates':12,'stages':{}}}})
 run=dict(id='S1',root=str(rr),config=str(config),config_sha256=sha(config),owner_state=str(state),native_gates=[8,16],gates=[8,16],target_updates=16,native_execution=C.PROTOCOL,exposure_mapping={'quarter':dict(updates=8,valid_scene_exposures=512,collections=2)})
 return root,run,cp,rows


def test_published_running_checkpoint_matches_legacy_SPEC_and_rejects_false_exposure(tmp_path,monkeypatch):
 from tools.qwen_world_safety_exploration import eval_owner as E
 _,run,cp,rows=native(tmp_path)
 original=Path.read_bytes
 monkeypatch.setattr(Path,'read_bytes',lambda p:(_ for _ in()).throw(AssertionError('discovery must not read tensors'))if p.suffix=='.pt'else original(p))
 expected,native_state=C.published_checkpoint_at_gate(run,8)
 assert native_state['status']=='TRAINING'and expected['identity_sha256']==sha(cp/'identity.json')
 assert expected['identity_sha256']!=json.loads((cp/'identity.json').read_text())['sha256']
 assert E.checkpoint_at_gate(run,8)[0]is None
 state=json.loads(Path(run['owner_state']).read_text());state['runs']['S1']['stages']['8']=dict(complete=True,checkpoint=str(cp));dump(Path(run['owner_state']),state)
 assert E.checkpoint_at_gate(run,8)[0]==expected  # Existing SPEC bytes identical.
 assert C.published_checkpoint_at_gate(run,16)[0]is None
 with pytest.raises(ValueError):C.published_checkpoint_at_gate(run,12)
 rows[4]['valid_scene_exposures']=511
 dump(cp/'transaction.json',dict(schema='simwam_checkpoint_transaction_v2',ranks=rows));files=json.loads((cp/'files.json').read_text());files['transaction.json']=sha(cp/'transaction.json');dump(cp/'files.json',files);seal=json.loads((cp/'COMPLETE').read_text());seal['files_sha256']=sha(cp/'files.json');dump(cp/'COMPLETE',seal)
 with pytest.raises(ValueError,match='exposure'):C.published_checkpoint_at_gate(run,8)


def test_future_targets_adopt_exact_legacy_job_and_never_swallow_science_failure(tmp_path,monkeypatch):
 run=dict(id='S1',native_gates=[8,16],target_updates=16,native_execution=C.PROTOCOL)
 old=dict(job=str(tmp_path/'jobs/S1/old'),command=['train','--stop-after-update','8'])
 state=dict(stages={'8':old});saved=copy.deepcopy(state)
 assert C.attempt_targets(run,state,tmp_path)==[8,16]and state==saved
 assert C.attempt_targets(run,{},tmp_path)==[16]
 assert C.attempt_targets(dict(run,native_execution=None),{},tmp_path)==[8,16]
 with pytest.raises(ValueError):C.attempt_targets(run,dict(stages={'8':old,'16':dict(job=str(tmp_path/'jobs/S1/new'))}),tmp_path)
 from tools.qwen_world_performance import launch as L
 from starVLA.rl.qwen_world_safety_exploration import contracts as CT
 monkeypatch.setattr(CT,'ROOT',L.ROOT)
 Owner=L.owner_class();monkeypatch.setattr(L,'ROOT',tmp_path);o=Owner.__new__(Owner);o.state={'runs':{'S1':state}};o.current=lambda arm:run;o.directory=tmp_path;o.handoff=False;calls=[];o.update=lambda *a,**kw:calls.append(kw)
 def failing(run,target,resume):calls.append(target);raise RuntimeError('real child failure retained')
 o.run_stage=failing;o.run_arm(run)
 assert 8 in calls and 16 not in calls and calls[-1]['status']=='FAILED_REVIEW_REQUIRED'


def test_prechild75_and_pending_resource_return_never_redispatch_short_attempt(tmp_path,monkeypatch):
 from tools.qwen_world_performance import launch as L
 from tools.qwen_world_safety_exploration import dispatch as D
 from starVLA.rl.qwen_world_safety_exploration import contracts as CT
 monkeypatch.setattr(CT,'authorization',lambda:({},None));monkeypatch.setattr(CT,'allowed_hosts',lambda **kw:[]);monkeypatch.setattr(CT,'ROOT',L.ROOT)
 Owner=L.owner_class();monkeypatch.setattr(L,'ROOT',tmp_path);monkeypatch.setattr(L.time,'sleep',lambda s:(_ for _ in()).throw(AssertionError('confirmed legacy return must handoff directly, not loop or sleep')))
 o=Owner.__new__(Owner);o.mutex=threading.RLock();o.directory=tmp_path/'owner';o.directory.mkdir();o.handoff=False
 run=dict(id='S1',native_execution=C.PROTOCOL,native_gates=[8,16],target_updates=16,root=str(tmp_path/'runs/S1'),source='fixed')
 command=['fixed-trainer','--stop-after-update','8'];node=dict(job=str(tmp_path/'jobs/S1/old'),host='h',command=list(command));o.stage=lambda *a:node;o.current=lambda a:run;o.update=lambda *a,**kw:None
 o.candidate=lambda r:(_ for _ in()).throw(AssertionError('no short redispatch'))
 monkeypatch.setattr(D,'job_state',lambda p:dict(returncode=75,child_not_started=True))
 assert o.run_stage(run,8,None)is None
 assert not node.get('complete')and node['future_attempt_target_updates']==16 and len(node['resource_refusals'])==1
 assert node['command']==command and C.attempt_targets(run,dict(stages={'8':node}),tmp_path)==[16]
 # A legitimate source-boundary EXIT0 with inner3 similarly carries its
 # exact pending checkpoint onward; no new rollout/Adam step occurs here.
 node.clear();node.update(job=str(tmp_path/'jobs/S1/old2'),host='h',command=list(command))
 dump(Path(run['root'])/'latest.json',dict(updates=7,checkpoint='own-sealed-inner3'))
 monkeypatch.setattr(D,'job_state',lambda p:dict(returncode=0))
 monkeypatch.setattr(L,'inspect_native',lambda r,p:dict(updates=7,next_inner=3,path=p))
 monkeypatch.setattr(L,'resource_return',lambda *a,**kw:dict(reason='EXACT_NATIVE_RESOURCE_BOUNDARY'))
 assert o.run_stage(run,8,None)=='own-sealed-inner3'
 assert not node.get('complete')and len(node['resource_returns'])==1
 assert node['command']==command and node['future_attempt_target_updates']==16 and C.attempt_targets(run,dict(stages={'8':node}),tmp_path)==[16]
 # A true child failure is retained, never converted to cap continuation.
 node.clear();node.update(job=str(tmp_path/'jobs/S1/failed'),host='h');monkeypatch.setattr(D,'job_state',lambda p:dict(returncode=1))
 with pytest.raises(RuntimeError,match='Real child failure'):o.run_stage(run,8,None)
 assert node['job'].endswith('/failed')

 # An unproven75 with child_started/unknown is a failure, not a capacity handoff.
 monkeypatch.setattr(D,'job_state',lambda p:dict(returncode=75,child_not_started=False))
 with pytest.raises(RuntimeError,match='Real child failure'):o.run_stage(run,8,None)
 assert node['job'].endswith('/failed')
