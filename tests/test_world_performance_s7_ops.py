"""Exact S7 control-plane contracts only; fixtures confer no GPU qualification."""
import copy,inspect,json,signal
from pathlib import Path
import pytest,yaml
from tools.qwen_world_performance import launch as L,evaluate as E,deadline_guard as G
from tools.qwen_world_safety_exploration import job_worker as J
from starVLA.rl.qwen_world_research_data.manifest import fingerprint


def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))


def test_S7_pure_registration_requires_own_proof_exact_family_and_parent_aware_export(tmp_path,monkeypatch):
 before=(L.ROOT/'RUNS.json').read_bytes();registry=copy.deepcopy(L.read(L.ROOT/'RUNS.json'));registry['runs']=[r for r in registry['runs']if r['id']!=L.STRUCTURED]
 base=next(r for r in registry['runs']if r['id']==L.QUALITY);cfg=yaml.safe_load(Path(base['config']).read_text())
 cfg['research'].update(arm=L.STRUCTURED,schema='qwen_world_structured_progress_rl_v1',family='structured_progress',structured_progress={'schema':'CPU-only contract fixture; not scientific qualification'})
 config=tmp_path/'S7.yaml';config.write_text(yaml.safe_dump(cfg));source=tmp_path/'frozen_fixture';source.mkdir();proof=tmp_path/'CPU_qualification_interface_fixture.json'
 protocol=fingerprint(cfg['research']['structured_progress']);parent=cfg['research']['performance']['parent_model_sha256']
 metadata=dict(schema='world_structured_progress_export_native_validation_v1',status='PASS',world=8,qualification=True,source=str(source),structured_progress_protocol_sha256=protocol,parent_model_sha256=parent)
 q=dict(schema='world_structured_progress_actual_world8_native_qualification_v1',status='PASS',world=8,B=64,G=16,M=4,full_IL70=True,trainable_numel=52_707_840,actual_initial_epoch=0.,qualification_only=True,native_bitexact_at_inner2_and4=True,structured_progress_protocol_sha256=protocol,actual_first_collection_teacher_classes={'.02':1,'.05':1,'structured':1},strict_reviews={k:metadata for k in('continuous2','continuous4','split2','split4')});dump(proof,q)
 actual=L.validate_run_recipe
 def validate(run):
  if run['id']!=L.STRUCTURED:return actual(run)
  if run['trainer_module']!=L.INDEPENDENT_FAMILIES[L.STRUCTURED]['trainer']:raise ValueError('Exact S7 trainer required')
  return cfg
 monkeypatch.setattr(L,'validate_run_recipe',validate);git=L.subprocess.check_output
 monkeypatch.setattr(L.subprocess,'check_output',lambda command,**kw:(''if 'status'in command else'CPU_FIXTURE_COMMIT\n')if command[:3]==['git','-C',str(source)]else git(command,**kw))
 result=L.structured_registration(registry,config,source,worker_source=tmp_path/'ops',qualification=proof)
 assert (L.ROOT/'RUNS.json').read_bytes()==before and len(result['runs'])==len(registry['runs'])+1
 run=next(r for r in result['runs']if r['id']==L.STRUCTURED)
 assert [r['id']for r in result['runs']][:3]==['S1_NOMINAL_RESCUE',L.STRUCTURED,'S5_PERFORMANCE_MAX']
 assert run['parent_checkpoint']==base['parent_checkpoint']and run['root']!=base['root']
 assert run['native_execution']==dict(schema='world_performance_continuous_to_registered_cap_v1',scope='future_attempts_only')
 assert run['gates'][:4]==[404,808,1616,3228]and run['target_updates']==38736
 assert not any(x['host']in L.PROTECTED|{'training-vlawm-tsinghua-zt5'}for x in run['deployment_candidates'])
 assert not any(k.startswith('execution_')and k!='execution_output_root'for k in run)
 bad=copy.deepcopy(run);bad['native_qualification']['file_sha256']='bad'
 with pytest.raises(ValueError,match='qualification evidence'):L.validate_structured_qualification(bad,cfg)
 q['status']='FAILED';dump(proof,q);run['native_qualification']['file_sha256']=L.sha(proof)
 with pytest.raises(ValueError,match='actual world8'):L.validate_structured_qualification(run,cfg)
 cp=dict(path=str(tmp_path/'sealedCP'),files_sha256='files',updates=404)
 receipt=dict(metadata,checkpoint=cp['path'],files_sha256='files',counts=dict(updates=404,next_inner=0),qualification=False)
 assert E.validate_native_export_receipt(run,cp,receipt)==receipt
 for change in(dict(structured_progress_protocol_sha256='wrong'),dict(qualification=True),dict(parent_model_sha256='wrong')):
  with pytest.raises(ValueError,match='S7 export'):E.validate_native_export_receipt(run,cp,dict(receipt,**change))
 cmd=E.native_adoption_command(run,'sameIL70',cp['path'],'CPU_PYTHON')
 assert cmd[-2:]==['sameIL70',cp['path']]and'load_validated(sys.argv[1],sys.argv[2])'in cmd[2]
 assert E.native_adoption_command(dict(run,id=L.QUALITY),'sameIL70',cp['path'],'CPU_PYTHON')[-1:]==[cp['path']]
 with pytest.raises(ValueError,match='Wrong exact'):actual(dict(run,trainer_module='unrelated.train'))


def test_guardian_and_worker_only_own_exact_S7_entry_never_D44_or_foreign(monkeypatch,tmp_path):
 entry='tools.qwen_world_structured_progress.train';output=str(L.ROOT/'runs'/L.STRUCTURED)
 assert entry in G.TRAIN_MODULES
 for module in('tools.qwen_world_performance.train','tools.qwen_world_multiscene.train','tools.qwen_world_verified_quality.train',entry):
  assert G.classify_direct(dict(args=['py','-m',module,'--output',output]),{L.TASK},'host')['role']=='native'
 assert G.classify_direct(dict(args=['py','-m',entry,'--output','/mnt/project/d44-native-rl-20261103/runs/foreign']),{L.TASK},'host')is None
 assert G.classify_direct(dict(args=['py','-m','tools.d44_native_rl.train','--output',output]),{L.TASK},'host')is None
 proc=tmp_path/'proc';proc.mkdir()
 def process(pid,parent,start,module,out):
  p=proc/str(pid);p.mkdir();v=['0']*22;v[0]=str(pid);v[1]='(py)';v[2]='S';v[3]=str(parent);v[21]=str(start);(p/'stat').write_text(' '.join(v));(p/'cmdline').write_bytes(('py\0-m\0'+module+'\0--output\0'+out+'\0').encode());(p/'status').write_text('SigCgt:\t'+format(1<<(signal.SIGUSR1-1),'x')+'\n')
 process(10,1,1000,'torch.distributed.run',output);process(11,10,1001,entry,output)
 process(12,10,1002,'tools.d44_native_rl.train',output);process(13,90,1003,entry,output);process(14,10,1004,entry,'/foreign/output')
 path=J.Path;monkeypatch.setattr(J,'Path',lambda p:proc if str(p)=='/proc'else path(p));monkeypatch.setattr(J.os,'getpgid',lambda pid:pid)
 rows=J.task_ranks(dict(command=['py','-m',entry,'--output',output],child_pid=10,child_start_ticks=1000))
 assert [r['pid']for r in rows]==[11]and rows[0]['entry']==entry
 assert 'd44_native_rl'not in inspect.getsource(J.task_ranks)
