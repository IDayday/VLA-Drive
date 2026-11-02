import copy,json,os
from pathlib import Path
from types import SimpleNamespace
import pytest
from tools.qwen_world_safety_exploration import known_peers as K,job_worker as J,eval_owner as E
from tools.qwen_world_performance import continuation as C


def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))
def fixture(tmp,monkeypatch):
 root=tmp/'consumer';peer=tmp/'peer';source=peer/'runtime/science';ops=peer/'runtime/ops';job=peer/'jobs/D0/attempt8';out=peer/'runs/D0'
 for p,commit in [(source,'a'*40),(ops,'b'*40)]:dump(p/'.git/unused',{});(p/'.git/HEAD').write_text(commit)
 (source/'train.py').write_text('frozen science');(ops/'worker.py').write_text('frozen worker');config=peer/'D0.json';dump(config,{'same':True})
 registration=dict(arm='D0',source=str(source),source_sha='a'*40,config=str(config),config_sha256=K.file_sha(config),output=str(out))
 dispatch=dict(registration,job=str(job),worker_source=str(ops),worker_source_sha='b'*40)
 running=dict(status='RUNNING',worker_pid=100,worker_start_ticks=1000,child_pid=101,child_start_ticks=1001,gpu_uuids=[str(i)for i in range(8)],execution_worker=str(ops/'worker.py'),execution_worker_sha256=K.file_sha(ops/'worker.py'))
 dump(peer/'CAMPAIGN.json',{'runs':{'D0':dict(registration,jobs=[dispatch])}});dump(job/'LAUNCH.json',dispatch);dump(job/'RUNNING.json',running)
 identity=K.sealed(dict(source_files={'train.py':K.file_sha(source/'train.py')}));dump(out/'identity.json',identity)
 grant=dict(task=K.PEER_TASK,root=str(peer),temporary_hosts=['h'],permanent_hosts=[],shared_training_hosts=['h'],expires_unix=K.CUTOFF)
 record=peer/'grant.json';dump(record,grant);policy=tmp/'policy.json';dump(policy,{'task_authorizations':{K.PEER_TASK:dict(grant,record=str(record))}})
 processes={str(p):dict(pid=p,start_ticks=9000+p,ppid=101 if p>=102 else 0,state='R',cwd=str(source),command_sha256='args'+str(p))for p in range(100,110)}
 owners=[dict(pid=102+i,uuid=str(i),used_MiB=30000,visible=True,entry=K.ENTRY,output=str(out),start_ticks=9102+i)for i in range(8)]
 receipt=K.sealed(dict(schema=K.SCHEMA,consumer_root=str(root),peer_root=str(peer),peer_task=K.PEER_TASK,entry=K.ENTRY,arm='D0',host='h',expires_unix=K.CUTOFF,
 registration=registration,dispatch=dispatch,job=str(job),launch_file_sha256=K.file_sha(job/'LAUNCH.json'),running=running,gpu_uuids=running['gpu_uuids'],identity_file_sha256=K.file_sha(out/'identity.json'),identity_sha256=identity['sha256'],scientific_source_files=identity['source_files'],resource_record=str(record),resource_record_sha256=K.file_sha(record),grant_binding=grant,processes=list(processes.values()),rank_owners=[{'pid':o['pid'],'uuid':o['uuid']}for o in owners]))
 path=root/'resources/known_peers/peer.json';dump(path,receipt);catalog=root/'resources/KNOWN_PEER_RECEIPTS.json';dump(catalog,K.sealed(dict(schema=K.CATALOG,consumer_root=str(root),expires_unix=K.CUTOFF,receipts=[dict(path=str(path),file_sha256=K.file_sha(path))])))
 sample=dict(host='h',gpus=[dict(uuid=str(i),free_MiB=50000,total_MiB=81920,utilization=20)for i in range(8)],owners=owners,processes=processes)
 monkeypatch.setattr(K,'PEER_ROOT',peer);monkeypatch.setattr(K,'POLICY',policy);return root,path,catalog,sample,processes


def test_exact_peer_CPU_and_worker_material_reject_PID_args_UUID_expiry_and_count_one(tmp_path,monkeypatch):
 root,path,catalog,s,processes=fixture(tmp_path,monkeypatch)
 assert len(K.accepted(s,root)['owners'])==8
 for field,value in [('start_ticks',0),('command_sha256','changed')]:
  bad=copy.deepcopy(s);bad['processes']['102'][field]=value
  assert not K.accepted(bad,root)['owners']
 bad=copy.deepcopy(s);bad['owners'][0]['uuid']='wrong'
 assert not K.accepted(bad,root)['owners']
 monkeypatch.setattr(K,'process',lambda pid:processes[str(pid)])
 assert len(K.accepted(s,root,receipt_paths=[str(path)],local=True)['owners'])==8
 processes['102']['command_sha256']='new-local-args'
 assert not K.accepted(s,root,receipt_paths=[str(path)],local=True)['owners']
 processes['102']['command_sha256']='args102'
 monkeypatch.setattr(E,'ROOT',root);monkeypatch.setattr(E,'lease_free',lambda p:True)
 assert E.usable_gpu(s['gpus'][0],s,minimum=32768,reserved={})
 monkeypatch.setattr(C,'ROOT',root)
 run=dict(root=str(root/'runs/own'),resource_protocol=C.PROTOCOL_V2)
 assert C.shared_sample_ready(s,run)[0]
 two=copy.deepcopy(s);two['owners']+=[dict(pid=500,uuid='0',entry='tools.qwen_world_performance.train',visible=True,output=str(root/'runs/another'))]
 assert not C.shared_sample_ready(two,run)[0]  #D44 counts1; no third learner.
 catalog.unlink();assert not K.accepted(s,root)['owners'];assert not E.usable_gpu(s['gpus'][0],s,minimum=32768,reserved={})
 monkeypatch.setattr(K.time,'time',lambda:K.CUTOFF+1)
 assert not K.accepted(s,root,receipt_paths=[str(path)])['owners']


def test_worker_keeps_two_actual_locked_admission_checks_and_same_headroom(tmp_path,monkeypatch):
 roots=[tmp_path/x for x in('primary','shared','research','evaluation')]
 for attr,p in zip(('PRIMARY_LEASE_ROOT','SHARED_LEASE_ROOT','RESEARCH_LEASE_ROOT','EVALUATION_LEASE_ROOT'),roots):monkeypatch.setattr(J,attr,p)
 devices={i:dict(index=i,uuid=str(i),free_MiB=50000,total_MiB=81920,utilization=10)for i in range(8)}
 owners=[dict(pid=100+i,uuid=str(i),used_MiB=30000)for i in range(8)];calls=[]
 monkeypatch.setattr(J,'_admission_sample',lambda:(devices,owners));monkeypatch.setattr(J,'shared_training_host',lambda *a:True)
 monkeypatch.setattr(J,'_known_owners',lambda r,m,i,**kw:calls.append(kw))
 args=SimpleNamespace(share_min_free_mib=32768,minimum_free_mib=None,lease_slot='primary',known_peer_receipt=['exact.json'],exclusive_all_slots=False)
 locks=[];J._gpu_admission(args,'h',{},list(range(8)),list(map(str,range(8))),{},locks)
 assert len(calls)==2 and all(x['peer_receipts']==['exact.json']and x['all_uuids']==list(map(str,range(8)))for x in calls)
 for f in locks:f.close()


def test_worker_loads_exact_sibling_helper_and_default_unknown_remains_refused(tmp_path,monkeypatch):
 monkeypatch.setattr(J,'__file__',str(tmp_path/'job_worker.py'))
 (tmp_path/'known_peers.py').write_text('def accepted(sample,root,**kw):\n return {"owners":{(sample["owners"][0]["pid"],"u")}}\n')
 rows=[dict(pid=os.getpid(),uuid='u')]
 J._known_owners(rows,{0:'u'},[0],peer_receipts=['unit-only'],host='h',all_uuids=['u'])
 with pytest.raises(J.ResourceRefused):J._known_owners(rows,{0:'u'},[0])


def test_evaluator_opt_in_keeps_default_unknown_and_two_owner_boundary(monkeypatch):
 gpu=dict(uuid='u',free_MiB=65000,total_MiB=81920,utilization=20)
 owner=dict(pid=101,uuid='u',used_MiB=15000,visible=True,entry='tools.qwen_world_compare.evaluate',output=E.KNOWN_ROOTS[0]+'evaluations/model1')
 sample=dict(host='unit',gpus=[gpu],owners=[owner])
 monkeypatch.setattr(E,'lease_free',lambda p:True)
 assert not E.usable_gpu(gpu,sample,minimum=32768,reserved={})
 assert E.usable_gpu(gpu,sample,minimum=32768,reserved={},max_known_evaluations_per_gpu=2)
 sample['owners'].append(dict(owner,pid=102,output=E.KNOWN_ROOTS[0]+'evaluations/model2'))
 assert not E.usable_gpu(gpu,sample,minimum=32768,reserved={},max_known_evaluations_per_gpu=2)
 sample['owners']=[dict(owner,output='/unregistered/project/eval')]
 assert not E.usable_gpu(gpu,sample,minimum=32768,reserved={},max_known_evaluations_per_gpu=2)
 # Admission makes the same count using actual cmdline at each locked check.
 commands={101:b'python\0-m\0tools.qwen_world_compare.evaluate\0--output\0/mnt/project/qwen-world-four-frameworks-unit/eval\0',102:b'python\0-m\0tools.qwen_world_compare.evaluate\0--output\0/mnt/project/qwen-world-four-frameworks-unit/eval2\0'}
 original=J.Path
 # _known_owners constructs Path('/proc') / PID / 'cmdline'.
 class ProcRoot:
  def __init__(self,pid=None):self.pid=pid
  def __truediv__(self,key):return ProcRoot(int(key))if self.pid is None else self
  def read_bytes(self):return commands[self.pid]
 monkeypatch.setattr(J,'Path',lambda p:ProcRoot()if str(p)=='/proc'else original(p))
 rows=[dict(pid=101,uuid='u')]
 with pytest.raises(J.ResourceRefused):J._known_owners(rows,{0:'u'},[0],max_known_evaluations_per_gpu=1)
 J._known_owners(rows,{0:'u'},[0],max_known_evaluations_per_gpu=2)
 with pytest.raises(J.ResourceRefused):J._known_owners(rows+[dict(pid=102,uuid='u')],{0:'u'},[0],max_known_evaluations_per_gpu=2)
 commands[101]=b'python\0-m\0tools.foreign.evaluate\0--output\0/unregistered/project\0'
 with pytest.raises(J.ResourceRefused):J._known_owners(rows,{0:'u'},[0],max_known_evaluations_per_gpu=2)
