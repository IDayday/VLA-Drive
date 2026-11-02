"""Exact registered D44 cohosts. Pure stdlib, fail closed, no process mutation.

The worker loads this file beside its own frozen __file__, independently of the
older scientific PYTHONPATH. Receipts confer no host grant or eviction right.
"""
import hashlib,json,os,socket,time
from pathlib import Path
SCHEMA='world_exact_registered_D44_peer_v1'
CATALOG='world_exact_known_peer_catalog_v1'
PEER_ROOT=Path('/mnt/project/d44-native-rl-20261103')
PEER_TASK='d44_native_rl_20261103'
ENTRY='tools.d44_native_rl.train'
POLICY=Path('/mnt/project/server_dispatch_policy.json')
CUTOFF=1793664000.

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def sealed(value):return dict(value,sha256=digest(value))
def unseal(path):
 x=read(path);body=dict(x);saved=body.pop('sha256',None)
 if saved!=digest(body):raise ValueError('Changed exact peer seal')
 return x

def process(pid,proc_root='/proc'):
 p=Path(proc_root)/str(pid);s=(p/'stat').read_text();z=s[s.rfind(')')+2:].split()
 argv=[a for a in(p/'cmdline').read_bytes().decode().split('\0')if a]
 return dict(pid=int(pid),start_ticks=int(z[19]),ppid=int(z[1]),state=z[0],cwd=str((p/'cwd').resolve()),command_sha256=digest(argv))

def detached_commit(root):
 root=Path(root);git=root/'.git'
 if git.is_file():git=(root/git.read_text().split('gitdir:',1)[1].strip()).resolve()
 head=(git/'HEAD').read_text().strip()
 if len(head)!=40 or any(c not in'0123456789abcdef'for c in head):raise ValueError('Peer source must be an exact detached frozen commit')
 return head

def paths_for_root(root):
 root=Path(root).resolve();p=root/'resources/KNOWN_PEER_RECEIPTS.json'
 if not p.exists():return []
 try:
  x=unseal(p)
  if x['schema']!=CATALOG or Path(x['consumer_root']).resolve()!=root or not time.time()<x['expires_unix']<=CUTOFF:return []
  paths=[]
  for r in x['receipts']:
   f=Path(r['path']).resolve()
   if not f.is_relative_to(root/'resources/known_peers')or file_sha(f)!=r['file_sha256']:raise ValueError('Peer catalog path/payload changed')
   paths.append(str(f))
  return paths
 except (OSError,ValueError,KeyError,TypeError):return []

def checked_receipt(path,root,host,now):
 path=Path(path).resolve();root=Path(root).resolve()
 if not path.is_relative_to(root/'resources/known_peers'):raise ValueError('Foreign peer receipt path')
 r=unseal(path)
 if(r['schema']!=SCHEMA or Path(r['consumer_root']).resolve()!=root or r['peer_root']!=str(PEER_ROOT)
    or r['peer_task']!=PEER_TASK or r['entry']!=ENTRY or r['host']!=host or not now<r['expires_unix']<=CUTOFF):raise ValueError('Wrong exact peer scope/expiry')
 registry=read(PEER_ROOT/'CAMPAIGN.json');run=registry['runs'][r['arm']]
 for k,v in r['registration'].items():
  if run.get(k)!=v:raise ValueError('Peer scientific registration changed: '+k)
 latest=run['jobs'][-1]
 if latest!=r['dispatch']or latest['job']!=r['job']:raise ValueError('Peer attempt changed')
 job=Path(r['job']).resolve()
 if not job.is_relative_to(PEER_ROOT/'jobs'/r['arm'])or(job/'RESULT.json').exists():raise ValueError('Peer attempt ended/escaped')
 # D44's actual dispatch publication is named LAUNCH.json, not DISPATCH.json.
 if file_sha(job/'LAUNCH.json')!=r['launch_file_sha256']or read(job/'LAUNCH.json')!=r['dispatch']:raise ValueError('Peer LAUNCH changed')
 running=read(job/'RUNNING.json')
 for k,v in r['running'].items():
  if running.get(k)!=v:raise ValueError('Peer RUNNING binding changed: '+k)
 if running['status']!='RUNNING'or running['gpu_uuids']!=r['gpu_uuids']:raise ValueError('Peer not exact RUNNING8UUID')
 identity_path=Path(run['output'])/'identity.json';identity=unseal(identity_path)
 if file_sha(identity_path)!=r['identity_file_sha256']or identity['sha256']!=r['identity_sha256']:raise ValueError('Peer native identity changed')
 if identity['source_files']!=r['scientific_source_files']or file_sha(run['config'])!=run['config_sha256']:raise ValueError('Peer source/config changed')
 if detached_commit(run['source'])!=run['source_sha']or detached_commit(latest['worker_source'])!=latest['worker_source_sha']:raise ValueError('Peer source commit changed')
 for name,h in r['scientific_source_files'].items():
  f=(Path(run['source'])/name).resolve()
  if Path(name).is_absolute()or not f.is_relative_to(Path(run['source']))or file_sha(f)!=h:raise ValueError('Peer scientific source payload changed')
 if file_sha(running['execution_worker'])!=running['execution_worker_sha256']:raise ValueError('Peer worker payload changed')
 policy=read(POLICY);grant=policy['task_authorizations'][PEER_TASK]
 if grant.get('record')!=r['resource_record']or file_sha(grant['record'])!=r['resource_record_sha256']:raise ValueError('Peer task grant record changed')
 record=read(grant['record'])
 for k,v in r['grant_binding'].items():
  if grant.get(k)!=v or record.get(k)!=v:raise ValueError('Peer host authorization changed')
 allowed=set(grant.get('permanent_hosts',[]))|set(grant.get('temporary_hosts',[]))|set(grant.get('unbounded_temporary_hosts',[]))
 if host not in allowed or host not in grant.get('shared_training_hosts',[])or now>=grant['expires_unix']:raise ValueError('Peer no longer authorized on host')
 return r

def accepted(sample,root,*,receipt_paths=None,local=False):
 """Return exact accepted (NVMLPID,physicalUUID) pairs, never a root prefix."""
 now=time.time();root=Path(root).resolve();paths=paths_for_root(root)if receipt_paths is None else list(receipt_paths)
 empty=dict(owners=set(),paths=[],native_outputs=set(),errors=[])
 if not paths:return empty
 host=(sample.get('host')or socket.gethostname().removesuffix('-worker-0'))if local else sample.get('host')
 if not host:return empty
 signature=digest(dict(host=host,gpus=[g['uuid']for g in sample['gpus']],owners=sample['owners'],processes=sample.get('processes',{})))if not local else None
 if not local:
  cache=sample.get('_exact_peer_validation')
  if cache and cache.get('paths_key')==tuple(paths)and cache.get('signature')==signature and now<cache['expires_unix']:return cache['value']
 result=dict(owners=set(),paths=[],native_outputs=set(),errors=[]);expires=CUTOFF
 for path in paths:
  try:
   r=checked_receipt(path,root,host,now);rows=sample['owners'];procs=sample.get('processes',{})
   if r['gpu_uuids']!=[g['uuid']for g in sample['gpus']]or len(set(r['gpu_uuids']))!=8:raise ValueError('Host physical8UUID mapping changed')
   for saved in r['processes']:
    actual=process(saved['pid'])if local else procs.get(str(saved['pid']))
    if not actual or actual.get('state')in('Z','X')or any(actual.get(k)!=saved[k]for k in('pid','start_ticks','ppid','cwd','command_sha256')):raise ValueError('Peer actual PID/start/args/source ancestry changed')
   seen=set()
   for expected in r['rank_owners']:
    matching=[o for o in rows if o['pid']==expected['pid']and o['uuid']==expected['uuid']]
    if len(matching)!=1:raise ValueError('Peer actual NVMLPID/UUID changed')
    owner=matching[0]
    if not local and(owner.get('visible')is not True or owner.get('entry')!=ENTRY or Path(owner.get('output','')).resolve()!=Path(r['registration']['output']).resolve()):raise ValueError('Peer owner visibility/output/entry changed')
    seen.add((expected['pid'],expected['uuid']))
   if len(seen)!=8:raise ValueError('Not exact8rank peer')
   result['owners'].update(seen);result['paths'].append(str(Path(path).resolve()));result['native_outputs'].add(str(Path(r['registration']['output']).resolve()));expires=min(expires,r['expires_unix'])
  except (OSError,ValueError,KeyError,TypeError,IndexError)as e:result['errors'].append(str(e))
 if not local:sample['_exact_peer_validation']=dict(paths_key=tuple(paths),signature=signature,expires_unix=expires,value=result)
 return result
