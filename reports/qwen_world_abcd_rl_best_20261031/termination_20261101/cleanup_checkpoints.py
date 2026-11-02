import collections,fcntl,hashlib,json,os,shutil,stat,time
from pathlib import Path
N=Path('/mnt/project/qwen-world-abcd-rl-best-20261031');T=N/'termination_20261101'
def atomic(p,v):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2));os.replace(tmp,p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((N/'USER_TERMINATION.json').read_text())['status']=='TERMINATED_BY_USER'
assert not(T/'CHECKPOINT_CLEANUP_COMPLETE.json').exists(),'Already complete; no duplicate deletion'
registry=json.loads((N/'RUNS.json').read_text());expected={'ARCH_A_SAFETY_EP','ARCH_B_SAFETY_EP','ARCH_C_SAFETY_EP','ARCH_D_JAT_ACTION'}
assert {r['id']for r in registry['runs']}==expected
locks=[]
for p in [T/'cleanup.lock',N/'owners/ABCD_NATIVE/owner.lock',N/'owners/ABCD_EVALUATION/owner.lock']+[N/'runs'/a/f'rank{r}.lock'for a in sorted(expected)for r in range(8)]:
 f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
for family in ['ABCD_NATIVE','ABCD_EVALUATION']:
 l=json.loads((N/'owners'/family/'LAUNCH.json').read_text());p=Path('/proc')/str(l['pid'])
 assert not(p.exists()and(p/'stat').read_text().split()[2]not in ('Z','X')and int((p/'stat').read_text().split()[21])==l['start_ticks'])
retained=[]
for pattern in ['evaluations/*/*/SUMMARY.json','evaluations/*/*/identity.json','evaluations/*/*/layout.json','exports/*/export.json','exports/*/COMPLETE','configs/*.yaml','BEST_PARENT_SELECTION_LOCKED.json']:
 for p in N.glob(pattern):
  if p.is_file():retained.append(dict(path=str(p),sha256=sha(p)))
for a in sorted(expected):
 for p in (N/'runs'/a).glob('metrics_rank*.jsonl'):retained.append(dict(path=str(p),sha256=sha(p)))
paths=[];files=[];inode={};counts=collections.Counter();archived=[]
for a in sorted(expected):
 root=N/'runs'/a/'checkpoints';assert root.is_dir()and not root.is_symlink()and root.resolve()==root
 paths.append(str(root));counts['checkpoint_directories']+=len(list(root.iterdir()))
 for parent,dirs,names in os.walk(root,followlinks=False):
  for name in names:
   p=Path(parent)/name;s=p.lstat();key=(s.st_dev,s.st_ino)
   files.append(dict(path=str(p),bytes=s.st_size,blocks_bytes=s.st_blocks*512,device=s.st_dev,inode=s.st_ino,links=s.st_nlink))
   item=inode.setdefault(key,dict(links=s.st_nlink,deleted_links=0,blocks=s.st_blocks*512));item['deleted_links']+=1
   counts['files']+=1;counts['logical_bytes']+=s.st_size
 for cp in root.iterdir():
  if not cp.is_dir()or cp.is_symlink():continue
  for name in ['COMPLETE','files.json','identity.json','transaction.json']:
   p=cp/name
   if p.is_file():
    target=T/'deleted_native_metadata'/a/cp.name/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);archived.append(dict(original=str(p),archive=str(target),sha256=sha(target)))
counts['unique_inodes']=len(inode);counts['unique_allocated_bytes']=sum(x['blocks']for x in inode.values());counts['estimated_released_allocated_bytes']=sum(x['blocks']for x in inode.values()if x['links']==x['deleted_links']);counts['inodes_with_retained_external_links']=sum(x['links']>x['deleted_links']for x in inode.values())
plan=dict(schema='ABCD_RL_checkpoint_deletion_v1',status='AUTHORIZED_BY_USER',request='可以清除ABCD 四组RL训练的ckpts',paths=paths,counts=dict(counts),files=files,archived_native_metadata=archived,retained_results=retained,at_unix=time.time(),IL_and_other_RL_roots_excluded=True)
atomic(T/'CHECKPOINT_DELETE_PLAN.json',plan)
for root in paths:shutil.rmtree(root)
assert all(not Path(root).exists()for root in paths)
for item in retained:assert sha(Path(item['path']))==item['sha256'],item['path']
for a in sorted(expected):
 root=N/'runs'/a
 for name in ['latest.json','progress.json']:
  p=root/name
  if not p.exists():continue
  copy=T/'deleted_native_metadata'/a/('original_'+name);copy.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(p,copy)
  v=json.loads(p.read_text());v.update(checkpoint_deleted_by_user=True,recoverable=False,termination_record=str(N/'USER_TERMINATION.json'));atomic(p,v)
complete=dict(schema=plan['schema'],status='COMPLETE',completed_unix=time.time(),deleted_roots=paths,counts=dict(counts),retained_artifacts_verified=len(retained),all_recorded_retained_hashes_unchanged=True,native_metadata_archive=str(T/'deleted_native_metadata'),no_IL_or_other_RL_deletion=True,automatic_resume_authorized=False)
atomic(T/'CHECKPOINT_CLEANUP_COMPLETE.json',complete)
p=N/'USER_TERMINATION.json';v=json.loads(p.read_text());v.update(checkpoints_deleted_by_user=True,recoverable=False,checkpoint_cleanup=str(T/'CHECKPOINT_CLEANUP_COMPLETE.json'),required_retention=['training_source','configs','logs','metrics','full_evaluations','analysis_reports'],preserve_all_checkpoints_and_results=False,preserve_code_and_results=True);atomic(p,v)
p=N/'owners/ABCD_NATIVE/STATE.json';v=json.loads(p.read_text());v.update(checkpoints_deleted_by_user=True,recoverable=False)
for r in v['runs'].values():r.update(checkpoint_deleted_by_user=True,recoverable=False)
atomic(p,v)
print(json.dumps(complete))
