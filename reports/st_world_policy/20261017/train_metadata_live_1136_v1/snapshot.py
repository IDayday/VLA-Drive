"""Bounded read-only committed metadata snapshot; never opens referenced image/raw paths."""
import os,json,time,hashlib,resource,signal
from pathlib import Path
os.sched_setaffinity(0,{0,1}); resource.setrlimit(resource.RLIMIT_AS,(2<<30,2<<30));signal.alarm(75)
started=time.time(); cap=760<<20; read=0; root=Path('/mnt/project/st-world-policy-20261016/caches/full_future_metadata_train_v1')
out=Path(__file__).parent
version=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
def proc(pid,ticks):
 p=Path('/proc')/str(pid); st=(p/'stat').read_text();a=st[st.rindex(')')+2:].split();cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
 assert int(a[19])==ticks and 'full_future_metadata_preparation_v1/run_build_once.py' in cmd
 assert (' --child ' in cmd)==(pid==480140)
 io={k:int(v) for k,v in (x.split(':') for x in (p/'io').read_text().splitlines())}
 return dict(pid=pid,start_ticks=int(a[19]),parent_pid=int(a[1]),state=a[0],cpu_seconds=(int(a[11])+int(a[12]))/os.sysconf('SC_CLK_TCK'),rss_bytes=int((p/'statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE'),io=io,elapsed_seconds=float(Path('/proc/uptime').read_text().split()[0])-ticks/os.sysconf('SC_CLK_TCK'),cmdline_exact_expected_role=True)
start=root/'CONTROL_BUILD_STARTED.json'; raw=start.read_bytes();read+=len(raw);control=json.loads(raw)
initial=[proc(480124,1366936766),proc(480140,1366936849)]
paths=sorted((root/'log_shards').iterdir()); records=[];errors=[];unique=set();logs=set()
for p in paths:
 s=p.lstat()
 if p.is_symlink() or not p.is_file() or not p.name.endswith('.json'):continue
 if s.st_size>128<<20 or read+s.st_size+1>cap:errors.append('READ_CAP_OR_SINGLE_FILE_LIMIT');break
 with p.open('rb') as f:
  before=os.fstat(f.fileno());raw=f.read(s.st_size+1);after=os.fstat(f.fileno())
 read+=len(raw)
 if version(before)!=version(after) or version(after)!=version(p.stat()):errors.append('SHARD_CHANGED');continue
 v=json.loads(raw);del raw
 assert v['schema']=='st_future_metadata_log_shard_v1' and len(v['scenes'])==len(v['selected'])
 assert v['log'] not in logs; logs.add(v['log'])
 keys=list(v['scenes'][0]) if v['scenes'] else []
 for scene in v['scenes']:
  key=scene.get('token',scene.get('scene_id'))
  if key is not None:unique.add(key)
 records.append(dict(shard=p.name,scenes=len(v['scenes']),logical_bytes=s.st_size,allocated_bytes=s.st_blocks*512,scene_keys=keys if not records else None,mtime_unix=s.st_mtime))
 del v
final=[proc(480124,1366936766),proc(480140,1366936849)]
# Only the already named cache root and its one log_shards level, no recursive project walk.
items=[root,*root.iterdir(),*(root/'log_shards').iterdir()]
statrows=[(p,p.lstat()) for p in items];logical=sum(s.st_size for p,s in statrows);charged=sum(max(s.st_size,s.st_blocks*512) for p,s in statrows)
scenes=sum(x['scenes'] for x in records);wall=time.time()-control['registered_unix'];full=101592
shard_bytes=sum(x['logical_bytes'] for x in records)
fs=os.statvfs(root)
result=dict(schema='bounded_train_metadata_live_snapshot_v1',snapshot_started_unix=started,snapshot_finished_unix=time.time(),source_root=str(root),start_receipt_sha256=hashlib.sha256(start.read_bytes()).hexdigest(),registered_unix=control['registered_unix'],limits=control['limits'],initial_processes=initial,final_processes=final,read_bytes=read,read_cap_bytes=cap,completed_shards=len(records),completed_scenes=scenes,unique_scene_ids_observed=len(unique),target_scenes=full,progress_fraction=scenes/full,shard_logical_bytes=shard_bytes,shard_allocated_bytes=sum(x['allocated_bytes'] for x in records),root_logical_bytes_including_directories=logical,root_charged_bytes_including_directories=charged,snapshot_root_entries=len(statrows),partial_files=[dict(name=str(p.relative_to(root)),bytes=s.st_size) for p,s in statrows if p.is_file() and ('.tmp' in p.name or '.partial' in p.name)],elapsed_build_wall_seconds=wall,remaining_wall_seconds=control['limits']['wall_seconds']-wall,live_parent_child_rchar=sum(x['io']['rchar'] for x in final),live_remaining_to_512GiB_unadjusted_baseline=control['limits']['read_bytes']-sum(x['io']['rchar'] for x in final),actual_complete_exists=(root/'COMPLETE.json').exists(),terminal_result_exists=(root/'CONTROL_BUILD_RESULT.json').exists(),shared_free_bytes=fs.f_bavail*fs.f_frsize,quota_verified=False,errors=errors,status='PARTIAL_READ_BOUND' if errors else 'LIVE_SNAPSHOT_COMPLETE_NOT_BUILD_COMPLETE',forecast={'method':'linear completed-scene exposure extrapolation, not a guarantee; excludes finalizer and log size variation','remaining_seconds_at_cumulative_scene_rate':wall*(full/scenes-1) if scenes else None,'final_committed_shard_logical_bytes_at_current_mean':shard_bytes*full/scenes if scenes else None},per_shard=records,review_process_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
(out/'SNAPSHOT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='per_shard'},indent=2))
