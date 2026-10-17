import os,stat,time,json,hashlib,resource,signal,re
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(2<<30,2<<30));os.sched_setaffinity(0,[0,1]);signal.alarm(180)
OUT=Path(__file__).parent;ROOT=Path('/mnt/project/st-world-policy-20261016/caches/full_future_metadata_train_v1')
STATE=OUT/'OBSERVATIONS.json';BUDGET=512<<20
state=json.loads(STATE.read_text()) if STATE.exists() else {'snapshots':[],'read_bytes':0,'completed':{}}
if len(state['snapshots'])>=2:raise RuntimeError('Exactly two snapshots maximum')
def read(path,cap):
    if state['read_bytes']+cap+1>BUDGET:raise RuntimeError('Review read budget exhausted')
    with Path(path).open('rb') as f:b=f.read(cap+1)
    state['read_bytes']+=len(b)
    if len(b)>cap:raise RuntimeError('Small control exceeds bound')
    return b
start_time=time.time();started_raw=read(ROOT/'CONTROL_BUILD_STARTED.json',65536);started=json.loads(started_raw)
assert started['controller']['sha256']=='a866ae4261cad36312355028cdfba3b24f52f64b174433635d0473420ddc84bf'
assert hashlib.sha256(read(started['controller']['path'],65536)).hexdigest()==started['controller']['sha256']
plan=json.loads(read(ROOT/'PLAN.json',1<<20));procs=[]
for pid,ticks,ischild in [(480124,1366936766,False),(480140,1366936849,True)]:
    p=Path('/proc')/str(pid)
    try:
        before=read(p/'stat',16384).decode().rsplit(') ',1)[1].split()
        cmd=[v.decode() for v in read(p/'cmdline',16384).split(b'\0') if v]
        assert int(before[19])==ticks and started['controller']['path'] in cmd
        assert ('--child' in cmd)==ischild
        assert started['admission']['path'] in cmd and started['admission']['sha256'] in cmd
        status=read(p/'status',16384).decode();rss=next(int(x.split()[1])*1024 for x in status.splitlines() if x.startswith('VmRSS:'))
        io={k:int(v) for k,v in (x.split(':') for x in read(p/'io',4096).decode().splitlines())}
        after=read(p/'stat',16384).decode().rsplit(') ',1)[1].split();assert int(after[19])==ticks
        paths=[]
        for fp in list((p/'fd').iterdir())[:128]:
            try:
                v=os.readlink(fp)
                if v.startswith(('/mnt/navsim','/mnt/project/st-world-policy')):paths.append(v)
            except FileNotFoundError:pass
        procs.append({'pid':pid,'start_ticks':ticks,'ppid':int(after[1]),'state':after[0],
          'utime_ticks':int(after[11]),'stime_ticks':int(after[12]),'rss_bytes':rss,'io':io,
          'cmdline':cmd,'cwd':os.readlink(p/'cwd'),'wchan':read(p/'wchan',4096).decode(),'open_data_paths':paths})
    except (FileNotFoundError,ProcessLookupError):
        procs.append({'pid':pid,'expected_start_ticks':ticks,'status':'EXITED_DURING_READ'})
charge=0;logical=0;entries=[];vanished=[];unread=[]
for directory in (ROOT,ROOT/'control',ROOT/'log_shards'):
    if not directory.exists():continue
    ds=directory.lstat();assert stat.S_ISDIR(ds.st_mode);charge+=max(ds.st_size,ds.st_blocks*512);logical+=ds.st_size
    for path in directory.iterdir():
        try:s=path.lstat()
        except FileNotFoundError:vanished.append(str(path));continue
        if stat.S_ISDIR(s.st_mode):continue
        assert stat.S_ISREG(s.st_mode)
        charge+=max(s.st_size,s.st_blocks*512);logical+=s.st_size
        if directory.name!='log_shards':continue
        row={'name':path.name,'bytes':s.st_size,'allocated_bytes':s.st_blocks*512,'mtime_ns':s.st_mtime_ns};entries.append(row)
        if not re.fullmatch('[0-9a-f]{64}\.json',path.name):continue
        if path.name in state['completed']:
            assert state['completed'][path.name]['bytes']==s.st_size and state['completed'][path.name]['mtime_ns']==s.st_mtime_ns
            continue
        tail=min(s.st_size,32768)
        while True:
            if state['read_bytes']+tail>BUDGET:unread.append(path.name);break
            with path.open('rb') as f:f.seek(s.st_size-tail);b=f.read(tail)
            state['read_bytes']+=len(b);i=b.rfind(b'\n  "selected": [')
            if i>=0:
                selected=json.loads(b'{'+b[i:].strip())['selected'];logs={v['log'] for v in selected};assert len(logs)==1
                a=path.stat();assert (a.st_size,a.st_mtime_ns)==(s.st_size,s.st_mtime_ns)
                state['completed'][path.name]={'log':logs.pop(),'scenes':len(selected),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns};break
            if tail==s.st_size or tail>=1<<20:unread.append(path.name);break
            tail=min(s.st_size,tail*2)
now=time.time();done={v['log'] for v in state['completed'].values()};remaining=[v for v in plan['raw_log_stats'] if v['log'] not in done]
partials=[v for v in entries if not re.fullmatch('[0-9a-f]{64}\.json',v['name'])];files=[v for v in entries if re.fullmatch('[0-9a-f]{64}\.json',v['name'])]
active=[v for v in plan['selected_logs'] if v not in done][:1]
snap={'unix':now,'capture_start_unix':start_time,'elapsed_seconds':now-started['registered_unix'],
'identity_verified':len(procs)==2 and all('io' in v for v in procs),'processes':procs,
'aggregate_rss_bytes':sum(v.get('rss_bytes',0) for v in procs),'running_parent_child_absolute_rchar':sum(v.get('io',{}).get('rchar',0) for v in procs),
'completed_file_count':len(files),'parsed_complete_logs':len(state['completed']),'completed_scenes':sum(v['scenes'] for v in state['completed'].values()),
'expected_logs':1176,'expected_scenes':101592,'unread_complete_metadata':unread,'partial_files':partials,'partial_logical_bytes':sum(v['bytes'] for v in partials),
'completed_shard_bytes':sum(v['bytes'] for v in files),'root_inclusive_charge_bytes':charge,'root_inclusive_logical_bytes':logical,
'root_cap_bytes':2<<30,'root_remaining_bytes':(2<<30)-charge,'wall_cap_seconds':14400,'wall_remaining_seconds':14400-(now-started['registered_unix']),
'logical_read_cap_bytes':512<<30,'COMPLETE_exists':(ROOT/'COMPLETE.json').exists(),'RESULT_exists':(ROOT/'CONTROL_BUILD_RESULT.json').exists(),
'candidate_next_log_in_plan_order':active,'candidate_next_raw_stats':[v for v in remaining if v['log'] in active],
'largest_remaining_raw_logs':sorted(remaining,key=lambda v:v['bytes'],reverse=True)[:3],'vanished_during_stat':vanished,'non_atomic_snapshot':True,
'review_cumulative_bytes_read':state['read_bytes'],'start_receipt_sha256':hashlib.sha256(started_raw).hexdigest()}
state['snapshots'].append(snap);STATE.write_text(json.dumps(state,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:v for k,v in snap.items() if k not in ('processes','partial_files','largest_remaining_raw_logs')},sort_keys=True))
