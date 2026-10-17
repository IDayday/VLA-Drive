import os,stat,time,json,hashlib,resource
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(512<<20,512<<20))
os.sched_setaffinity(0,[0])
OUT=Path(__file__).parent;ROOT=Path('/mnt/project/st-world-policy-20261016/caches/full_future_metadata_train_v1')
statepath=OUT/'OBSERVATIONS.json';state=json.loads(statepath.read_text()) if statepath.exists() else {'snapshots':[],'read_bytes':0,'completed':{}}
assert len(state['snapshots'])<2
BUDGET=15<<20

def read(path,cap=1<<20):
    p=Path(path)
    with p.open('rb') as f:b=f.read(cap+1)
    state['read_bytes']+=len(b)
    if len(b)>cap or state['read_bytes']>BUDGET:raise RuntimeError('bounded read exceeded')
    return b
plan=json.loads(read(ROOT/'PLAN.json'));started=json.loads(read(ROOT/'CONTROL_BUILD_STARTED.json'))
controller=started['controller']['path'];procs=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        cmd=read(p/'cmdline',16384).split(b'\0')
        if controller.encode() not in cmd:continue
        raw=read(p/'stat',16384).decode();t=raw.rsplit(') ',1)[1].split()
        status=read(p/'status',16384).decode();rss=next(int(l.split()[1])*1024 for l in status.splitlines() if l.startswith('VmRSS:'))
        io={k:int(v) for k,v in (l.split(':') for l in read(p/'io',4096).decode().splitlines())}
        fds=[]
        for f in list((p/'fd').iterdir())[:128]:
            try:
                v=os.readlink(f)
                if v.startswith(('/mnt/navsim','/mnt/project/st-world-policy')):fds.append(v)
            except FileNotFoundError:pass
        procs.append({'pid':int(p.name),'ppid':int(t[1]),'state':t[0],'start_ticks':int(t[19]),'utime_ticks':int(t[11]),'stime_ticks':int(t[12]),'RSS':rss,'IO':io,'cmdline':[x.decode() for x in cmd if x],'cwd':os.readlink(p/'cwd'),'open_data_paths':fds,'wchan':read(p/'wchan',4096).decode()})
    except (FileNotFoundError,ProcessLookupError):continue
rootbytes=0;files=[]
for d in [ROOT,ROOT/'control',ROOT/'log_shards']:
    z=d.lstat();rootbytes+=max(z.st_size,z.st_blocks*512)
    for p in d.iterdir():
        z=p.lstat()
        if stat.S_ISDIR(z.st_mode):continue
        assert stat.S_ISREG(z.st_mode)
        rootbytes+=max(z.st_size,z.st_blocks*512)
        if d.name!='log_shards':continue
        files.append({'name':p.name,'bytes':z.st_size,'allocated_bytes':z.st_blocks*512,'mtime_ns':z.st_mtime_ns})
        if p.suffix!='.json' or p.name in state['completed']:continue
        size=z.st_size;tail=min(size,32768)
        while True:
            if state['read_bytes']+tail>BUDGET:raise RuntimeError('tail budget exhausted')
            with p.open('rb') as f:f.seek(size-tail);b=f.read(tail)
            state['read_bytes']+=len(b)
            start=b.rfind(b'\n  "selected": [')
            if start>=0:
                rows=json.loads(b'{'+b[start:].strip())['selected']
                logs={x['log'] for x in rows};assert len(logs)==1
                state['completed'][p.name]={'log':logs.pop(),'scenes':len(rows),'bytes':size,'mtime_ns':z.st_mtime_ns};break
            if tail==size or tail>=1<<20:raise RuntimeError('bounded selected-tail not found')
            tail=min(size,tail*2)
now=time.time();done=set(p['log'] for p in state['completed'].values());raw=plan['raw_log_stats'];remaining=[x for x in raw if x['log'] not in done]
active=[x for x in plan['selected_logs'] if x not in done][:1]
snap={'unix':now,'elapsed_from_start_seconds':now-started['registered_unix'],'processes':procs,'process_tree_RSS':sum(p['RSS'] for p in procs),'conservative_running_sum_absolute_rchar':sum(p['IO']['rchar'] for p in procs),'log_shard_files':files,'completed_logs':len(state['completed']),'completed_scenes':sum(p['scenes'] for p in state['completed'].values()),'expected_total_logs':1176,'expected_total_scenes':101592,'candidate_next_log_in_original_order':active,'candidate_next_raw_stats':[x for x in raw if x['log'] in active],'largest_remaining_raw_logs':sorted(remaining,key=lambda x:x['bytes'],reverse=True)[:3],'root_inclusive_charge_bytes':rootbytes,'root_cap_bytes':2<<30,'remaining_root_cap_bytes':(2<<30)-rootbytes,'source_read_cap_bytes':512<<30,'remaining_logical_read_lower_bound':(512<<30)-sum(p['IO']['rchar'] for p in procs),'wall_cap_seconds':14400,'COMPLETE_exists':(ROOT/'COMPLETE.json').exists(),'RESULT_exists':(ROOT/'CONTROL_BUILD_RESULT.json').exists(),'review_cumulative_bytes_read':state['read_bytes']}
state['snapshots'].append(snap);statepath.write_text(json.dumps(state,indent=2,sort_keys=True)+'\n')
print(json.dumps(snap,indent=2))
