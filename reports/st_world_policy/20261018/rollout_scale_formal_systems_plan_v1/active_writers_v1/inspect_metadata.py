"""Read-only resource/process metadata inventory; no data/model opens or signals."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time

MAX_COMMAND_BYTES = 16384

def query(argv):
    p = subprocess.run(argv, capture_output=True, text=True, timeout=10)
    return dict(exit_code=p.returncode, stdout=p.stdout[:65536], stderr=p.stderr[:4096])

def redact(argv):
    result=[]; secret_next=False
    for arg in argv:
        if secret_next:
            result.append('<redacted>'); secret_next=False; continue
        if re.search(r'(?i)(api[-_]?key|password|secret|access[-_]?token|auth[-_]?token)', arg):
            if '=' in arg: result.append(arg.split('=',1)[0]+'=<redacted>')
            else: result.append(arg); secret_next=True
        else: result.append(re.sub(r'(https?://)[^/\s:@]+:[^/\s@]+@',r'\1<redacted>@',arg))
    return result

clock = os.sysconf('SC_CLK_TCK')
btime = next(int(x.split()[1]) for x in Path('/proc/stat').read_text().splitlines() if x.startswith('btime '))

def process(pid):
    root=Path('/proc')/str(pid)
    parts=(root/'stat').read_text().rsplit(')',1)[1].split()
    with (root/'cmdline').open('rb') as f: argv=f.read(MAX_COMMAND_BYTES).decode('utf-8','replace').split('\0')
    argv=[x for x in argv if x]
    value=dict(pid=pid, ppid=int(parts[1]), state=parts[0], start_ticks=int(parts[19]),
               start_utc=datetime.datetime.fromtimestamp(btime+int(parts[19])/clock,datetime.timezone.utc).isoformat(),
               argv=redact(argv))
    for name in ['cwd','exe']:
        try:value[name]=os.readlink(root/name)
        except OSError as exc:value[name]='UNAVAILABLE:'+type(exc).__name__
    for name in ['io']:
        try:value[name]={k:int(v) for k,v in (line.split(':',1) for line in (root/name).read_text().splitlines())}
        except OSError:value[name]=None
    value['output_links']={}
    for fd in ['1','2']:
        try:value['output_links'][fd]=os.readlink(root/'fd'/fd)
        except OSError:value['output_links'][fd]=None
    value['open_project_write_paths']=[]
    try:
        for fd in sorted((root/'fd').iterdir(),key=lambda p:int(p.name))[:128]:
            try:
                target=os.readlink(fd)
                if not target.startswith('/mnt/project/'):continue
                info=(root/'fdinfo'/fd.name).read_text()
                flags=int(next(line.split(':',1)[1].strip() for line in info.splitlines() if line.startswith('flags:')),8)
                if flags & os.O_ACCMODE in [os.O_WRONLY,os.O_RDWR]:
                    value['open_project_write_paths'].append({'fd':int(fd.name),'path':target,'flags':oct(flags)})
            except (OSError,ValueError,StopIteration):continue
    except OSError:pass
    return value

result=dict(schema='read_only_active_writer_inventory_v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            host=socket.gethostname(),no_signals=True,no_model_cache_or_data_reads=True,
            disk_free_bytes=shutil.disk_usage('/mnt/project').free,
            project_device=Path('/mnt/project').stat().st_dev,cpu_affinity=sorted(os.sched_getaffinity(0)))
result['gpu']=query(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'])
result['compute_apps']=query(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'])
gpu_pids=set()
if result['compute_apps']['exit_code']==0:
    for line in result['compute_apps']['stdout'].splitlines():
        values=[x.strip() for x in line.split(',')]
        if len(values)>=2 and values[1].isdigit():gpu_pids.add(int(values[1]))
result['gpu_processes']=[];result['unmapped_gpu_pids']=[];table={}
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    pid=int(p.name)
    try:
        with (p/'cmdline').open('rb') as f: cmd=f.read(MAX_COMMAND_BYTES).decode('utf-8','replace')
        candidate=('/mnt/project/' in cmd and re.search(r'(?i)(train|torch|score|cache|worker|evaluate|launch|supervis)',cmd))
        if pid in gpu_pids or candidate:table[pid]=process(pid)
    except (OSError,ValueError,IndexError):continue
for pid in sorted(gpu_pids):
    if pid in table:result['gpu_processes'].append(table[pid])
    else:result['unmapped_gpu_pids'].append(pid)
selected=set(gpu_pids)
# Include known parent process identities; no unrelated process details are emitted.
for _ in range(6):
    parents={x['ppid'] for pid,x in table.items() if pid in selected}-selected
    if not parents:break
    for pid in parents:
        if pid==1:continue
        try:table[pid]=process(pid)
        except (OSError,ValueError,IndexError):continue
    selected.update(parents)
result['related_and_project_processes']=[x for pid,x in sorted(table.items()) if pid not in gpu_pids][:128]
result['process_count_emitted']=len(result['gpu_processes'])+len(result['related_and_project_processes'])
result['process_limit']=128
print(json.dumps(result,ensure_ascii=False))
