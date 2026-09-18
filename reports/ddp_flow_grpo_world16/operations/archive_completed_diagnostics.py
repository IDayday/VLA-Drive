from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,os,shutil,time,threading
workspace=Path("/mnt/project/DriveDreamer-Policy-paired")
archive=Path("/root/ddp-flow-grpo-diagnostic-archive/20260918")
report=workspace/"reports/ddp_flow_grpo_world16/diagnostic_archive.jsonl"
lock=threading.Lock()
paths=[]
for name in ["resource_reallocation_v1","resource_reallocation_v2"]:
 for complete in (workspace/"runs"/name).glob("*/checkpoints/update_*/COMPLETE"):
  for path in complete.parent.rglob("*"):
   if path.suffix in {".pt",".bin",".pkl"} and path.is_file() and not path.is_symlink() and path.stat().st_size>1024**2:
    paths.append(path)
needed=sum(p.stat().st_size for p in paths)
archive.mkdir(parents=True,exist_ok=True)
if shutil.disk_usage(archive).free<needed+100*1024**3:raise RuntimeError("insufficient local archive headroom")
print(json.dumps({"files":len(paths),"bytes":needed,"destination":str(archive)}),flush=True)
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for chunk in iter(lambda:f.read(8*1024**2),b""):h.update(chunk)
 return h.hexdigest()
def copy(path):
 target=archive/path.relative_to(workspace)
 target.parent.mkdir(parents=True,exist_ok=True)
 temp=target.with_name(target.name+".copying")
 if target.exists() or temp.exists():raise FileExistsError(target)
 before=path.stat();h=hashlib.sha256();count=0;started=time.monotonic()
 with path.open("rb") as source,temp.open("xb") as dest:
  for chunk in iter(lambda:source.read(8*1024**2),b""):
   dest.write(chunk);h.update(chunk);count+=len(chunk)
   delay=count/(128*1024**2)-(time.monotonic()-started)
   if delay>0:time.sleep(delay)
  dest.flush();os.fsync(dest.fileno())
 after=path.stat()
 if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or count!=before.st_size or sha(temp)!=h.hexdigest():
  raise RuntimeError("diagnostic copy validation failed; original retained: "+str(path))
 os.replace(temp,target)
 link=path.with_name(path.name+".archive-link")
 link.symlink_to(target)
 os.replace(link,path)
 row={"time":time.time(),"host":"training-vla-zt-worker-0","original":str(path),"archive":str(target),"bytes":count,"sha256":h.hexdigest(),"status":"VERIFIED_COPY_AND_SYMLINK","scope":"completed diagnostic snapshot only; read archived state on this host"}
 with lock:
  with report.open("a") as log:log.write(json.dumps(row)+"\n");log.flush();os.fsync(log.fileno())
 return count
with ThreadPoolExecutor(max_workers=2) as pool:
 total=sum(pool.map(copy,paths))
(workspace/"runs/resource_reallocation_v2/diagnostic_archive_complete.json").write_text(json.dumps({"status":"PASS","files":len(paths),"bytes":total,"archive":str(archive)}))
print(json.dumps({"status":"PASS","files":len(paths),"bytes":total}),flush=True)
