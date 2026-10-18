"""Systems CLI check only; refuses all campaign payload access."""
import contextlib, io, json, os, resource, runpy, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
opens=[]
def audit(event,args):
    if event != "open" or not isinstance(args[0],(str,bytes,os.PathLike)): return
    p=os.fsdecode(args[0])
    if p.startswith("/mnt/project/") and ("/metric_cache" in p or "/rollouts/" in p or "/official_feedback/" in p or "/checkpoints/" in p):
        raise RuntimeError("Forbidden real-asset read during systems review: "+p)
    if p.startswith(str(ROOT)): opens.append(p)
sys.addaudithook(audit)
start=time.monotonic(); cpu=resource.getrusage(resource.RUSAGE_SELF)
ns=runpy.run_path(str(ROOT/"tools/st_world_policy/diagnose_road_direction.py"),run_name="systems_review_module")
def call(args):
    o,e=io.StringIO(),io.StringIO()
    with contextlib.redirect_stdout(o),contextlib.redirect_stderr(e):
        try: code=ns["main"](args)
        except SystemExit as ex: code=ex.code
    return {"args":args,"exit_code":code,"stdout":o.getvalue(),"stderr":e.getvalue()}
path=OUT/"PREPARATION_WITNESS.json"
rows=[call(["--help"]),call(["--mode","prepare","--output",str(path)]),
      call(["--mode","prepare","--output",str(path)]),
      call(["--mode","execute","--output",str(OUT/"NEVER_EXECUTED.json")])]
assert [r["exit_code"] for r in rows]==[0,0,2,2],rows
p=json.loads(path.read_text());assert p["status"]=="PREPARED_NOT_EXECUTED" and not p["dispatch_authorized"]
assert p["new_official_calls"]==0 and p["gpu_calls"]==0
import torch
assert not torch.cuda.is_initialized()
end=resource.getrusage(resource.RUSAGE_SELF)
result={"status":"PASS_PREPARATION_ONLY","rows":rows,"wall_seconds":time.monotonic()-start,
 "cpu_user_seconds":end.ru_utime-cpu.ru_utime,"cpu_system_seconds":end.ru_stime-cpu.ru_stime,
 "peak_rss_kib":end.ru_maxrss,"gpu_initialized":torch.cuda.is_initialized(),
 "real_asset_reads":0,"official_calls":0,"training_updates":0,
 "source_open_paths":sorted(set(opens)),"scope":"Actual CLI prepare/help/refusal, imports permitted; no actual scene data or scoring"}
(OUT/"CLI_CHECKS.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items() if k not in ("rows","source_open_paths")},indent=2))
