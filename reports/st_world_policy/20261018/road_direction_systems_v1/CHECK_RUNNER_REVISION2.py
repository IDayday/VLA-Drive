"""Independent CPU fixture invocation, no actual assets or backend calls."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
def audit(event,args):
    if event=="open" and isinstance(args[0],(str,bytes,os.PathLike)):
        p=os.fsdecode(args[0])
        if p.startswith("/mnt/project/") and any(k in p for k in ("/metric_cache","/rollouts/","/official_feedback/","/checkpoints/")):
            raise RuntimeError("Forbidden actual asset read in systems fixture")
sys.addaudithook(audit)
if __name__=="__main__":
    import contextlib,io,json,resource,time,pytest,torch
    before=resource.getrusage(resource.RUSAGE_SELF);start=time.monotonic();output=io.StringIO()
    with contextlib.redirect_stdout(output),contextlib.redirect_stderr(output):
        code=pytest.main(["-q","tests/st_world/test_run_road_direction.py"])
    after=resource.getrusage(resource.RUSAGE_SELF);child=resource.getrusage(resource.RUSAGE_CHILDREN)
    result={"scope":"independent CPU fixture rerun only","exit_code":code,
        "stdout_stderr":output.getvalue(),"wall_seconds":time.monotonic()-start,
        "parent_cpu_seconds":after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        "child_cpu_seconds":child.ru_utime+child.ru_stime,
        "peak_parent_rss_kib":after.ru_maxrss,"peak_child_rss_kib":child.ru_maxrss,
        "cuda_initialized":torch.cuda.is_initialized(),"real_asset_reads":0,"official_calls":0}
    (Path(__file__).resolve().parent/"RUNNER_TESTS_REVISION2.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2));raise SystemExit(code)
