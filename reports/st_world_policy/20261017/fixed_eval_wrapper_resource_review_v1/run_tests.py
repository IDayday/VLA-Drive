import os,sys,time,signal,threading,resource,json,contextlib
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES="",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
os.sched_setaffinity(0,{0,1});signal.alarm(300)
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path.cwd()))
out=Path(__file__).parent;stop=threading.Event();peak=[0]
def monitor():
    while not stop.wait(.05):
        rss=int(Path("/proc/self/statm").read_text().split()[1])*os.sysconf("SC_PAGE_SIZE")
        peak[0]=max(peak[0],rss)
        if rss>4*(1<<30):os._exit(137)
threading.Thread(target=monitor,daemon=True).start();start=time.monotonic()
with (out/"FINAL_TESTS.log").open("w") as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
    import pytest
    code=pytest.main(["-q","-p","no:cacheprovider","--basetemp=/tmp/st-fixed-eval-final-review","reports/st_world_policy/20261016/evaluation/FULL_PAYLOAD_C2_C3_1024_EXECUTION_v2/test_fixed_pairs.py",str(out/"test_independent.py")])
stop.set();report={"exit_code":int(code),"elapsed_seconds":time.monotonic()-start,"sampled_parent_peak_rss_bytes":peak[0],"linux_self_ru_maxrss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"cpu_affinity":sorted(os.sched_getaffinity(0)),"fixture_logical_bytes":sum(x.stat().st_size for x in Path("/tmp/st-fixed-eval-final-review").rglob("*") if x.is_file()),"gpu":False,"real_payload_reads":False}
(out/"FINAL_TEST_RESOURCES.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report));print((out/"FINAL_TESTS.log").read_text());raise SystemExit(code)
