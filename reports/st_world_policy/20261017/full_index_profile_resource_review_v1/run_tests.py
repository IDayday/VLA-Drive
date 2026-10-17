import os, sys, time, threading, resource, signal, json, contextlib
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES="",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
os.sched_setaffinity(0,{0,1})
signal.alarm(600)
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path.cwd()))
sys.path.insert(0,str(Path.cwd()/"tests/st_world"))
out=Path(__file__).parent
stop=threading.Event();peak=[0]
def monitor():
    while not stop.wait(.05):
        rss=int(Path("/proc/self/statm").read_text().split()[1])*os.sysconf("SC_PAGE_SIZE")
        peak[0]=max(peak[0],rss)
        if rss>4*(1<<30):
            (out/"RSS_LIMIT_FAILURE.json").write_text(json.dumps({"rss":rss,"limit":4*(1<<30)}))
            os._exit(137)
threading.Thread(target=monitor,daemon=True).start()
start=time.monotonic()
with (out/"TESTS.log").open("w") as log, contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
    import pytest
    code=pytest.main(["-q","-p","no:cacheprovider","--basetemp=/tmp/full-index-profile-resource-rss-small","tests/st_world/test_provider_full_index_profile.py","tests/st_world/test_provider_profile_upgrade.py",str(out/"test_resource_profile.py")])
stop.set()
size=sum(p.stat().st_size for p in Path("/tmp/full-index-profile-resource-rss-small").rglob("*") if p.is_file())
r={"exit_code":int(code),"elapsed_seconds":time.monotonic()-start,"peak_rss_bytes":max(peak[0],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),"fixture_logical_bytes":size,"cpu_affinity":sorted(os.sched_getaffinity(0)),"rss_limit_bytes":4*(1<<30),"rss_poll_seconds":.05,"gpu_used":False,"full_manifest_read":False,"ledger_mutated":False}
(out/"TEST_RESOURCES.json").write_text(json.dumps(r,indent=2)+"\n")
print(json.dumps(r));print((out/"TESTS.log").read_text())
raise SystemExit(code)
