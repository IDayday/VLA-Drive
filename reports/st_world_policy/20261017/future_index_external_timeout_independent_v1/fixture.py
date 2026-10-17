"""Synthetic process fixture: no actual task, ledger, data or model execution."""
import hashlib,json,os,runpy,sys,time
from pathlib import Path
from types import ModuleType,SimpleNamespace
BASE=Path('/mnt/project/VLA-Drive-st-future-finalize-runtime-v1-20261017/reports/st_world_policy/20261017')
CORE=BASE/'full_future_metadata_preparation_v1/run_build_once.py'
CORE_SHA='4b2f0fccfd4bf3d655198f61179c88f6b2a51840731465c720b8addb6169b1a2'
ROOT=Path(sys.argv[2])
if sys.argv[1]=='fifo':
    from unittest.mock import patch
    source=Path('/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261017/full_future_index_execution_v1/run_index_once.py')
    original=Path.read_bytes
    def read(path):
        if path==CORE:
            (ROOT/'fifo_reached.json').write_text(json.dumps({'pid':os.getpid(),'core':str(path)}))
            return original(ROOT/'core.fifo')
        return original(path)
    with patch.object(Path,'read_bytes',read):runpy.run_path(str(source),run_name='__main__')
elif sys.argv[1]=='monitor':
    raw=CORE.read_bytes();assert hashlib.sha256(raw).hexdigest()==CORE_SHA
    core=ModuleType('actual_reviewed_monitor');core.__file__=str(CORE)
    exec(compile(raw,str(CORE),'exec'),core.__dict__)
    helper=BASE/'spatial_probe_actual_preparation_v1'
    pins={'supervise_once.py':'d829cf48764b23b5fce08cdd6ea45c6d3e61e633763ad2cb876e5ab8d96d8902',
          'observed_cli.py':'c1af891b440663d75ca3b75c1227f5a7d4247d62319a566214848e0a5550f643',
          'prepare_bindings.py':'31d2d302d5dd441df9251762e34cd9d1762afc3463cdabfe8dbf5b1f12577c84'}
    for name,sha in pins.items():assert hashlib.sha256((helper/name).read_bytes()).hexdigest()==sha
    sys.path.insert(0,str(helper))
    import supervise_once as s
    import observed_cli as obs
    # Only memory sampling is substituted: this test covers real process ownership,
    # signals and cleanup, not metadata provider or storage accounting.
    def rss(pid):
        return int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    with (ROOT/'fixture.lock').open('w') as lock:
        result=core.monitor([sys.executable,str(Path(__file__).resolve()),'worker',str(ROOT)],
            runtime=str(ROOT),env=dict(os.environ),fd=lock.fileno(),root=ROOT,s=s,
            m=SimpleNamespace(process_tree_rss=rss),obs=obs,
            budget={'wall_seconds':20,'cleanup_seconds':40,'read_bytes':64<<20,'rss_bytes':1<<30},cpu_affinity=(8,9))
    (ROOT/'monitor_result.json').write_text(json.dumps(result,indent=2)+'\n')
elif sys.argv[1]=='worker':
    import subprocess
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
    rows=[]
    for pid in (os.getpid(),child.pid):
        raw=Path(f'/proc/{pid}/stat').read_text();rows.append({'pid':pid,'start_ticks':int(raw[raw.rfind(")")+2:].split()[19])})
    (ROOT/'owned_fixture_pids.json').write_text(json.dumps(rows))
    time.sleep(60)
else:raise ValueError('Unknown synthetic mode')
