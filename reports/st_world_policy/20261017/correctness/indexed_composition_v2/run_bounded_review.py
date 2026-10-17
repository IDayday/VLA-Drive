"""CPU-only reviewer harness; watch this owned child against 2 GiB / 600 s."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

SOURCE = '42854f40d0dab551463d63abae80a1045b64722f'
ROOT = Path('/mnt/project/VLA-Drive-st-indexed-composition-review-runtime-42854f40')
HERE = Path(__file__).resolve().parent
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip() == SOURCE
assert not subprocess.check_output(['git','status','--porcelain=v1'],cwd=ROOT,text=True).strip()
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and len(os.sched_getaffinity(0)) <= 2
paths = ['tools/st_world_policy/qualify_indexed_provider.py','tools/st_world_policy/teacher_provider.py']
hashes = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
tests = [str(HERE.parent/'indexed_composition_v1/test_composition_independent.py'),
    str(HERE/'test_counter_consistency.py'),
    'tests/st_world/test_indexed_composition.py::test_receipt_strict_full_tensor_summaries_and_integer_ledgers',
    'tests/st_world/test_indexed_composition.py::test_actual_adapter_layouts_masks_rng_and_factory_admission',
    'tests/st_world/test_indexed_composition.py::test_multiple_scenes_reorder_shared_keys_and_distinct_missing_masks']
program = 'import os,sys;sys.path[:0]=[os.getcwd(),os.getcwd()+"/tests/st_world"];import pytest;raise SystemExit(pytest.main('+repr(['-vv','-p','no:cacheprovider',*tests])+'))'
started=time.monotonic(); peak=0; outcome='COMPLETED'; budget=2 << 30
with (HERE/'retest.log').open('x') as log:
    child=subprocess.Popen([sys.executable,'-c',program],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    while child.poll() is None:
        try:
            data=Path(f'/proc/{child.pid}/status').read_text()
            values={line.split(':',1)[0]:line.split(':',1)[1] for line in data.splitlines() if ':' in line}
            peak=max(peak,int(values.get('VmHWM','0 kB').split()[0])*1024)
        except FileNotFoundError:
            pass
        if peak>budget or time.monotonic()-started>600:
            outcome='RESOURCE_LIMIT';os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            break
        time.sleep(.1)
code=child.wait()
assert hashes == {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
assert not subprocess.check_output(['git','status','--porcelain=v1'],cwd=ROOT,text=True).strip()
report={'tested_source':SOURCE,'source_sha256':hashes,'runtime':str(ROOT),'runtime_clean':True,
    'exit_code':code,'outcome':outcome,'wall_seconds':time.monotonic()-started,
    'child_observed_VmHWM_bytes':peak,'rss_limit_bytes':budget,'wall_limit_seconds':600,
    'cpu_affinity':sorted(os.sched_getaffinity(0)),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
    'tests':tests,'real_teacher_qualification':False,'read_production_data_or_GPU':False}
with (HERE/'RUN.json').open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
print(json.dumps(report))
raise SystemExit(code if outcome=='COMPLETED' else 3)
