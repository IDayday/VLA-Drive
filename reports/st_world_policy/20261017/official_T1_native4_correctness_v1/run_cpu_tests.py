"""Bounded synthetic/config CPU tests for this independent review only."""
import json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
TMP=Path('/tmp/st-official-t1-native4-correctness-20261017')
if TMP.exists():raise SystemExit('Prior test evidence exists; no overwrite')
TMP.mkdir()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
    OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
    PYTHONPATH=str(ROOT)+':'+str(ROOT/'tests/st_world'),TMPDIR=str(TMP))
env.pop('ST_TEST_NATIVE_CPU',None)
cmd=['taskset','-c','0,1','/root/miniconda3/envs/ddp/bin/python','-m','pytest','-vv',
    '--basetemp',str(TMP/'pytest'),'tests/st_world/test_budget_extension.py',
    'tests/st_world/test_training_state.py','tests/st_world/test_official_runtime.py',
    'tests/st_world/test_official_training_integration.py',
    'tests/st_world/test_fp32_master_init.py::test_native_resume_and_budget_continuation_are_unconditional_noops',
    str(OUT/'test_preparation.py')]
start=time.monotonic();peak_rss=peak_output=0;reason=None
with (OUT/'PYTEST_FULL.log').open('w') as log:
 p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 while p.poll() is None:
  rss=0
  for path in Path('/proc').glob('[0-9]*/stat'):
   try:
    fields=path.read_text().rsplit(')',1)[1].split()
    if int(fields[2])==p.pid:rss+=int(fields[21])*os.sysconf('SC_PAGE_SIZE')
   except (OSError,ValueError,IndexError):continue
  peak_rss=max(peak_rss,rss);size=0
  for directory in (TMP,OUT):
   for base,dirs,files in os.walk(directory):
    for name in files:
     try:size+=(Path(base)/name).stat().st_blocks*512
     except FileNotFoundError:pass
  peak_output=max(peak_output,size)
  if time.monotonic()-start>300:reason='WALL_300S'
  if peak_rss>4*1024**3:reason='RSS_4GiB'
  if peak_output>16*1024**2:reason='OUTPUT_16MiB'
  if reason:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   break
  time.sleep(.5)
 code=p.wait()
receipt=dict(command=cmd,exit_code=code,wall_seconds=time.monotonic()-start,
 observed_peak_group_rss_bytes=peak_rss,observed_peak_output_allocated_bytes=peak_output,
 sampling_interval_seconds=.5,cpu_affinity=[0,1],library_threads=2,cuda_visible_devices='',
 violation=reason,actual_model_image_target_payload_reads=0,actual_native_checkpoint_test=False)
(OUT/'TEST_EXECUTION.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt));raise SystemExit(code or bool(reason))
