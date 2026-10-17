"""This review's bounded CPU-only synthetic test invocation, not a dispatcher."""
import json,os,signal,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
TMP=Path('/tmp/st-full-index-profile-independent-review-20261017')
if TMP.exists():raise SystemExit('Refuse to overwrite prior test artifacts')
TMP.mkdir()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
         MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(TMP))
suites=['test_provider_full_index_profile.py','test_provider_profile_upgrade.py','test_provider_scope_boundaries.py',
        'test_indexed_provider_factory.py','test_fixed_provider_training.py','test_training_config.py',
        'test_full_index_profile_independent_review.py']
cmd=['taskset','-c','0,1','/root/miniconda3/envs/ddp/bin/python','-m','pytest','-vv',
     '--basetemp',str(TMP/'pytest'),*[f'tests/st_world/{x}' for x in suites]]
started=time.monotonic();peak_rss=peak_disk=0;violation=None
with (OUT/'PYTEST_FULL.log').open('w') as log:
    proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    while proc.poll() is None:
        group=[]
        for path in Path('/proc').glob('[0-9]*/stat'):
            try:
                stat=path.read_text().rsplit(')',1)[1].split()
                if int(stat[2])==proc.pid:group.append(int(stat[21])*os.sysconf('SC_PAGE_SIZE'))
            except (OSError,ValueError,IndexError):continue
        peak_rss=max(peak_rss,sum(group))
        allocated=0
        for parent,dirs,files in os.walk(TMP):
            for name in files:
                try:allocated+=(Path(parent)/name).stat().st_blocks*512
                except FileNotFoundError:pass
        peak_disk=max(peak_disk,allocated)
        if time.monotonic()-started>600:violation='WALL_600_SECONDS'
        if peak_rss>4*1024**3:violation='RSS_4_GIB'
        if peak_disk>512*1024**2:violation='TEMP_DISK_512_MIB'
        if violation:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            break
        time.sleep(.5)
    code=proc.wait()
receipt=dict(command=cmd,working_directory=str(ROOT),exit_code=code,wall_s=time.monotonic()-started,
    observed_peak_process_group_rss_bytes=peak_rss,observed_peak_test_temp_allocated_bytes=peak_disk,
    sampling_interval_s=.5,violation=violation,cpu_affinity=[0,1],cuda_visible_devices='',
    real_dataset_reads=False,teacher_or_gpu_invocations=0,formal_or_real_profile_approval=False)
(OUT/'TEST_EXECUTION.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt));raise SystemExit(code or bool(violation))
