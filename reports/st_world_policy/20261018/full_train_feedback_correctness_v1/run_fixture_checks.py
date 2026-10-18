"""Bounded local fixture execution only; captures CPU/RSS and full pytest logs."""
import json
import os
from pathlib import Path
import resource
import subprocess
import time

HERE=Path(__file__).resolve().parent
SOURCE=Path('/mnt/project/VLA-Drive-st-consequence-resource-review-20261018')
REVIEW=HERE.parents[3]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
         OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
         FULL_FEEDBACK_REVIEW_SOURCE_ROOT=str(SOURCE))
results=[]
for label,root,test in [('AUTHOR',SOURCE,'tests/st_world/test_full_feedback_support.py'),
                        ('INDEPENDENT',REVIEW,str(HERE/'test_independent_math.py'))]:
    command=['/root/miniconda3/envs/ddp/bin/python','-m','pytest','-q','-p','no:cacheprovider',test]
    before=resource.getrusage(resource.RUSAGE_CHILDREN)
    start=time.monotonic()
    def limits():resource.setrlimit(resource.RLIMIT_CPU,(55,55))
    with (HERE/(label+'_TEST_LOG.txt')).open('x') as output:
        completed=subprocess.run(command,cwd=root,env=env,stdout=output,stderr=subprocess.STDOUT,
                                 timeout=60,preexec_fn=limits)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    result={'label':label,'command':command,'cwd':str(root),'exit_code':completed.returncode,
            'wall_seconds':time.monotonic()-start,'cpu_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
            'child_cumulative_peak_rss_bytes':after.ru_maxrss*1024}
    results.append(result)
    assert completed.returncode==0,result
    assert after.ru_maxrss*1024<=2<<30,result
(HERE/'FIXTURE_EXECUTION.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results))
