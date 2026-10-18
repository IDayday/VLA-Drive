"""Independent CPU fixture/receipt review; refuses real campaign tensor reads."""
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

RUNTIME = Path('/mnt/project/VLA-Drive-st-consequence-diagnostics-runtime-v2-20261018')
REGISTRATION = '/mnt/project/st-world-policy-20261016/audits/consequence/gradient_balance_boundary3584_v1_controls/REGISTRATION.json'
OUT = Path(__file__).resolve().parent
PYTHON = '/root/miniconda3/envs/ddp/bin/python'
GUARD = '''
import os,sys
def audit(event,args):
    if event != 'open' or not isinstance(args[0],(str,bytes)): return
    p=os.fsdecode(args[0])
    if not p.startswith('/mnt/project/'): return
    if (p.endswith(('.pt','.pth','.safetensors','.xz','.jpg','.png','.jpeg'))
        or p.endswith('TRAINING_MANIFEST.json') or '/metric_cache/' in p):
        raise RuntimeError('Independent review forbids real tensor/image/manifest/cache read: '+p)
sys.addaudithook(audit)
'''

def main():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        PYTHONDONTWRITEBYTECODE='1')
    cases = [
        ('receipt_preflight', GUARD + '\nfrom tools.st_world_policy.audit_flow_gradient_balance import main\nraise SystemExit(main(["--registration",'+repr(REGISTRATION)+']))'),
        ('targeted_cpu_tests', GUARD + '\nimport pytest\nraise SystemExit(pytest.main(["-q","-p","no:cacheprovider","tests/st_world/test_audit_flow_gradient.py","tests/st_world/test_audit_flow_gradient_balance.py"]))'),
    ]
    results=[]
    for label,source in cases:
        before=resource.getrusage(resource.RUSAGE_CHILDREN);started=time.monotonic()
        p=subprocess.run([PYTHON,'-c',source],cwd=RUNTIME,env=env,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
        usage=resource.getrusage(resource.RUSAGE_CHILDREN)
        result={'case':label,'exit_code':p.returncode,'stdout':p.stdout,
            'wall_seconds':time.monotonic()-started,
            'cpu_seconds':usage.ru_utime+usage.ru_stime-before.ru_utime-before.ru_stime,
            'children_peak_rss_bytes':usage.ru_maxrss*1024}
        results.append(result)
        if len(p.stdout.encode())>1<<20: raise RuntimeError('Unexpected review output volume')
    record={'scope':'CPU fixtures and actual small immutable receipts only',
        'real_model_or_rollout_or_image_or_GT_or_metric_cache_reads':0,
        'GPU_used':False,'official_calls':0,'runtime':str(RUNTIME),
        'results':results,'passed':all(r['exit_code']==0 for r in results)}
    target=OUT/'PREPARATION_CHECKS.json'
    with target.open('x') as f:json.dump(record,f,indent=2)
    print(json.dumps({k:v for k,v in record.items() if k!='results'}))
    return 0 if record['passed'] else 1

if __name__=='__main__': raise SystemExit(main())
