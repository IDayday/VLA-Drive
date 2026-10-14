"""Bounded native two-collection qualification of the recovered C numerical repair."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from starVLA.rl.constrained_flow_noise.contracts import require_dispatch, file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json
from .train_stable_probability import source_identity


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--values-proof',required=True)
    a=p.parse_args();host=require_dispatch();root=Path(a.root);out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    if 'probability_layout_qualification' not in str(out):raise ValueError('Explicit qualification path required')
    proof=json.loads(Path(a.values_proof).read_text())
    if proof['status']!='PASS' or proof['adapter_sha256']!=file_sha(Path(__file__).with_name('stable_saved_probability.py')):
        raise ValueError('Missing value and gradient proof')
    out.mkdir(parents=True)
    record=dict(status='RUNNING',host=host,execution_sha256=source_identity()['sha256'],steps=[],started_unix=time.time())
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',CUBLAS_WORKSPACE_CONFIG=':4096:8',
             OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NO_ALBUMENTATIONS_UPDATE='1',PYTHONUNBUFFERED='1')
    def execute(label,command):
        with (out/(label+'.log')).open('x') as stream:
            process=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        step=dict(label=label,pid=process.pid,args=command,started_unix=time.time())
        record['steps'].append(step);atomic_json(out/'status.json',record)
        step['returncode']=process.wait();step['finished_unix']=time.time();atomic_json(out/'status.json',record)
        if step['returncode']:raise RuntimeError('Qualification failed: '+label)
    def command(target,checkpoint,stop):
        args=list(json.loads((root/'formal_C_SDE_seed42_v1.launcher.json').read_text())['args'])
        i=args.index('scripts.analysis.constrained_flow_noise.train')
        args[i:i+1]=['scripts.analysis.constrained_flow_noise.train_stable_probability','--qualify-stop-update',str(stop),'--']
        args[0]=sys.executable;args[args.index('--output')+1]=str(target)
        args+=['--resume',str(checkpoint)]
        return args
    try:
        execute('continuous',command(out/'C_continuous',a.checkpoint,1854))
        execute('interrupted',command(out/'C_resumed',a.checkpoint,1851))
        cp=json.loads((out/'C_resumed/latest_checkpoint.json').read_text())['path']
        execute('resumed',command(out/'C_resumed',cp,1854))
        execute('verify',[sys.executable,'-m','scripts.analysis.constrained_flow_noise.verify_probability_recovery',
            '--root',str(out),'--checkpoint',a.checkpoint,'--original-run',str(root/'formal_C_SDE_seed42_v1'),
            '--values-proof',a.values_proof,'--batch-proof',str(root/'efficient_execution_qualification_v4/acceptance.json')])
        record.update(status='PASS',finished_unix=time.time())
    except BaseException as error:
        record.update(status='FAILED',error=repr(error),finished_unix=time.time());raise
    finally:atomic_json(out/'status.json',record)


if __name__=='__main__':main()
