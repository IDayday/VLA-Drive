"""Dispatch the registered paired single-ODE readouts on one allowed host.

Two cohorts can share each GPU; each inference process retains the unchanged
raw-observation actor. No extra candidate selection or inference seeds.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
from starVLA.rl.simwam_recipe.contracts import ROOT, file_sha


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--host',required=True);p.add_argument('--export',required=True)
    p.add_argument('--export-sha256',required=True);p.add_argument('--label',required=True)
    p.add_argument('--cohorts',nargs='+',choices=['DEV','TRAIN_DIAGNOSTIC','NAVTEST'],default=['DEV','TRAIN_DIAGNOSTIC'])
    p.add_argument('--final-lock');p.add_argument('--repo',default=str(Path.cwd()))
    p.add_argument('--root',default=ROOT)
    a=p.parse_args()
    if not a.label.replace('_','').replace('-','').isalnum():raise ValueError('Unsafe run label')
    policy=json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    if (a.host not in policy['task_authorizations']['flow_safe_improvement_20261013']['allowed_hosts']
        or a.host not in policy['allowed_hosts']
        or policy.get('global_dispatch_prohibitions',{}).get(a.host,{}).get('new_task_dispatch_allowed') is False):
        raise PermissionError(a.host)
    if 'NAVTEST' in a.cohorts and not a.final_lock:raise ValueError('Navtest needs final lock')
    if len(a.cohorts)>2:raise ValueError('At most two registered cohorts per GPU')
    python='/mnt/project/flow-safe-improvement-20261013/venv/bin/python'
    payload=dict(vars(a),python=python,
                 evaluation_source_sha256=file_sha(Path(a.repo)/'scripts/analysis/simwam_recipe/evaluate_policy.py'))
    code='''import json,os,subprocess,time
from pathlib import Path
p=json.loads(PAYLOAD)
out=Path(p['root'])/'evaluation_dispatch'/p['label'];out.mkdir(parents=True,exist_ok=True)
free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
if len(free.splitlines())!=8 or min(map(int,free.splitlines()))<12000*len(p['cohorts']):
 raise RuntimeError('Insufficient live free GPU memory; no eviction')
for cohort in p['cohorts']:
 for gpu in range(8):
  suffix=cohort+'_rank'+str(gpu);receipt=out/(suffix+'.json')
  output=Path(p['root'])/'evaluations'/(p['label']+'_'+cohort)
  complete=output/('rank'+str(gpu)+'.json')
  if complete.exists() and json.loads(complete.read_text())['status']=='ASSIGNED_COMPLETE':
   print('COMPLETE',suffix);continue
  if receipt.exists():
   old=json.loads(receipt.read_text());proc=Path('/proc')/str(old['pid'])
   if proc.exists() and 'scripts.analysis.simwam_recipe.evaluate_policy' in (proc/'cmdline').read_bytes().decode(errors='replace'):
    print('ALREADY_RUNNING',suffix,old['pid']);continue
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),LOCAL_RANK='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',NO_ALBUMENTATIONS_UPDATE='1')
  cmd=[p['python'],'-m','scripts.analysis.simwam_recipe.evaluate_policy','--export',p['export'],'--export-sha256',p['export_sha256'],'--output',str(output),'--cohort',cohort,'--shard',str(gpu),'--shards','8']
  if p['final_lock']:cmd+=['--final-lock',p['final_lock']]
  log=(out/(suffix+'.log')).open('ab',buffering=0)
  child=subprocess.Popen(cmd,cwd=p['repo'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  receipt.write_text(json.dumps(dict(p,pid=child.pid,GPU=gpu,cohort=cohort,output=str(output),command=cmd,started=time.time()),indent=2)+'\\n')
  print('STARTED',suffix,child.pid)
'''.replace('PAYLOAD',repr(json.dumps(payload)))
    subprocess.run(['ssh','-o','BatchMode=yes',a.host,shlex.join([python,'-c',code])],check=True)


if __name__=='__main__':main()
