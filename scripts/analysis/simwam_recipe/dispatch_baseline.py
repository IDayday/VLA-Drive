"""Launch the fixed missing single-ODE TRAIN screen on allowed permanent hosts."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import argparse
from starVLA.rl.simwam_recipe.contracts import ROOT,file_sha


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--lanes',type=int,default=1);args=parser.parse_args()
    if args.lanes not in (1,2):raise ValueError('Only qualified one/two-worker GPU layouts')
    repo=str(Path.cwd());python='/mnt/project/flow-safe-improvement-20261013/venv/bin/python'
    policy=json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    hosts=['training-vlawm-tsinghua-zt','training-vlawm-tsinghua-zt3','training-vlawm-tsinghua-zt4']
    allowed=policy['task_authorizations']['flow_safe_improvement_20261013']['allowed_hosts']
    for host in hosts:
        if host not in allowed or host not in policy['allowed_hosts']:raise PermissionError(host)
    root=Path(ROOT);(root/'baseline_dispatch').mkdir(parents=True,exist_ok=True)
    # Reuse the four exact original-observation comparisons and actual score smoke.
    q=json.loads((root/'SFT_NATIVE_TRAIN_seed2101/rank00.json').read_text())
    if q['new_scores']<2:raise ValueError('Real scoring smoke has not completed')
    for host_index,host in enumerate(hosts):
        payload=dict(repo=repo,python=python,root=ROOT,host=host,index=host_index,lanes=args.lanes,
                     source_sha256=file_sha('scripts/analysis/simwam_recipe/baseline_train.py'))
        code='''import json,os,subprocess,time
from pathlib import Path
p=json.loads(PAYLOAD)
out=Path(p['root'])/'baseline_dispatch'
for assignment in range(8*p['lanes']):
 gpu=assignment//p['lanes'];lane=assignment%p['lanes'];shard=p['index']*8+gpu
 suffix=('shard%02d'%shard) if p['lanes']==1 else ('shard%02d_lane%dof%d'%(shard,lane,p['lanes']))
 receipt=out/(suffix+'.json')
 if receipt.exists():
  previous=json.loads(receipt.read_text());proc=Path('/proc')/str(previous['pid'])
  if proc.exists() and 'scripts.analysis.simwam_recipe.baseline_train' in (proc/'cmdline').read_bytes().decode(errors='replace'):
   print('ALREADY_RUNNING',shard,previous['pid']);continue
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',NO_ALBUMENTATIONS_UPDATE='1')
 cmd=[p['python'],'-m','scripts.analysis.simwam_recipe.baseline_train','--shard',str(shard),'--shards','24','--lane',str(lane),'--lanes',str(p['lanes'])]
 log=(out/(suffix+'.log')).open('ab',buffering=0)
 child=subprocess.Popen(cmd,cwd=p['repo'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 d=dict(p,shard=shard,lane=lane,GPU=gpu,pid=child.pid,started=time.time(),command=cmd)
 receipt.write_text(json.dumps(d,indent=2)+'\\n');print('STARTED',shard,child.pid)
'''.replace('PAYLOAD',repr(json.dumps(payload)))
        subprocess.run(['ssh','-o','BatchMode=yes',host,shlex.join([python,'-c',code])],check=True)


if __name__=='__main__':main()
