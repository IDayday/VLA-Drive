"""Launch exactly one registered formal variant from a frozen source snapshot."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
from starVLA.rl.simwam_recipe.contracts import ROOT,file_sha,load_recipe
from scripts.analysis.simwam_recipe.registered_followup import FOLLOWUPS,require_registration


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--variant',choices=['R_SOURCE','R_CANONICAL',*FOLLOWUPS],required=True)
    p.add_argument('--host',required=True);p.add_argument('--snapshot',required=True);p.add_argument('--resume')
    p.add_argument('--root',default=ROOT)
    p.add_argument('--qualification',default='eight_gpu_recovery_v3_fixed_buckets')
    a=p.parse_args();root=Path(a.root);assets=Path(ROOT);snapshot=Path(a.snapshot).resolve()
    if a.variant in FOLLOWUPS:
        require_registration(a.variant,root)
        if json.loads((root/'qualification/cpu_contract.json').read_text())['status']!='PASS':
            raise ValueError('Follow-up contract not qualified')
    elif root.resolve()!=assets.resolve():
        raise ValueError('Historical variants retain their original run root')
    policy=json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    if (a.host not in policy['task_authorizations']['flow_safe_improvement_20261013']['allowed_hosts']
        or a.host not in policy['allowed_hosts']
        or policy.get('global_dispatch_prohibitions',{}).get(a.host,{}).get('new_task_dispatch_allowed') is False):raise PermissionError(a.host)
    source=json.loads((snapshot/'SOURCE_IDENTITY.json').read_text())
    qualification=root/'qualification'/a.qualification
    if json.loads((qualification/'result.json').read_text())['status']!='PASS':raise ValueError('Eight-rank exact recovery not qualified')
    qi=json.loads((qualification/'continuous/identity.json').read_text())
    for relative,sha in qi['source_files'].items():
        if file_sha(snapshot/relative)!=sha:raise ValueError('Formal core differs from qualified core: '+relative)
    manifest=assets/'hard_manifest.json';hard=json.loads(manifest.read_text())
    if hard['status']!='COMPLETE_FIXED_SFT_SINGLE_NATIVE_ODE_HARD_SUBSET' or hard['full_TRAIN_scenes']!=101592:raise ValueError('Incomplete fixed hard population')
    for name in ('math_extended.json','lora.json','source_engine.json','raw_observation_export.json'):
        if json.loads((assets/'qualification'/name).read_text())['status']!='PASS':raise ValueError('Failed acceptance: '+name)
    for cohort in ('DEV','TRAIN_DIAGNOSTIC'):
        baseline=assets/'evaluations'/('SFT_update000000_'+cohort)
        for rank in range(8):
            if json.loads((baseline/f'rank{rank}.json').read_text())['status']!='ASSIGNED_COMPLETE':raise ValueError('SFT baseline incomplete')
    config_name=FOLLOWUPS[a.variant]['config'] if a.variant in FOLLOWUPS else {'R_SOURCE':'source','R_CANONICAL':'canonical'}[a.variant]
    config='configs/flow_rl/simwam_'+config_name+'_ddp.yaml'
    cfg=load_recipe(snapshot/config)
    expected=dict(qi['recipe']);expected['variant']=a.variant
    if a.variant=='R_CANONICAL':
        import copy
        expected=copy.deepcopy(expected);expected['grpo']['reward']['progress_weight']=5.
    if cfg!=expected:raise ValueError('Configuration differs from qualified recipe')
    output=root/'runs'/(a.variant+'_seed42_v1')
    python='/mnt/project/flow-safe-improvement-20261013/venv/bin/python'
    payload=dict(vars(a),snapshot=str(snapshot),source=source,python=python,config=config,manifest=str(manifest),
                 hard_manifest_sha256=file_sha(manifest),output=str(output),root=str(root))
    code='''import json,os,subprocess,time
from pathlib import Path
p=json.loads(PAYLOAD);out=Path(p['root'])/'formal_dispatch';out.mkdir(parents=True,exist_ok=True)
receipt=out/(p['variant']+'.json')
if receipt.exists():
 old=json.loads(receipt.read_text());proc=Path('/proc')/str(old['pid'])
 if proc.exists() and 'scripts.analysis.simwam_recipe.train' in (proc/'cmdline').read_bytes().decode(errors='replace'):
  print('ALREADY_RUNNING',old['pid']);raise SystemExit(0)
free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
if len(free.splitlines())!=8 or min(map(int,free.splitlines()))<12000:raise RuntimeError('Insufficient live GPU capacity; no eviction')
if (Path(p['output'])/'identity.json').exists() and not p['resume']:raise RuntimeError('Existing formal run requires explicit exact resume')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',NO_ALBUMENTATIONS_UPDATE='1',TOKENIZERS_PARALLELISM='false',PYTHONPATH=p['snapshot'])
cmd=[p['python'],'-m','torch.distributed.run','--standalone','--nproc_per_node=8','--module','scripts.analysis.simwam_recipe.train','--config',p['config'],'--manifest',p['manifest'],'--output',p['output']]
if p['resume']:cmd+=['--resume',p['resume']]
log=(out/(p['variant']+'.log')).open('ab',buffering=0)
child=subprocess.Popen(cmd,cwd=p['snapshot'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt.write_text(json.dumps(dict(p,pid=child.pid,started=time.time(),command=cmd,live_free_MiB=list(map(int,free.splitlines()))),indent=2)+'\\n')
print('STARTED',p['variant'],child.pid,p['output'])
'''.replace('PAYLOAD',repr(json.dumps(payload)))
    subprocess.run(['ssh','-o','BatchMode=yes',a.host,shlex.join([python,'-c',code])],check=True)


if __name__=='__main__':main()
