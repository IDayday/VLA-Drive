"""Finite real eight-rank gate for the new runtime, reusing existing comparisons."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from starVLA.rl.fm_transfer.source_process import ROOT, B_SFT, OWN_MANIFEST, MODES, load_process_recipe
from starVLA.rl.fm_transfer.contracts import require_dispatch
from starVLA.rl.simwam_recipe.contracts import file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json

PYTHON='/mnt/project/flow-safe-improvement-20261013/venv/bin/python'


def configuration(mode):
    name='simwam_upstream_data_long.yaml' if mode=='UPSTREAM_DATA' else 'simwam_own_data_long.yaml'
    return 'configs/flow_rl/'+name, ROOT/'source_manifest.json' if mode=='UPSTREAM_DATA' else OWN_MANIFEST


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--mode',choices=MODES,required=True);p.add_argument('--output',required=True)
    a=p.parse_args();host=require_dispatch();source=Path.cwd().resolve();root=Path(a.output)
    if not json.loads((source/'SOURCE_COMPLETE.json').read_text())['git_archive_extraction_complete']:
        raise ValueError('Unpublished execution source')
    cfg,manifest=configuration(a.mode);load_process_recipe(cfg)
    root.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',NO_ALBUMENTATIONS_UPDATE='1',
        TOKENIZERS_PARALLELISM='false',VLM_ATTN_IMPLEMENTATION='flash_attention_2',PYTHONPATH=str(source))
    record=dict(status='RUNNING',mode=a.mode,source=str(source),host=host,world=8,phases=[],start_unix=time.time(),
        manifest_sha256=file_sha(manifest),config_sha256=file_sha(cfg),SFT_files_sha256=file_sha(B_SFT/'files.json'))
    def run(name,command,gpu=True):
        require_dispatch();local_env=dict(env)
        if not gpu:local_env['CUDA_VISIBLE_DEVICES']=''
        free=list(map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()))
        if gpu and min(free)<36000:raise RuntimeError('Preserve existing jobs: qualification no longer has headroom')
        with (root/(name+'.log')).open('xb') as log:
            child=subprocess.Popen(command,cwd=source,env=local_env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            event=dict(phase=name,command=command,start_unix=time.time(),pid=child.pid,
                start_ticks=int((Path('/proc')/str(child.pid)/'stat').read_text().split()[21]))
            record['phases'].append(event);atomic_json(root/'invocation.json',record)
            event.update(exit_code=child.wait(),end_unix=time.time());atomic_json(root/'invocation.json',record)
            if event['exit_code']:raise RuntimeError('Failed source-process gate: '+name)
    try:
        for name,directory,stop,resume in [('continuous','continuous',4,None),('split2','resumed',2,None),
                ('resume4','resumed',4,root/'resumed/checkpoints/update000002_inner2')]:
            command=[PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','--module',
                'scripts.analysis.fm_transfer.train_source_process','--config',cfg,'--manifest',str(manifest),
                '--sft-checkpoint',str(B_SFT),'--output',str(root/directory),'--qualification','--stop-after-update',str(stop),'--save-every','2']
            if resume:command+=['--resume',str(resume)]
            run(name,command)
        record['status']='EXECUTION_COMPLETE_REQUIRES_COMPARISON';atomic_json(root/'invocation.json',record)
        run('compare',[PYTHON,'-m','scripts.analysis.fm_transfer.qualify_rl_resume','--root',str(root)],False)
        checkpoint=root/'continuous/checkpoints/update000004_inner0'
        run('merge',[PYTHON,'-m','scripts.analysis.fm_transfer.qualify_merged_ode','--checkpoint',str(checkpoint),'--output',str(root/'merged_ODE.json')])
        run('export',[PYTHON,'-m','scripts.analysis.fm_transfer.export_policy','--checkpoint',str(checkpoint),'--output',str(root/'export'),'--qualification'],False)
        info=json.loads((root/'export/export.json').read_text())
        run('raw_deployment',[PYTHON,'-m','scripts.analysis.fm_transfer.evaluate_policy','--export',str(root/'export'),
            '--export-sha256',info['weight_sha256'],'--output',str(root/'raw_deployment'),'--cohort','TRAIN_DIAGNOSTIC',
            '--shard','0','--shards','1','--max-scenes','2'])
        if json.loads((root/'raw_deployment/rank0.json').read_text())['status']!='QUALIFICATION_COMPLETE':
            raise ValueError('Original-observation deployment incomplete')
        record.update(status='PASS',end_unix=time.time(),qualification_only=True,formal_initialization_allowed=False,
            physical_training_candidates=128,raw_observation_ODE_candidates=4,
            evidence={rel:file_sha(root/rel) for rel in ('validation.json','merged_ODE.json','export/export.json','raw_deployment/rank0.json')})
        atomic_json(root/'COMPLETE.json',record)
    except BaseException:
        import traceback
        record.update(status='FAILED',end_unix=time.time(),error=traceback.format_exc());raise
    finally:atomic_json(root/'invocation.json',record)

if __name__=='__main__':main()
