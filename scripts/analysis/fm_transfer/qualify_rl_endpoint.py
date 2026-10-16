"""Actual eight-GPU source-RL gate on an own complete supervised endpoint.

Reuse the existing trainer, checkpoint and comparison code: continuous four
updates versus inner2 save/resume4, then merged full ODE and original-observation
deployment. These weights are qualification-only, never formal initialization.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from scripts.analysis.fm_transfer.dispatch_screen import completed_sft
from starVLA.rl.fm_transfer.contracts import require_dispatch
from starVLA.rl.simwam_recipe.contracts import file_sha,POPULATION
from starVLA.rl.flow_grpo.transactions import atomic_json

PYTHON='/mnt/project/flow-safe-improvement-20261013/venv/bin/python'


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--sft-checkpoint',required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    source=Path.cwd().resolve();host=require_dispatch()
    if not json.loads((source/'SOURCE_COMPLETE.json').read_text())['git_archive_extraction_complete']:
        raise ValueError('Qualification requires an immutable source')
    sft,files=completed_sft(a.sft_checkpoint)
    hard=json.loads(Path(a.manifest).read_text());pop=json.loads(Path(POPULATION).read_text())
    expected=dict(SFT_model_sha256=files['model.pt'],SFT_identity_sha256=sft['sha256'],
        architecture=sft['config']['architecture'],seed=2101,grid='native',threshold=.9,
        statistic='one canonical v1.1 score per scene')
    if (hard['status']!='COMPLETE_FIXED_SFT_SINGLE_NATIVE_ODE_HARD_SUBSET' or
            hard['screened_scene_count']!=len(pop['train_tokens']) or hard['screen_contract']!=expected):
        raise ValueError('Own full TRAIN hard screen must be fixed before endpoint RL qualification')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',NO_ALBUMENTATIONS_UPDATE='1',TOKENIZERS_PARALLELISM='false',
        VLM_ATTN_IMPLEMENTATION='flash_attention_2',PYTHONPATH=str(source))
    record=dict(status='RUNNING',host=host,source=str(source),start_unix=time.time(),phases=[],world=8,
        SFT_identity=sft['sha256'],SFT_model_sha256=files['model.pt'],manifest_sha256=file_sha(a.manifest),qualification=True)
    def run(name,cmd,*,gpu=True):
        require_dispatch()
        free=list(map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()))
        if gpu and (len(free)!=8 or min(free)<36000):raise RuntimeError('No qualified eight-GPU headroom; preserve existing jobs')
        local_env=dict(env)
        if not gpu:local_env['CUDA_VISIBLE_DEVICES']=''
        with (root/(name+'.log')).open('xb') as log:
            child=subprocess.Popen(cmd,cwd=source,env=local_env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            event=dict(phase=name,pid=child.pid,start_ticks=int((Path('/proc')/str(child.pid)/'stat').read_text().split()[21]),
                start_unix=time.time(),command=cmd,resources_free_MiB=free)
            record['phases'].append(event);atomic_json(root/'invocation.json',record)
            event['exit_code']=child.wait();event['end_unix']=time.time();atomic_json(root/'invocation.json',record)
            if event['exit_code']:raise RuntimeError('Qualification phase failed: '+name)
    try:
        for name,directory,stop,resume in [('continuous','continuous',4,None),('split2','resumed',2,None),
                ('resume4','resumed',4,root/'resumed/checkpoints/update000002_inner2')]:
            cmd=[PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','--module','scripts.analysis.fm_transfer.train_rl',
                '--config','configs/flow_rl/simwam_source_ddp.yaml','--sft-checkpoint',a.sft_checkpoint,'--manifest',a.manifest,
                '--output',str(root/directory),'--qualification','--stop-after-update',str(stop),'--save-every','2']
            if resume:cmd+=['--resume',str(resume)]
            run(name,cmd)
        record['status']='EXECUTION_COMPLETE_REQUIRES_COMPARISON';atomic_json(root/'invocation.json',record)
        run('compare',[PYTHON,'-m','scripts.analysis.fm_transfer.qualify_rl_resume','--root',str(root)],gpu=False)
        checkpoint=root/'continuous/checkpoints/update000004_inner0'
        run('merge_ODE',[PYTHON,'-m','scripts.analysis.fm_transfer.qualify_merged_ode','--checkpoint',str(checkpoint),'--output',str(root/'merged_ODE.json')])
        run('export',[PYTHON,'-m','scripts.analysis.fm_transfer.export_policy','--checkpoint',str(checkpoint),'--output',str(root/'export'),'--qualification'],gpu=False)
        export=json.loads((root/'export/export.json').read_text())
        run('raw_deployment',[PYTHON,'-m','scripts.analysis.fm_transfer.evaluate_policy','--export',str(root/'export'),
            '--export-sha256',export['weight_sha256'],'--output',str(root/'raw_deployment'),'--cohort','TRAIN_DIAGNOSTIC',
            '--shard','0','--shards','1','--max-scenes','2'])
        if json.loads((root/'raw_deployment/rank0.json').read_text())['status']!='QUALIFICATION_COMPLETE':
            raise ValueError('Original deployment gate did not complete')
        record.update(status='PASS',end_unix=time.time(),qualification_only=True,formal_initialization_allowed=False,
            real_new_training_candidates=128,raw_deployment_candidates=4,
            evidence={p:file_sha(root/p) for p in ('validation.json','merged_ODE.json','export/export.json','raw_deployment/rank0.json')})
        atomic_json(root/'COMPLETE.json',record)
    except BaseException:
        import traceback
        record.update(status='FAILED',end_unix=time.time(),error=traceback.format_exc());raise
    finally:atomic_json(root/'invocation.json',record)


if __name__=='__main__':main()
