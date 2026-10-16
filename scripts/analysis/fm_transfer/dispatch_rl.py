"""Admit one own-SFT R_SOURCE run after the actual endpoint gates.

Loads the complete supervised endpoint, never the qualification adapters. Native
same-source resume is explicit; cross-host ownership needs a separate verified
migration rather than a guessed dead PID on a different machine.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
from scripts.analysis.fm_transfer.dispatch_screen import completed_sft
from starVLA.rl.fm_transfer.contracts import ROOT,require_dispatch
from starVLA.rl.simwam_recipe.contracts import POPULATION,file_sha,fingerprint,load_recipe
from starVLA.rl.flow_grpo.transactions import atomic_json


def qualify(source,sft_checkpoint,manifest,gate_root,baseline_root):
    source=Path(source);parent,files=completed_sft(sft_checkpoint)
    hard=json.loads(Path(manifest).read_text());body=dict(hard);sha=body.pop('sha256')
    pop=json.loads(Path(POPULATION).read_text())
    expected=dict(SFT_model_sha256=files['model.pt'],SFT_identity_sha256=parent['sha256'],
        architecture=parent['config']['architecture'],seed=2101,grid='native',threshold=.9,
        statistic='one canonical v1.1 score per scene')
    if (fingerprint(body)!=sha or hard['status']!='COMPLETE_FIXED_SFT_SINGLE_NATIVE_ODE_HARD_SUBSET' or
            hard['screen_contract']!=expected or hard['screened_scene_count']!=len(pop['train_tokens']) or
            hard['population_identity']!=pop['identity_sha256']):
        raise ValueError('Incomplete/mismatched own fixed hard population')
    gate_root=Path(gate_root);gate=json.loads((gate_root/'COMPLETE.json').read_text())
    if (gate['status']!='PASS' or gate['world']!=8 or gate['source']!=str(source) or
            gate['SFT_identity']!=parent['sha256'] or gate['SFT_model_sha256']!=files['model.pt'] or
            gate['manifest_sha256']!=file_sha(manifest)):
        raise ValueError('Actual eight-GPU endpoint qualification is missing or mismatched')
    for relative,sha in gate['evidence'].items():
        if file_sha(gate_root/relative)!=sha:raise ValueError('Qualification evidence changed: '+relative)
    validation=json.loads((gate_root/'validation.json').read_text())
    if (validation['status']!='PASS' or validation['world']!=8 or
            not validation['full_adapter_Adam_scheduler_RNG_equal'] or
            not validation['actual_chains_probabilities_advantages_scores_equal']):
        raise ValueError('Incomplete eight-rank numerical/recovery evidence')
    qi=json.loads((gate_root/'continuous/identity.json').read_text())
    for relative,sha in qi['source_files'].items():
        if file_sha(source/relative)!=sha:raise ValueError('Formal code differs from the actual qualified code: '+relative)
    cfg=load_recipe(source/'configs/flow_rl/simwam_source_ddp.yaml')
    trained=dict(qi['recipe']);trained.pop('supervised_transfer')
    if cfg!=trained:raise ValueError('Formal recipe differs from qualification')
    baselines={}
    for cohort in ('DEV','TRAIN_DIAGNOSTIC'):
        name=f"{parent['config']['variant']}_SFT_update{parent['config']['max_updates']:06d}_{cohort}"
        path=Path(baseline_root)/name;meta=json.loads((path/'manifest.json').read_text())
        if (meta['new_SFT']['model_sha256']!=files['model.pt'] or meta['seed']!=2101 or
                meta['qualification_only'] or meta['modes']!=['E_NATIVE','E_SHIFT5'] or meta['stage']!='SFT'):
            raise ValueError('Own complete SFT baseline is required')
        for rank in range(meta['shards']):
            status=json.loads((path/f'rank{rank}.json').read_text())
            if status['status']!='ASSIGNED_COMPLETE' or status['identity_sha256']!=meta['sha256']:
                raise ValueError('Incomplete new SFT baseline')
        # The full analyzer has verified every stored payload and denominator.
        analysis=Path(baseline_root).parent/'analysis'/(name+'.json')
        result=json.loads(analysis.read_text())
        if result['status']!='COMPLETE_POPULATION' or result['identity']!=meta:
            raise ValueError('Baseline payload analysis incomplete')
        baselines[cohort]=dict(path=str(path),manifest_sha256=file_sha(path/'manifest.json'),analysis_sha256=file_sha(analysis))
    return parent,dict(SFT_model_sha256=files['model.pt'],qualification_sha256=file_sha(gate_root/'COMPLETE.json'),
        manifest_sha256=file_sha(manifest),baselines=baselines,recipe=cfg)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--sft-checkpoint',required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--qualification-root',required=True)
    p.add_argument('--baseline-root',default=str(ROOT/'evaluations'));p.add_argument('--output',required=True)
    p.add_argument('--resume');a=p.parse_args();host=require_dispatch();source=Path.cwd().resolve()
    snapshot=json.loads((source/'SOURCE_COMPLETE.json').read_text())
    if not snapshot['git_archive_extraction_complete']:raise ValueError('Unpublished formal source')
    parent,gates=qualify(source,a.sft_checkpoint,a.manifest,a.qualification_root,a.baseline_root)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'dispatch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=list(out.glob('dispatches/*.json'))
    if previous and not a.resume:raise ValueError('Existing dispatch; no implicit duplicate/restart')
    for path in previous:
        receipt=json.loads(path.read_text())
        if receipt['host']!=host:raise ValueError('Different host: verify migration before admitting a new owner')
        stat=Path('/proc')/str(receipt['pid'])/'stat'
        if stat.exists() and int(stat.read_text().split()[21])==receipt['start_ticks']:
            raise ValueError('The existing trainer is still running')
    if a.resume:
        checkpoint=Path(a.resume).resolve()
        if checkpoint.parent!=out.resolve()/'checkpoints' or '.incomplete' in checkpoint.name:
            raise ValueError('Same native run checkpoint required')
        seal=json.loads((checkpoint/'COMPLETE').read_text())
        if seal['schema']!=2 or seal['world']!=8 or seal['files_sha256']!=file_sha(checkpoint/'files.json'):
            raise ValueError('Invalid native recovery boundary')
    elif (out/'identity.json').exists():raise ValueError('Existing run requires explicit native resume')
    free=list(map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()))
    if len(free)!=8 or min(free)<36000:raise RuntimeError('No current eight-GPU headroom; do not evict real jobs')
    owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',NO_ALBUMENTATIONS_UPDATE='1',TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(source))
    cmd=['/mnt/project/flow-safe-improvement-20261013/venv/bin/python','-m','torch.distributed.run','--standalone','--nproc_per_node=8',
        '--module','scripts.analysis.fm_transfer.train_rl','--config','configs/flow_rl/simwam_source_ddp.yaml',
        '--sft-checkpoint',a.sft_checkpoint,'--manifest',a.manifest,'--output',str(out)]
    if a.resume:cmd+=['--resume',a.resume]
    dest=out/'dispatches';dest.mkdir(exist_ok=True);attempt=uuid.uuid4().hex;logfile=dest/(attempt+'.log')
    require_dispatch()
    with logfile.open('xb') as log:
        child=subprocess.Popen(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    receipt=dict(schema='fm_transfer_own_SFT_formal_RL_dispatch_v1',host=host,pid=child.pid,
        start_ticks=int((Path('/proc')/str(child.pid)/'stat').read_text().split()[21]),
        start_unix=time.time(),attempt=attempt,source=str(source),source_commit=snapshot['commit'],
        variant=parent['config']['variant'],own_SFT=parent['sha256'],gates=gates,
        command=cmd,resume=a.resume,log=str(logfile),live_free_MiB=free,live_GPU_owners=owners,
        dispatcher_sha256=file_sha(__file__),qualification_weights_used=False)
    atomic_json(dest/(attempt+'.json'),receipt);print(json.dumps(receipt))


if __name__=='__main__':main()
