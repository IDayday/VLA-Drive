"""Bounded native eight-GPU equivalence against existing corrected runs."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from starVLA.rl.constrained_flow_noise.contracts import require_dispatch,file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json
from .qualify_collection_overlap import exact


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--adapter-module',default='scripts.analysis.constrained_flow_noise.train_efficient',
                   choices=['scripts.analysis.constrained_flow_noise.train_efficient','scripts.analysis.constrained_flow_noise.train_efficient_cast'])
    a=p.parse_args();host=require_dispatch();root=Path(a.run_root);out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    identity=importlib.import_module(a.adapter_module).source_identity()
    record=dict(status='RUNNING',host=host,execution_sha256=identity['sha256'],steps=[],started_unix=time.time())
    atomic_json(out/'status.json',record)
    env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             OPENBLAS_NUM_THREADS='1',NO_ALBUMENTATIONS_UPDATE='1',PYTHONUNBUFFERED='1')
    def execute(label,args):
        log=out/(label+'.log')
        with log.open('w') as stream:
            proc=subprocess.Popen(args,cwd=Path.cwd(),env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        entry=dict(label=label,args=args,pid=proc.pid,started_unix=time.time(),log=str(log))
        record['steps'].append(entry);atomic_json(out/'status.json',record)
        code=proc.wait();entry.update(returncode=code,finished_unix=time.time())
        atomic_json(out/'status.json',record)
        if code:raise RuntimeError('Bounded qualification failed: '+label)
    def native_args(letter,target):
        old=json.loads((root/f'qualify_replay_{letter}_v3.launch.json').read_text())['args']
        args=list(old)
        i=args.index('scripts.analysis.constrained_flow_noise.train')
        args[i:i+1]=[a.adapter_module,'--']
        args[args.index('--output')+1]=str(target)
        return args
    def compare(left,right,label):
        execute(label+'_state',[sys.executable,'-m','scripts.analysis.constrained_flow_noise.verify_resume',
            '--continuous',str(left),'--resumed',str(right),'--output',str(out/(label+'_state.json'))])
        files=[sorted((p/'behavior_collections').glob('*.pt')) for p in (left,right)]
        if [p.name for p in files[0]]!=[p.name for p in files[1]]:raise ValueError('Collection count changed')
        for x,y in zip(*files):
            for path in (x,y):
                if file_sha(path)!=json.loads(path.with_suffix('.json').read_text())['sha256']:raise ValueError('Changed chain archive')
            exact(torch.load(x,map_location='cpu',weights_only=False),torch.load(y,map_location='cpu',weights_only=False),'saved_collection')
        atomic_json(out/(label+'_chains.json'),dict(status='PASS',collections=len(files[0]),
                    all_chain_probabilities_requests_simulations_costs_advantages_selection_exact=True,
                    excluded='Only execution timings and explicit collection_execution provenance'))
    try:
        for letter in ('C','N'):
            target=out/f'{letter}_continuous'
            execute(letter+'_continuous',native_args(letter,target))
            compare(root/f'qualify_replay_{letter}_v3',target,letter+'_vs_original')
        target=out/'N_resumed'
        execute('N_interrupted',native_args('N',target)+['--stop-after-updates','1'])
        cp=json.loads((target/'latest_checkpoint.json').read_text())['path']
        execute('N_resumed',native_args('N',target)+['--resume',cp])
        compare(out/'N_continuous',target,'N_resume')
        record.update(status='PASS',both_variants_exact_learning=True,all_rank_optimizer_rng_exact=True,
                      saved_chains_feedback_exact=True,resume_exact=True,
                      actual_training_candidates=3*32*16,finished_unix=time.time(),
                      scope='Real full-head native training, two inner epochs, original replay, KL, dual, optimizer, ODE export; qualification only')
        atomic_json(out/'acceptance.json',record)
    except BaseException as error:
        record.update(status='FAILED',error=repr(error),finished_unix=time.time())
        raise
    finally:atomic_json(out/'status.json',record)


if __name__=='__main__':main()
