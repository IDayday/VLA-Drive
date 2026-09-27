"""Continue the already-authorized fixed64 campaign after its primary pair finishes.

This process only sequences existing entry points. It never changes training source,
hyperparameters or data, and never retries/overwrites a failed or existing run.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from tools.joint_local_scene_v3.budget import atomic_json


def main():
    p=argparse.ArgumentParser()
    for key in ('campaign','source','train','holdout'):p.add_argument('--'+key,required=True)
    p.add_argument('--gpus',type=int,nargs=2,required=True);a=p.parse_args();root=Path(a.campaign)
    decision=json.loads((root/'FINAL_ENDPOINT_DECISION.json').read_text())
    if decision['primary_final_epochs']!=64 or decision['second_seed_final_epochs']!=64:raise ValueError('This sequencer requires the frozen shared64 decision')
    state_path=root/'CONTINUATION.json'
    with state_path.open('x') as f:json.dump({'pid':os.getpid(),'status':'starting'},f)
    state={'pid':os.getpid(),'status':'running','stage':'waiting_primary64','training_source':a.source,'started_utc_seconds':time.time(),'analysis_source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    def status(stage):
        state['stage']=stage;atomic_json(state_path,state);print(json.dumps(state),flush=True)
    def wait_primary():
        while True:
            ledger=json.loads((root/'budget_ledger.json').read_text());runs={r['id']:r for r in ledger['runs']}
            pair=[runs['formal42_'+m] for m in ('all','mask')]
            if all(r['status']=='complete' for r in pair):break
            if any(r['status'] not in ('running','complete') for r in pair):raise RuntimeError('Primary pair paused/failed; inspect before continuing')
            time.sleep(5)
        for mode in ('all','mask'):
            completed=json.loads((root/f'formal42_{mode}/status.json').read_text())
            if completed['step']!=14592 or completed['epochs']!=64:raise ValueError('Primary endpoint mismatch')
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    def command(module,*args):return [sys.executable,'-m','tools.joint_local_scene_v3.'+module,*map(str,args)]
    def diagnostics(seed):
        status(f'diagnostics_seed{seed}');jobs=[]
        for mode,gpu in zip(('all','mask'),a.gpus):
            name=f'diag{seed}_{mode}';argv=command('endpoint_diagnostics','--data',a.holdout,'--checkpoint',root/f'formal{seed}_{mode}/milestones/step_14592.pt','--output',root/name,'--ledger',root/'budget_ledger.json','--run-id',name,'--protocol',root/'relation_queries.json','--samples',8)
            with (root/'commands.jsonl').open('a') as f:f.write(json.dumps({'run_id':name,'cwd':str(Path.cwd()),'argv':argv,'CUDA_VISIBLE_DEVICES':str(gpu),'started_utc_seconds':time.time()})+'\n')
            log=(root/(name+'_process.log')).open('x');jobs.append((subprocess.Popen(argv,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT),log,name))
        codes=[proc.wait() for proc,_,_ in jobs]
        for _,log,_ in jobs:log.close()
        if any(codes):raise RuntimeError('Endpoint diagnostic failed; retained progress, no retry')
        for _,_,name in jobs:subprocess.run(command('analyze_diagnostics','--input',root/name,'--output',root/name/'analysis.json'),env=env,check=True)
        subprocess.run(command('verify_pair','--campaign',root,'--seed',seed,'--epochs',64,'--output',root/f'VERIFY_SEED{seed}_FINAL.json'),env=env,check=True)
    try:
        status('waiting_primary64');wait_primary();diagnostics(42)
        base=command('launch_pair','--campaign',root,'--source',a.source,'--train',a.train,'--holdout',a.holdout,'--seed',43,'--gpus',*a.gpus)
        status('seed43_initial32');subprocess.run(base,env=env,check=True)
        for mode in ('all','mask'):
            saved=json.loads((root/f'formal43_{mode}/status.json').read_text())
            if saved['status']!='paused' or saved['step']!=7296:raise RuntimeError('Second-seed planned boundary not reached')
        status('seed43_continue_to_fixed64');subprocess.run(base+['--resume-to64'],env=env,check=True)
        for mode in ('all','mask'):
            saved=json.loads((root/f'formal43_{mode}/status.json').read_text())
            if saved['status']!='complete' or saved['step']!=14592:raise RuntimeError('Second-seed final boundary not reached')
        diagnostics(43)
        status('final_analysis');subprocess.run(command('analyze_campaign','--campaign',root,'--output',root/'analysis_final'),env=env,check=True)
        state['status']='complete';status('ready_for_results_review')
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));atomic_json(state_path,state);raise


if __name__=='__main__':main()
