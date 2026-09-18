"""Single F full-data epoch; baseline/final full-navtest run on separate GPUs.

No model/seed/checkpoint selection from navtest. Training and scoring failures
are reported separately; a scoring I/O failure does not terminate valid training.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import threading
import time
from scripts.cluster_flow_grpo.cluster import run, write_json, exclusive_controller, ROOT, PYTHON, base_env


def orchestrate(spec, train, evaluate, *, poll_seconds=5):
    """Production control flow, injectable only at real subprocess boundaries."""
    root=Path(spec['control_dir']);root.mkdir(parents=True,exist_ok=True)
    state={'status':'RUNNING','started':time.time(),'training':None,'evaluations':{}}
    cancel=threading.Event()
    with ThreadPoolExecutor(max_workers=2) as pool:
        training=pool.submit(train,cancel)
        baseline=pool.submit(evaluate,'sft',spec['baseline_checkpoint'],cancel)
        # CPU polling only of OUR active training. No GPU availability waiter.
        while not (training.done() and baseline.done()):
            state['training_status']='FINISHED' if training.done() else 'RUNNING'
            state['baseline_status']='FINISHED' if baseline.done() else 'RUNNING'
            if baseline.done():
                try:state['evaluations']['sft']=baseline.result()
                except Exception as exc:state['baseline_error']=repr(exc)
            write_json(root/'progress.json',state)
            try: training.result(timeout=poll_seconds)
            except TimeoutError: pass
            except Exception: break
            if training.done() and not baseline.done():
                try: baseline.result(timeout=poll_seconds)
                except TimeoutError: pass
                except Exception: pass
        failures={}
        try:
            state['training']=training.result()
        except Exception as exc:
            failures['training']=repr(exc)
        try:
            state['evaluations']['sft']=baseline.result()
        except Exception as exc:
            failures['sft_evaluation']=repr(exc)
        if state['training'] and state['training'].get('status')=='PASS':
            try:state['evaluations']['last']=evaluate('last',state['training']['exported'],cancel)
            except Exception as exc:failures['last_evaluation']=repr(exc)
        else:failures.setdefault('training','no completed optimizer budget')
        state.update(status='FAIL' if failures else 'COMPLETE',failures=failures,finished=time.time())
        write_json(root/'result.json',state)
        write_json(root/'progress.json',state)
        if failures:raise RuntimeError(str(failures))
        return state


def train_epoch(spec,cancel):
    from starVLA.rl.flow_grpo.config import resolve_config
    from starVLA.rl.flow_grpo.checkpoint import validate_checkpoint,export_checkpoint
    from starVLA.rl.flow_grpo.orchestration import checkpoint_inventory
    from starVLA.rl.flow_grpo.transactions import preserve_attempt
    from starVLA.rl.flow_grpo.reproducibility import training_provenance
    from starVLA.rl.flow_grpo.acceptance import acceptance_context,enforce_training_budget
    import os
    os.environ['WORLD_SIZE']='8'
    cfg,sft=resolve_config(spec['config'])
    context=json.loads(Path(cfg['runtime']['epoch_evidence']).read_text())['context']
    current=acceptance_context(cfg,context['resume_identity'])
    enforce_training_budget(cfg,current)
    provenance=training_provenance(cfg,sft,current['resume_identity'])
    output=Path(cfg['runtime']['output_dir'])
    validate=lambda p:validate_checkpoint(p,cfg,provenance,8)
    complete,incomplete=checkpoint_inventory(output,validate)
    target=cfg['runtime']['max_updates']
    if complete and max(complete)>target:raise ValueError('checkpoint beyond registered epoch budget')
    for path in incomplete:preserve_attempt(path)
    if target not in complete:
        resume=complete[max(complete)] if complete else Path(spec['initial_resume'])
        validate(resume)
        job=json.loads(Path(spec['train_spec']).read_text())
        job['entry'] += ['--resume',str(resume)]
        attempt=Path(job['control_dir']).with_name(Path(job['control_dir']).name+f'_attempt_{time.time_ns()}')
        job['control_dir']=str(attempt)
        path=Path(spec['control_dir'])/f'train_attempt_{time.time_ns()}.json'
        write_json(path,job)
        result=run(path,cancel_event=cancel)
        if result['status']!='PASS':raise RuntimeError('native epoch process failed')
    boundary=output/f'checkpoints/update_{target:06d}'
    validate(boundary)
    exported=export_checkpoint(boundary,output/f'export_update{target}')
    return {'status':'PASS','update':target,'checkpoint':str(boundary),'exported':str(exported)}


def evaluate_policy(spec,label,checkpoint,cancel):
    from scripts.cluster_flow_grpo.parallel_evaluation import evaluate_parallel
    from starVLA.rl.flow_grpo.loading import file_sha
    common=spec['evaluation'];results={}
    for seed in [42,43,44,45,46]:
        dest=Path(common['root'])/f'{label}_navtest_seed{seed}'
        report=evaluate_parallel(spec['config'],checkpoint,str(dest),'navtest',common['tokens'],
                  common['data_root'],common['metric_cache'],seed,common['slots'],cancel_event=cancel)
        if report['scene_count']!=12146 or report.get('status')!='COMPLETE':
            raise ValueError('complete official navtest is mandatory')
        v1=dest.with_name(dest.name+'_pdms_v1')
        v1spec={'split':'navtest','seed':seed,'raw_logs':common['raw_logs'],'maps':common['maps'],
                'cache_root':common['v1_cache'],'scene_csv':str(dest/'original_protocol_scores.csv'),
                'evaluations':{label:str(dest)}}
        v1spec_path=dest/'v1_spec.json';write_json(v1spec_path,v1spec)
        if (v1/'COMPLETE').is_file():
            proof=json.loads((v1/'identity.json').read_text())
            if proof['spec']!=v1spec or proof['inputs'][label]['trajectory_sha256']!=file_sha(dest/'trajectories.npz'):
                raise ValueError('v1 result identity differs')
            seal=json.loads((v1/'COMPLETE').read_text())
            if seal['summary_sha256']!=file_sha(v1/'summary.json') or seal['identity_sha256']!=file_sha(v1/'identity.json'):
                raise ValueError('v1 result changed')
        else:
            from starVLA.rl.flow_grpo.transactions import preserve_attempt
            if v1.exists():preserve_attempt(v1)
            command=[PYTHON,'-m','scripts.analysis.paired_dev_pdms_v1','--spec',str(v1spec_path),
                     '--workers','8','--output',str(v1)]
            # CPU scoring on the evaluation host, using only the project's own
            # trusted raw scenes/caches. No GPU claim or additional inference.
            cpu_job={'job_id':str(v1),'nodes':[{'host':common['slots'][0]['host'],'devices':[],
                       'cpu_affinity':list(range(64,96))}], 'direct_command':command,
                     'control_dir':str(v1)+f'_control_{time.time_ns()}',
                     'timeout_seconds':21600,'environment':{'CUDA_VISIBLE_DEVICES':''}}
            job_path=dest/f'v1_job_{time.time_ns()}.json';write_json(job_path,cpu_job)
            run(job_path,cancel_event=cancel)
        summary=json.loads((v1/'summary.json').read_text())
        # The v1 CSV itself is content sealed in its summary.
        if summary['results'][label]['csv_sha256']!=file_sha(v1/(label+'.csv')):
            raise ValueError('v1 score CSV changed')
        results[str(seed)]={'v2':str(dest),'v1':str(v1),'epdms':report['epdms'],
                           'pdms':summary['results'][label]['means']['score']}
    return {'status':'COMPLETE','seeds':results,'scenes_per_seed':12146}


def summarize(spec,result):
    import numpy as np
    import pandas as pd
    from starVLA.rl.flow_grpo.evaluation import paired_scores
    root=Path(spec['control_dir']);paired={};all_frames={}
    for metric,column in [('v1','score'),('v2','score')]:
        seeds={};frames=[]
        for seed in [42,43,44,45,46]:
            inputs=[]
            for label in ['sft','last']:
                folder=Path(result['evaluations'][label]['seeds'][str(seed)][metric])
                path=folder/(label+'.csv' if metric=='v1' else 'original_protocol_scores.csv')
                inputs.append(pd.read_csv(path,dtype={'token':str,'log_name':str}))
            rows,summary=paired_scores(*inputs)
            rows.to_csv(root/f'paired_{metric}_seed{seed}.csv',index=False)
            seeds[str(seed)]=summary;frames.append(inputs)
        # Average the five fixed-noise evaluations per scene before log bootstrap;
        # never treat five seeds times adjacent frames as independent observations.
        averaged=[pd.concat([f[i] for f in frames]).groupby(['token','log_name'],as_index=False)['score'].mean()
                  for i in range(2)]
        rows,summary=paired_scores(*averaged)
        rows.to_csv(root/f'paired_{metric}_five_seed_mean.csv',index=False)
        means=[[float(f[i]['score'].mean()) for f in frames] for i in range(2)]
        paired[metric]={'per_seed':seeds,'five_seed_mean_paired':summary,
                       'sft_mean':float(np.mean(means[0])),'sft_seed_std':float(np.std(means[0])),
                       'rl_mean':float(np.mean(means[1])),'rl_seed_std':float(np.std(means[1]))}
    write_json(root/'paired_performance.json',{'status':'COMPLETE','protocols':{'v1':'NAVSIM v1.1 PDMS','v2':'NAVSIM v2 official one-stage EPDMS'},'results':paired})


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--spec',required=True)
    args=parser.parse_args();spec=json.loads(Path(args.spec).read_text())
    # Bind this independently executable controller in the launch manifest.
    from starVLA.rl.flow_grpo.loading import file_sha
    if spec['controller_sha256']!=file_sha(__file__):raise ValueError('controller source changed')
    with exclusive_controller(spec['control_dir']):
        result=orchestrate(spec,lambda c:train_epoch(spec,c),lambda l,p,c:evaluate_policy(spec,l,p,c))
        summarize(spec,result)
