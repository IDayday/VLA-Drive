"""Frozen-checkpoint Euler-step sensitivity; no training or scoring formula changes.

Each scene is encoded once. Every step count uses a fresh clone of the same
token-hashed noise and the original action head's Euler implementation. Scores
use tools.foresight.score_pdms and the canonical unmodified NAVSIM v1 caches.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time

from .navtest_schedule import atomic, lease, read, sha, source_identity
from .navtest_milestones import environment, release_owned_pressure, restore_owned_pressure

MODULE = 'tools.full_foresight.fm_step_sweep'
STEPS = [1, 2, 3, 5, 8, 10, 15, 20, 30]


def signature(value):
    # The GPU/official scorer contract uses compact JSON, whereas the older
    # observer's internal registration signature uses whitespace separators.
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_steps(steps):
    if (not steps or 10 not in steps or len(set(steps)) != len(steps) or
            any(type(n) is not int or not 1 <= n <= 30 for n in steps)):
        raise ValueError('Unique integer steps in [1,30], including reference 10, required')


def sample_encoded(model, encoded, noise, steps):
    """Temporarily change only the integrator count, never config or weights."""
    if type(steps) is not int or not 1 <= steps <= 30:
        raise ValueError('Invalid inference steps')
    from starVLA.dataloader.foresight_dataset import decode_ego
    previous = model.action_model.num_inference_timesteps
    try:
        model.action_model.num_inference_timesteps = steps
        with model.amp():
            actions = model.action_model.predict_action(encoded['action_queries'], initial_noise=noise.clone())
        return decode_ego(actions.float())
    finally:
        model.action_model.num_inference_timesteps = previous


def load_registration(path):
    registration = read(path)
    if registration.get('schema') != 'ddp_fm_step_sweep_v1' or signature({k:v for k,v in registration.items() if k!='identity'}) != registration['identity']:
        raise ValueError('Changed sweep registration')
    validate_steps(registration['config']['steps'])
    if source_identity(registration['config']['source_worktree']) != registration['source_sha']:
        raise ValueError('Sweep source changed')
    return registration, registration['config']


def register(config_path, output):
    from tools.foresight.checkpoints import checkpoint_identity
    from tools.foresight.lock_navtest import validate_lock
    config = read(config_path); validate_steps(config['steps'])
    source = source_identity(config['source_worktree'])
    training, checkpoint = checkpoint_identity(config['training_run'], config['checkpoint_tag'])
    baseline = read(Path(config['baseline_scores'])/'summary.json')
    old = baseline['export_identity']
    current = read(Path(config['current_root'])/'identity.json')
    index = read(Path(config['current_root'])/'index.json')
    metric = read(config['metric_index'])
    if (not baseline['valid'] or baseline['failed'] or baseline['scenes']!=12146 or baseline['logs']!=136 or
            old['checkpoint']!=checkpoint or old['protocol']['steps']!=10 or
            old['protocol']['precision']!='FP32' or old['protocol']['sampling_seed']!=config['sampling_seed'] or
            old['current_identity']!=current or len(index)!=12146 or len({r['token'] for r in index})!=12146 or
            {(r['token'],r['log']) for r in metric}!={(r['token'],r['log']) for r in index}):
        raise ValueError('Need the identical fixed-checkpoint complete original 10-step reference')
    if (len(config['hosts'])!=2 or config['hosts'][0]['host'] is not None or
            config['hosts'][1]['host']!='training-vla-zt2' or
            any(h['gpus']!=list(range(8)) for h in config['hosts']) or
            config['gpu_hour_cap']<=0 or config['max_seconds']<=0 or config['cpu_workers']*config['cpu_parallel']>64):
        raise ValueError('Invalid authorized bounded resource plan')
    for host in config['hosts']:
        if read(Path(host['training_run'])/'status.json')['status']!='COMPLETE':
            raise ValueError('Use only the two finished training allocations')
    root = Path(config['artifact_root']); root.mkdir(parents=True, exist_ok=False)
    value = {'schema':'ddp_fm_step_sweep_v1','source_sha':source,'config':config,
             'checkpoint':checkpoint,'training_identity':training['identity'],
             'baseline_summary_sha256':sha(Path(config['baseline_scores'])/'summary.json'),
             'baseline_csv_sha256':sha(Path(config['baseline_scores'])/'scenes.csv'),
             'official_metric_index_sha256':sha(config['metric_index']),
             'current_identity':current,'created_unix':time.time(),
             'purpose':'User-requested fixed-weight inference sensitivity; no automatic protocol promotion',
             'real_optimizer_updates':0}
    value['identity']=signature(value)
    for n in config['steps']:
        job=root/f'steps_{n:02d}';job.mkdir()
        lock = {'schema':'foresight_navtest_checkpoint_probe_lock_v1',
                'evaluation_purpose':'user_requested_fixed_checkpoint','requested_update':checkpoint['completed'],
                'authorization':'User explicitly requested representative Euler step counts from 1 to 30 without weight changes',
                'evaluation_source_sha':source,'checkpoint_selection':'One fixed C1 100k checkpoint for all step counts',
                'checkpoints':[checkpoint['sha256']],'checkpoint_records':{checkpoint['sha256']:checkpoint},
                'current_data_identity':current['identity'],'current_index_identity':current['index_sha256'],
                'scene_count':12146,'log_count':136,'official_metric_index_sha256':sha(config['metric_index']),
                'sampling_seeds':[config['sampling_seed']],'inference_steps':n,'candidates_per_scene':1,
                'learned_scorer':None,'precision':'FP32','final_endpoint_comparison':False,
                'sensitivity_registration':value['identity']}
        lock['identity']=signature(lock)
        validate_lock(lock, checkpoint, source, current, len(index), config['sampling_seed'], n)
        atomic(job/'lock.json',lock)
        protocol = dict(old['protocol'], steps=n, evaluation_purpose='user_requested_fixed_checkpoint')
        bank=job/'predictions';(bank/'predictions').mkdir(parents=True)
        atomic(bank/'identity.json',{'checkpoint':checkpoint,'current_identity':current,
            'evaluation_source':source,'protocol':protocol,'world_size':16,'limit':0})
    atomic(output,value)


def sweep_usage(registration, config):
    total=0.
    for path in (Path(config['campaign_root'])/'runs').glob('fm_sweep_'+registration['identity'][:12]+'*/status.json'):
        row=read(path);end=time.time() if row['status']=='RUNNING' else row.get('end_unix',row['start_unix'])
        total+=(end-row['start_unix'])*row['gpu_count']/3600
    return total


def export(registration_path, rank, attempt, smoke_count=0):
    import numpy as np
    import torch
    from starVLA.dataloader.foresight_dataset import ForesightCurrentDataset
    from tools.ddpolicy_vehicle.run_meter import metered_run
    from tools.foresight.checkpoints import checkpoint_identity, load_student, scene_noise
    from tools.ddpolicy_vehicle.campaign import charged_gpu_hours
    registration,config=load_registration(registration_path)
    if not 0<=rank<16 or smoke_count not in (0,4):raise ValueError('Invalid shard/smoke')
    root=Path(config['artifact_root']);run_id=f'fm_sweep_{registration["identity"][:12]}_r{rank}_a{attempt}'
    if smoke_count:run_id+='_smoke'
    with metered_run(config['campaign_root'],run_id,1,{'kind':'fixed_weight_FM_step_sweep','registration':registration['identity']}) as (meter,_,save):
        training,checkpoint=checkpoint_identity(config['training_run'],config['checkpoint_tag'])
        if checkpoint!=registration['checkpoint']:raise ValueError('Checkpoint changed before export')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.cuda.set_per_process_memory_fraction(.8)
        model=load_student(config['training_run'],config['checkpoint_tag'],training)
        if any(p.dtype!=torch.float32 or p.requires_grad and p.grad is not None for p in model.parameters()):
            raise ValueError('Requires FP32, no accumulated training gradients')
        # Disable gradients explicitly; no optimizer exists in this program.
        model.requires_grad_(False)
        parameter_versions={k:p._version for k,p in model.named_parameters()}
        atomic(root/f'precision_rank_{rank}.json',model.deployment_precision)
        data=ForesightCurrentDataset(config['current_root'])
        if data.identity!=registration['current_identity']:raise ValueError('Current input identity changed')
        ids=list(range(len(data)))[rank::16]
        if smoke_count:ids=list(range(smoke_count))
        complete={n:0 for n in config['steps']};failed={n:0 for n in config['steps']};parity=[]
        stopped=False
        for ordinal,index in enumerate(ids):
            if ((root/'STOP_REQUESTED').exists() or time.time()-meter['start_unix']>=config['max_seconds'] or
                    sweep_usage(registration,config)>=config['gpu_hour_cap'] or
                    charged_gpu_hours(Path(config['campaign_root']))>=config['campaign_gpu_hours']):
                stopped=True;meter['status']='PAUSED';break
            scene=data.index[index];token=scene['token'];pending=[]
            for n in config['steps']:
                bank=root/f'steps_{n:02d}'/'predictions';meta=bank/'predictions'/(token+'.json')
                if not smoke_count and meta.exists():
                    old=read(meta)
                    if old['identity_sha256']!=signature(read(bank/'identity.json')):raise ValueError('Scene identity changed')
                    if old['status']=='ok' and sha(meta.with_suffix('.npz'))!=old['proposal_sha256']:raise ValueError('Trajectory changed')
                    complete[n]+=1;failed[n]+=old['status']!='ok'
                else:pending.append(n)
            if not pending:continue
            try:
                observation=data[index];noise=scene_noise(token,config['sampling_seed'],'cuda')
                before=noise.clone();torch.cuda.synchronize();start=time.perf_counter()
                with torch.inference_mode():encoded=model.encode_current([observation])
                torch.cuda.synchronize();encode_seconds=time.perf_counter()-start
                for n in pending:
                    bank=root/f'steps_{n:02d}'/'predictions';dest=bank/'predictions'/(token+'.npz')
                    row={'token':token,'log':scene['log'],'identity_sha256':signature(read(bank/'identity.json')),'status':'ok',
                         'steps':n,'encode_current_seconds':encode_seconds}
                    try:
                        torch.cuda.synchronize();start=time.perf_counter()
                        with torch.inference_mode():trajectory=sample_encoded(model,encoded,noise,n)[0]
                        torch.cuda.synchronize();row['solver_seconds']=time.perf_counter()-start
                        row['inference_seconds']=encode_seconds+row['solver_seconds']
                        if trajectory.shape!=(8,3) or not torch.isfinite(trajectory).all():raise FloatingPointError('Invalid trajectory')
                        if not torch.equal(noise,before):raise AssertionError('Noise mutated across step counts')
                        if smoke_count or ordinal<4:
                            previous=model.action_model.num_inference_timesteps
                            try:
                                model.action_model.num_inference_timesteps=n
                                torch.cuda.synchronize();start=time.perf_counter()
                                with torch.inference_mode():standard=model.predict_action([observation],initial_noise=noise.clone())[0]
                                torch.cuda.synchronize();row['standalone_inference_seconds']=time.perf_counter()-start
                            finally:model.action_model.num_inference_timesteps=previous
                            if not torch.equal(trajectory,standard):raise AssertionError('Condition reuse differs from standard predict_action')
                        if smoke_count and n==10:
                            baseline=Path(config['baseline_predictions'])/'predictions'/(token+'.npz')
                            prior=np.load(baseline)['trajectory']
                            difference=float(np.max(np.abs(prior-trajectory.cpu().numpy())))
                            if difference!=0:raise AssertionError(f'10-step original trajectory differs: {difference}')
                            parity.append({'token':token,'maximum_absolute_difference':difference})
                        if not smoke_count:
                            temp=dest.with_suffix(f'.{os.getpid()}.tmp')
                            with temp.open('wb') as stream:np.savez_compressed(stream,trajectory=trajectory.cpu().numpy())
                            temp.replace(dest);row['proposal_sha256']=sha(dest)
                    except Exception as error:
                        row.update(status='failed',error=repr(error));failed[n]+=1
                        if smoke_count:raise
                    if not smoke_count:atomic(dest.with_suffix('.json'),row)
                    complete[n]+=1
            except Exception as error:
                if smoke_count:raise
                for n in pending:
                    meta=root/f'steps_{n:02d}'/'predictions'/'predictions'/(token+'.json')
                    if not meta.exists():
                        bank=meta.parent.parent
                        atomic(meta,{'token':token,'log':scene['log'],'identity_sha256':signature(read(bank/'identity.json')),
                                     'status':'failed','error':repr(error),'steps':n})
                        complete[n]+=1;failed[n]+=1
            meter.update(inference_scenes=ordinal+1,trajectories=sum(complete.values()),failed=sum(failed.values()),
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            save()
        if {k:p._version for k,p in model.named_parameters()}!=parameter_versions:
            raise AssertionError('Model parameters modified by inference')
        if model.action_model.num_inference_timesteps!=10:raise AssertionError('Original integrator default not restored')
        if smoke_count:
            if stopped or len(parity)!=4:raise RuntimeError('Incomplete four-scene parity smoke')
            atomic(root/'SMOKE.json',{'status':'COMPLETE','scenes':4,'original10_parity':parity,
                  'shared_condition_vs_standard':'bitwise identical for every tested N','weight_updates':0,
                  'steps':config['steps'],'checkpoint':checkpoint['sha256'],'precision':model.deployment_precision})
        else:
            for n in config['steps']:
                bank=root/f'steps_{n:02d}'/'predictions'
                atomic(bank/f'shard_{rank}.json',{'status':'paused' if stopped else 'complete','requested':len(ids),
                     'completed':complete[n],'failed':failed[n],'identity_sha256':signature(read(bank/'identity.json'))})
        if sum(failed.values()):raise RuntimeError('Failed scenes retained; sweep invalid')


def group(registration_path, host_index, attempt, smoke=False):
    registration,config=load_registration(registration_path);host=config['hosts'][host_index]
    if host['hostname']!=socket.gethostname():raise ValueError('Host identity changed')
    job=Path(config['artifact_root'])/(f'host_{host_index}_smoke' if smoke else f'host_{host_index}')
    job.mkdir(exist_ok=True);processes=[]
    requested=[host['gpus'][0]] if smoke else host['gpus']
    resource=dict(config,gpus=requested,evaluation_source_sha=registration['source_sha'])
    with lease(job/'allocation.lock'):
        try:
            release_owned_pressure(resource,host,job,reason='User-requested frozen-weight Euler-step sensitivity')
            deadline=time.time()+60
            while True:
                output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
                used={int(l.split(',')[0]):int(l.split(',')[1]) for l in output.splitlines()}
                if all(used.get(g,81920)<512 for g in requested):break
                if time.time()>deadline:raise RuntimeError('GPU still occupied; refusing to interrupt another process')
                time.sleep(2)
            for j,gpu in enumerate(requested):
                rank=host_index*8+j
                command=[config['inference_python'],'-u','-m',MODULE,'export','--registration',str(registration_path),
                         '--rank',str(rank),'--attempt',str(attempt)]
                if smoke:command+=['--smoke-count','4']
                with (job/f'rank_{rank}_a{attempt}.log').open('x') as log:
                    processes.append(subprocess.Popen(command,cwd=config['source_worktree'],env=environment(gpu),
                         stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
            codes=[p.wait() for p in processes]
            if any(codes):raise RuntimeError(f'Export group failure {codes}; logs retained')
        finally:
            # Wait only for this group's children before restoring idle reserves.
            for process in processes:process.wait()
            restore_owned_pressure(resource,host,job)


def command_host(config,host,command):
    if not host['host']:return command
    return ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host['host'],
            'cd '+shlex.quote(config['source_worktree'])+' && exec '+shlex.join(command)]


def run(registration_path, attempt):
    registration,config=load_registration(registration_path);root=Path(config['artifact_root'])
    if not (root/'SMOKE.json').exists() or read(root/'SMOKE.json')['status']!='COMPLETE':
        raise ValueError('Complete four-scene original-path parity smoke first')
    with lease(root/'campaign.lock'):
        atomic(root/'status.json',{'status':'RUNNING','pid':os.getpid(),'registration':registration['identity'],
               'source_sha':registration['source_sha'],'started_unix':time.time(),'real_optimizer_updates':0})
        groups=[]
        try:
            for i,host in enumerate(config['hosts']):
                args=[config['orchestrator_python'],'-u','-m',MODULE,'group','--registration',str(registration_path),
                      '--host-index',str(i),'--attempt',str(attempt)]
                with (root/f'group_{i}_a{attempt}.log').open('x') as log:
                    groups.append(subprocess.Popen(command_host(config,host,args),cwd=config['source_worktree'],
                        stdout=log,stderr=subprocess.STDOUT,env=environment(),start_new_session=True))
            def score(n):
                job=root/f'steps_{n:02d}'
                args=[config['scoring_python'],'-u','-m','tools.foresight.score_pdms','--devkit',config['devkit'],
                      '--metric-index',config['metric_index'],'--current-index',str(Path(config['current_root'])/'index.json'),
                      '--predictions',str(job/'predictions'),'--output',str(job/'scores'),'--campaign-root',config['campaign_root'],
                      '--run-id',f'fm_sweep_{registration["identity"][:12]}_s{n}_a{attempt}',
                      '--workers',str(config['cpu_workers']),'--timeout-seconds',str(config['max_seconds'])]
                if (job/'scores'/'summary.json').exists():return
                if (job/'scores').exists():args+=['--resume']
                with (job/f'score_a{attempt}.log').open('x') as log:
                    subprocess.run(args,cwd=config['source_worktree'],env=environment(),stdout=log,
                        stderr=subprocess.STDOUT,check=True,timeout=config['max_seconds']+300)
            with ThreadPoolExecutor(max_workers=config['cpu_parallel']) as pool:
                futures=[pool.submit(score,n) for n in config['steps']]
                codes=[p.wait() for p in groups]
                if any(codes):raise RuntimeError(f'Export host failure: {codes}')
                for future in futures:future.result()
            summarize(registration_path)
            atomic(root/'status.json',{'status':'COMPLETE','registration':registration['identity'],
                  'source_sha':registration['source_sha'],'real_optimizer_updates':0,
                  'gpu_hours':sweep_usage(registration,config),'completed_unix':time.time()})
        except BaseException as error:
            atomic(root/'STOP_REQUESTED',{'reason':repr(error),'unix':time.time()})
            atomic(root/'status.json',{'status':'PAUSED' if isinstance(error,KeyboardInterrupt) else 'FAILED',
                  'error':repr(error),'registration':registration['identity'],'real_optimizer_updates':0,
                  'gpu_hours':sweep_usage(registration,config)})
            raise


def summarize(registration_path):
    import numpy as np
    from tools.local_interaction_mask_v2.compare_pdms import read as read_scores, compare, METRICS
    registration,config=load_registration(registration_path);root=Path(config['artifact_root'])
    baseline=read_scores(root/'steps_10'/'scores'/'scenes.csv');historical=read_scores(Path(config['baseline_scores'])/'scenes.csv')
    if set(baseline)!=set(historical):raise ValueError('Original 10-step population differs')
    max_difference=max(abs(float(baseline[t][k])-float(historical[t][k])) for t in baseline for k in METRICS)
    if max_difference>1e-8:raise ValueError(f'Original 10-step scoring parity failed: {max_difference}')
    rows=[];paired={};timings={}
    for n in config['steps']:
        job=root/f'steps_{n:02d}';summary=read(job/'scores'/'summary.json');population=read_scores(job/'scores'/'scenes.csv')
        if (not summary['valid'] or summary['failed'] or summary['scenes']!=12146 or summary['logs']!=136 or
                summary['export_identity']['checkpoint']!=registration['checkpoint'] or
                summary['export_identity']['protocol']['steps']!=n or set(population)!=set(baseline) or
                any(population[t]['metric_cache_sha256']!=baseline[t]['metric_cache_sha256'] for t in baseline)):
            raise ValueError('Incomplete, changed checkpoint/cache/population sweep')
        paired[str(n)]=compare(population,baseline)[0]
        metadata=[read(job/'predictions'/'predictions'/(t+'.json')) for t in population]
        if any(r['status']!='ok' for r in metadata):raise ValueError('Inference failures retained')
        latency=[r['standalone_inference_seconds'] for r in metadata if 'standalone_inference_seconds' in r]
        timings[str(n)]={'standalone_scenes':len(latency),'precision':'FP32','tf32':False,
            'batch':1,'includes':'current encode + original Euler + decode; excludes image I/O/model loading',
            'p50_seconds':float(np.median(latency)),'p95_seconds':float(np.quantile(latency,.95)),
            'all_scenes_solver_p50_seconds':float(np.median([r['solver_seconds'] for r in metadata]))}
        row={'steps':n,'dt':1/n,'PDMS':100*summary['PDMS'],'delta_PDMS_vs10':100*(summary['PDMS']-read(root/'steps_10'/'scores'/'summary.json')['PDMS']),
             'zero_scenes':sum(float(r['score'])==0 for r in population.values()),'failed':summary['failed'],
             **{k:100*v for k,v in summary['metrics'].items()},**timings[str(n)]}
        rows.append(row)
    with (root/'RESULTS.csv').open('w') as stream:
        writer=csv.DictWriter(stream,list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic(root/'RESULTS.json',{'schema':'ddp_fm_step_sweep_results_v1','registration':registration['identity'],
           'checkpoint':registration['checkpoint'],'rows':rows,'paired_vs10':paired,
           'original10_maximum_metric_difference':max_difference,'cost_gpu_hours':sweep_usage(registration,config),
           'inference_seed':config['sampling_seed'],'training_updates':0,
           'interpretation':'Fixed-weight, single-noise Navtest sensitivity diagnostic; no automatic production protocol selection'})


def main():
    parser=argparse.ArgumentParser(__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('register');p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    for name in ('export','group','run','summarize'):
        p=sub.add_parser(name);p.add_argument('--registration',required=True)
        if name!='summarize':p.add_argument('--attempt',type=int,default=1)
        if name=='export':p.add_argument('--rank',type=int,required=True);p.add_argument('--smoke-count',type=int,default=0)
        if name=='group':p.add_argument('--host-index',type=int,required=True);p.add_argument('--smoke',action='store_true')
    a=parser.parse_args()
    if a.command=='register':register(a.config,a.output)
    elif a.command=='export':export(a.registration,a.rank,a.attempt,a.smoke_count)
    elif a.command=='group':group(a.registration,a.host_index,a.attempt,a.smoke)
    elif a.command=='run':run(a.registration,a.attempt)
    else:summarize(a.registration)


if __name__=='__main__':main()
