"""Explicit S0--S4 50k:10k:100k Navtest queue; never modifies a trainer.

The fast observer hard-links COMPLETE exact checkpoints before rolling retention.
Separate workers hash/load them, share the existing host evaluation lease, and
score locally on the canonical CPU host. All model/metric code stays immutable.
"""
import argparse
import contextlib
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

from tools.full_foresight.navtest_schedule import (
    atomic, busy, find_exact, lease, preserve, read, sha, source_identity,
    validate_scores, write_csv,
)
from starVLA.model.modules.vehicle_joint.initialization import identity_hash

MODULE = 'tools.action_video_foresight.navtest_milestones'
UPDATES = [50000, 60000, 70000, 80000, 90000, 100000]
ARMS = ['S0', 'S1', 'S2', 'S3', 'S4']
TERMINAL = {'COMPLETE', 'FAILED', 'MISSED_CHECKPOINT'}


def environment(gpu=''):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                TOKENIZERS_PARALLELISM='false')


def model_for(config, key):
    arm, number = key.split('_')
    if arm not in ARMS or int(number) not in UPDATES or key != f'{arm}_{int(number):06d}':
        raise ValueError('Task not in the explicit 30-evaluation registration')
    return config['models'][arm], int(number)


def validate_arm(plan, arm):
    root = Path(plan['campaign_root']); spec = plan['runs'][arm]
    path = root/'registrations'/(spec['run_id']+'.json'); reg = read(path)
    run = root/'students'/spec['run_id']; training = read(run/'identity.json')
    if (training['identity'] != identity_hash({k:v for k,v in training.items() if k != 'identity'})
            or training['registration_sha256'] != sha(path)
            or reg['arm'] != arm or reg['run_id'] != spec['run_id']
            or reg['training_source_sha'] != plan['training_source_sha']
            or training['source_sha'] != plan['training_source_sha']
            or training['scope'] != 'formal' or training['schema'] != 'ddp_action_video_student_v1'
            or training['updates'] != 100000 or training['world_size'] != 8
            or training['global_batch'] != 32
            or training['config']['framework']['action_model']['num_inference_timesteps'] != 10):
        raise ValueError('Wrong formal S-arm identity (C1 denotes visual configuration only)')
    return dict(arm=arm, training_run=str(run), run_identity=training['identity'],
                training_source_sha=training['source_sha'], host=spec['host'],
                hostname=spec['hostname'], gpus=spec['gpus'], registration=str(path))


def register(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Never overwrite a Navtest registration')
    plan = read(args.plan)
    if plan['identity'] != identity_hash({k:v for k,v in plan.items() if k != 'identity'}):
        raise ValueError('Formal plan changed')
    # Reuse audited assets only; do not reuse old C-model queues or budgets.
    assets = read(args.asset_registration)['config']
    keys = ['current_root', 'metric_index', 'devkit', 'cache_snapshot', 'cache_audit',
            'ego_labels', 'inference_python', 'scoring_python']
    config = {k:assets[k] for k in keys}
    config.update(schema='action_video_navtest_schedule_v1', plan=str(Path(args.plan).resolve()),
        plan_identity=plan['identity'], models={arm:validate_arm(plan, arm) for arm in ARMS},
        artifact_root=str(output.parent.resolve()), campaign_root=plan['campaign_root'],
        source_worktree=str(Path.cwd()), observer_source=source_identity(Path.cwd()),
        evaluation_worktree=plan['source_worktree'], evaluation_source=plan['training_source_sha'],
        canonical_hostname=socket.gethostname(), worker_python=sys.executable,
        updates=UPDATES, sampling_seed=42, gpu_memory_fraction=.45, minimum_free_mib=20000,
        cpu_slots=2, cpu_workers=16, poll_seconds=5, task_timeout_seconds=21600,
        gpu_hours_limit=None, total_tasks=30,
        host_lease_root=str(Path(plan['campaign_root'])/'evaluation_leases'),
        authorization='User requested each S0-S4 at 50k then every10k through100k; replaces final-only Navtest timing. No test-based training/configuration selection.')
    if source_identity(config['evaluation_worktree']) != config['evaluation_source']:
        raise ValueError('Immutable model/metric source changed')
    audit = read(config['cache_audit']); index = read(Path(config['current_root'])/'index.json')
    current = read(Path(config['current_root'])/'identity.json'); metric = read(config['metric_index'])
    if (current['split'] != 'navtest' or len(index) != 12146 or len(metric) != 12146
            or len({r['token'] for r in index}) != 12146 or len({r['log'] for r in index}) != 136
            or {(r['token'],r['log']) for r in index} != {(r['token'],r['log']) for r in metric}
            or identity_hash(index) != current['index_sha256'] or not audit['passed']
            or sha(config['metric_index']) != audit['metric_index_sha256']
            or sha(config['cache_snapshot']) != audit['cache_snapshot_sha256']):
        raise ValueError('Canonical full Navtest assets changed')
    from tools.local_interaction_mask_v2.score_async import python_tree_digest
    if python_tree_digest(Path(config['devkit'])/'navsim') != audit['navsim_tree_sha256']:
        raise ValueError('Audited official scoring implementation changed')
    files = [args.plan, config['metric_index'], config['cache_audit'], config['cache_snapshot'],
             str(Path(config['current_root'])/'identity.json'), str(Path(config['current_root'])/'index.json')]
    for model in config['models'].values():
        files += [str(Path(model['training_run'])/'identity.json'), model['registration']]
    value = {'config':config, 'asset_hashes':{str(Path(p).resolve()):sha(p) for p in files},
             'created_unix':time.time()}
    value['identity'] = identity_hash(value)
    atomic(output, value)
    return value


def load_registration(path):
    r = read(path); config = r['config']
    if (r['identity'] != identity_hash({k:v for k,v in r.items() if k != 'identity'})
            or config['schema'] != 'action_video_navtest_schedule_v1'
            or config['updates'] != UPDATES or set(config['models']) != set(ARMS)):
        raise ValueError('Immutable registration changed')
    if source_identity(config['source_worktree']) != config['observer_source']:
        raise ValueError('Observer source changed')
    if source_identity(config['evaluation_worktree']) != config['evaluation_source']:
        raise ValueError('Model/scoring source changed')
    for path, expected in r['asset_hashes'].items():
        if sha(path) != expected:
            raise ValueError('Registered asset changed: '+path)
    return r


def build_lock(config, model, update, checkpoint):
    if (checkpoint['run_identity'] != model['run_identity'] or checkpoint['completed'] != update
            or checkpoint['training_source_sha'] != model['training_source_sha']
            or checkpoint['scope'] != 'formal' or update not in UPDATES
            or checkpoint['model_class'] != 'starVLA.model.framework.ddp_action_video_foresight.DDPActionVideoForesight'):
        raise ValueError('Wrong exact model/checkpoint for Navtest task')
    current = read(Path(config['current_root'])/'identity.json')
    value = dict(schema='foresight_navtest_checkpoint_probe_lock_v1',
        evaluation_purpose='user_requested_fixed_checkpoint', requested_update=update,
        authorization=config['authorization'], experimental_arm=model['arm'],
        evaluation_source_sha=config['evaluation_source'], observer_source_sha=config['observer_source'],
        checkpoints=[checkpoint['sha256']], checkpoint_records={checkpoint['sha256']:checkpoint},
        current_data_identity=current['identity'], current_index_identity=current['index_sha256'],
        scene_count=12146, log_count=136, official_metric_index_sha256=sha(config['metric_index']),
        sampling_seeds=[42], inference_steps=10, candidates_per_scene=1, learned_scorer=None,
        precision='FP32', final_endpoint_comparison=False,
        checkpoint_selection='Exact user-requested steps; no best-Navtest selection')
    value['identity'] = identity_hash(value)
    return value


def export_complete(bank, lock=None):
    bank = Path(bank)
    if not (bank/'identity.json').exists():
        return False
    manifest = read(bank/'identity.json'); protocol = manifest['protocol']
    if (manifest['limit'] or manifest['world_size'] != 8
            or manifest['current_identity']['split'] != 'navtest'
            or protocol['precision'] != 'FP32' or protocol['tf32']
            or protocol['steps'] != 10 or protocol['sampling_seed'] != 42
            or protocol['future_conditioning'] or protocol['scorer'] is not None
            or protocol['candidates_per_scene'] != 1 or not protocol['auxiliary_heads_removed']):
        raise ValueError('Changed inference protocol or partial population')
    if lock:
        from tools.foresight.lock_navtest import validate_lock
        validate_lock(lock, manifest['checkpoint'], manifest['evaluation_source'],
                      manifest['current_identity'], 12146, 42, 10)
    for rank in range(8):
        path = bank/f'shard_{rank}.json'
        if not path.exists():
            return False
        row = read(path)
        if row['identity_sha256'] != identity_hash(manifest) or row['failed']:
            raise ValueError('Invalid export shard; failed scenes retained')
        if row['status'] != 'complete':
            return False
        if row['completed'] != len(range(rank,12146,8)):
            raise ValueError('Incomplete Navtest population')
    return True


def remote_command(config, model, command):
    if model['hostname'] == socket.gethostname():
        return command
    return ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',model['host'],
            'cd '+shlex.quote(config['source_worktree'])+' && exec '+shlex.join(command)]


def launch(command, cwd, log, env=None):
    with Path(log).open('x') as stream:
        return subprocess.Popen(command, cwd=cwd, env=env or environment(),
                                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)


def gpu_headroom(config, model):
    text = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free',
                                   '--format=csv,noheader,nounits'],text=True)
    free = {int(r.split(',')[0]):int(r.split(',')[1]) for r in text.splitlines()}
    return all(free.get(card,0) >= config['minimum_free_mib'] for card in model['gpus'])


def export_group(path, key, attempt, preflight=False):
    registration = load_registration(path); config = registration['config']
    model, update = model_for(config,key); job = Path(config['artifact_root'])/'jobs'/key
    if socket.gethostname() != model['hostname']:
        raise ValueError('Wrong authorized GPU host')
    if preflight:
        # Real image references on every authorized host; no future/teacher data.
        root = Path(config['current_root']); index = read(root/'index.json')
        from starVLA.dataloader.foresight_dataset import ForesightCurrentDataset
        data = ForesightCurrentDataset(root)
        for i in (0, len(index)//2, len(index)-1):
            data[i]
        return dict(host=socket.gethostname(), ready=gpu_headroom(config,model),
                    current_scenes=len(index), tested_images=9, optimizer_updates=0)
    with lease(job/'gpu.lock'):
        # Same filesystem lock as the unchanged DEV exporter. No overlapping
        # model-load waves on the same host; training continues concurrently.
        with lease(Path(config['host_lease_root'])/(model['hostname']+'.lock')):
            if not gpu_headroom(config,model):
                return 75
            bank = job/'predictions'; lock = read(job/'lock.json'); snapshot = read(job/'snapshot.json')
            children = []
            try:
                for rank, card in enumerate(model['gpus']):
                    shard = bank/f'shard_{rank}.json'
                    if shard.exists() and read(shard)['status']=='complete' and not read(shard)['failed']:
                        continue
                    rid = f'navtest_{registration["identity"][:12]}_{key}_a{attempt}_rank{rank}'
                    command = [config['inference_python'],'-u','-m','tools.foresight.export_predictions',
                        '--training-run',snapshot['training_run'],'--checkpoint-tag',snapshot['tag'],
                        '--current-root',config['current_root'],'--output',str(bank),
                        '--campaign-root',config['campaign_root'],'--run-id',rid,
                        '--sampling-seed','42','--rank',str(rank),'--world-size','8',
                        '--max-seconds',str(config['task_timeout_seconds']), '--campaign-gpu-hours','1000000000',
                        '--gpu-memory-fraction',str(config['gpu_memory_fraction']), '--final-lock',str(job/'lock.json')]
                    child = launch(command,config['evaluation_worktree'],job/(rid+'.log'),environment(card))
                    children.append(child)
                    atomic(job/f'export_a{attempt}_rank{rank}.json',dict(pid=child.pid,command=command,
                        host=socket.gethostname(),started_unix=time.time(),optimizer_updates=0))
            finally:
                # Never orphan a partially launched wave, nor signal a trainer.
                codes = [child.wait() for child in children]
            if any(codes):
                raise RuntimeError('Export failed, all partial/failed rows retained: '+str(codes))
            return 0 if export_complete(bank,lock) else 76


@contextlib.contextmanager
def cpu_slot(config):
    root = Path(config['artifact_root'])/'cpu_slots'
    while True:
        for i in range(config['cpu_slots']):
            guard = lease(root/f'{i}.lock')
            try:
                guard.__enter__()
            except BlockingIOError:
                continue
            try:
                yield
            finally:
                guard.__exit__(None,None,None)
            return
        time.sleep(5)


def finalize(config, job, model, update):
    summary, rows = validate_scores(job/'scores', config['cache_snapshot'], update)
    lock = read(job/'lock.json')
    if not export_complete(job/'predictions',lock) or summary['export_identity'] != read(job/'predictions/identity.json'):
        raise ValueError('Score/export identity mismatch')
    if summary['export_identity']['checkpoint']['run_identity'] != model['run_identity']:
        raise ValueError('Wrong experimental model')
    fit = read(job/'ego/summary.json')
    if not fit['valid'] or fit['failed'] or fit['scenes'] != 12146:
        raise ValueError('Incomplete ego evaluation')
    ego = {r['token']:r for r in csv.DictReader((job/'ego/scenes.csv').open())}
    combined = []
    for row in rows:
        item = ego[row['token']]
        if item['failure'] or item['log'] != row['log']:
            raise ValueError('Ego population mismatch')
        values = {k:float(item[k]) for k in ('ADE','FDE','yaw_MAE_rad')}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError('Nonfinite ego metric')
        combined.append(dict(row, experimental_arm=model['arm'], checkpoint_update=update, **values))
    write_csv(job/'complete.csv',combined)
    result = dict(status='COMPLETE', arm=model['arm'], update=update, scenes=12146, logs=136,
        failed=0, zero_scenes=sum(float(r['score'])==0 for r in rows),
        PDMS_points=100*summary['PDMS'], metrics_points={k:100*v for k,v in summary['metrics'].items()},
        ego=fit['groups']['all'], checkpoint=summary['export_identity']['checkpoint'],
        training_source=model['training_source_sha'], evaluation_source=config['evaluation_source'],
        observer_source=config['observer_source'], complete_csv_sha256=sha(job/'complete.csv'),
        completed_unix=time.time(), sampling_seed=42, precision='FP32 masters/FP32 compute/TF32off',
        optimizer_updates_in_evaluation=0)
    atomic(job/'result.json',result)
    return result


def task(path,key):
    registration = load_registration(path); config = registration['config']
    if socket.gethostname() != config['canonical_hostname']:
        raise ValueError('Scoring must run on the canonical host, never SSH back from exporters')
    model, update = model_for(config,key); job = Path(config['artifact_root'])/'jobs'/key
    with lease(job/'worker.lock'):
        state = read(job/'status.json') if (job/'status.json').exists() else {}
        if state.get('status') in TERMINAL:
            return
        attempt = state.get('attempt',0)+1
        def status(**kw):
            state.update(kw, attempt=attempt, updated_unix=time.time(), registration=registration['identity'])
            atomic(job/'status.json',state)
        status(status='RUNNING',phase='LOCK',pid=os.getpid())
        try:
            snapshot = read(job/'snapshot.json')
            if not (job/'lock.json').exists():
                from tools.foresight.checkpoints import checkpoint_identity
                _, checkpoint = checkpoint_identity(snapshot['training_run'],snapshot['tag'])
                atomic(job/'lock.json',build_lock(config,model,update,checkpoint))
            lock = read(job/'lock.json')
            if not export_complete(job/'predictions',lock):
                status(phase='GPU_EXPORT')
                command = remote_command(config,model,[config['worker_python'],'-u','-m',MODULE,
                    'export-group','--registration',str(path),'--task',key,'--attempt',str(attempt)])
                child = launch(command,config['source_worktree'],job/f'export_group_a{attempt}.log')
                atomic(job/f'export_group_a{attempt}.json',dict(pid=child.pid,command=command))
                code = child.wait()
                if code in (75,76,255):
                    status(status='QUEUED',phase='WAIT_RESOURCE_OR_TRANSPORT',retry_after=time.time()+60,exit_code=code)
                    return
                if code or not export_complete(job/'predictions',lock):
                    raise RuntimeError('Export failed; retained artifacts, exit '+str(code))
            status(phase='WAIT_CPU_SLOT')
            with cpu_slot(config):
                status(phase='CANONICAL_CPU_SCORE')
                if not (job/'scores/summary.json').exists():
                    command = [config['scoring_python'],'-u','-m','tools.foresight.score_pdms',
                        '--devkit',config['devkit'],'--metric-index',config['metric_index'],
                        '--current-index',str(Path(config['current_root'])/'index.json'),
                        '--predictions',str(job/'predictions'),'--output',str(job/'scores'),
                        '--campaign-root',config['campaign_root'],
                        '--run-id',f'navtest_{registration["identity"][:12]}_{key}_a{attempt}_score',
                        '--workers',str(config['cpu_workers']),'--timeout-seconds',str(config['task_timeout_seconds'])]
                    if (job/'scores/identity.json').exists():
                        command.append('--resume')
                    code = launch(command,config['evaluation_worktree'],job/f'score_a{attempt}.log').wait()
                    if code:
                        raise RuntimeError('Canonical scoring failed; every row retained')
                if not (job/'ego/summary.json').exists():
                    ego = job/f'ego_attempt_{attempt}'
                    command = [config['inference_python'],'-m','tools.foresight.evaluate_ego',
                        '--predictions',str(job/'predictions'),'--current-root',config['current_root'],
                        '--processed-root',config['ego_labels'],'--output',str(ego)]
                    if launch(command,config['evaluation_worktree'],job/f'ego_a{attempt}.log').wait():
                        raise RuntimeError('Ego evaluation failed')
                    ego.rename(job/'ego')
                finalize(config,job,model,update)
            status(status='COMPLETE',phase='COMPLETE')
        except Exception as error:
            status(status='FAILED',error=repr(error))
            raise


def alive(record,path,key):
    try:
        args = Path(f'/proc/{record["pid"]}/cmdline').read_bytes().split(b'\0')
        args = [a.decode() for a in args if a]
        stat = Path(f'/proc/{record["pid"]}/stat').read_text().split(') ',1)[1].split()[0]
        return stat != 'Z' and MODULE in args and 'task' in args and str(path) in args and key in args
    except (FileNotFoundError,KeyError,ProcessLookupError):
        return False


def watch(path,once=False):
    path = Path(path).resolve(); registration = load_registration(path); config = registration['config']
    root = Path(config['artifact_root'])
    if socket.gethostname() != config['canonical_hostname']:
        raise ValueError('Wrong observer host')
    with lease(root/'observer.lock'):
        if (root/'status.json').exists() and read(root/'status.json')['registration'] != registration['identity']:
            raise ValueError('Different registration already uses this output')
        children = []
        while True:
            states = {}; stopped = (root/'STOP_SCHEDULING').exists()
            # Fast preservation is separate from hashing and scoring.
            for arm, model in config['models'].items():
                progress = read(Path(model['training_run'])/'status.json')
                if progress['identity'] != model['run_identity']:
                    raise ValueError('Training status identity changed')
                for update in UPDATES:
                    key = f'{arm}_{update:06d}'; job = root/'jobs'/key
                    state = read(job/'status.json') if (job/'status.json').exists() else {'status':'WAITING'}
                    if not (job/'snapshot.json').exists() and state['status'] not in TERMINAL:
                        try:
                            snapshot = preserve(model,update,job)
                            if snapshot is not None:
                                state.update(status='QUEUED')
                            elif progress['completed'] >= update+200:
                                state.update(status='MISSED_CHECKPOINT',error='Exact checkpoint gone; never substitute another step')
                        except Exception as error:
                            # One corrupt/missed checkpoint cannot disable the
                            # other 29 preservation/evaluation jobs.
                            state.update(status='FAILED',error=repr(error))
                        atomic(job/'status.json',state)
                    states[key] = state['status']
            for arm, model in config['models'].items():
                jobs = [root/'jobs'/f'{arm}_{u:06d}' for u in UPDATES]
                if stopped or any(busy(j/'worker.lock') or busy(j/'gpu.lock') or
                    ((j/'launch.json').exists() and alive(read(j/'launch.json'),path,j.name)) for j in jobs):
                    continue
                for job in jobs:
                    state = read(job/'status.json')
                    if state['status'] in TERMINAL or not (job/'snapshot.json').exists() or state.get('retry_after',0)>time.time():
                        continue
                    if state['status'] in ('RUNNING','STARTING'):
                        # A dead local task with a live CPU scorer must not be duplicated.
                        # Full process recovery is explicit if the task was interrupted.
                        state.update(status='FAILED',error='Task exited without completion; inspect retained child evidence before resume')
                        atomic(job/'status.json',state);states[job.name]='FAILED';continue
                    command = [config['worker_python'],'-u','-m',MODULE,'task',
                               '--registration',str(path),'--task',job.name]
                    atomic(job/'status.json',dict(state,status='STARTING',updated_unix=time.time()))
                    child = launch(command,config['source_worktree'],job/f'task_{time.time_ns()}.log')
                    children.append(child)
                    atomic(job/'launch.json',dict(pid=child.pid,command=command,started_unix=time.time()))
                    states[job.name]='STARTING';break
            for child in list(children):
                if child.poll() is not None:
                    children.remove(child)
            results = [read(p) for p in sorted((root/'jobs').glob('*/result.json'))]
            write_csv(root/'SUMMARY.csv',[{k:r[k] for k in ('arm','update','PDMS_points','scenes','failed','zero_scenes')} for r in results])
            finished = all(s in TERMINAL for s in states.values())
            atomic(root/'TASKS.json',states)
            atomic(root/'status.json',dict(registration=registration['identity'],pid=os.getpid(),
                status=('COMPLETE' if all(s=='COMPLETE' for s in states.values()) else 'FINISHED_WITH_FAILURES') if finished else 'PAUSED' if stopped else 'RUNNING',
                tasks=states, completed_tasks=sum(s=='COMPLETE' for s in states.values()),
                updated_unix=time.time(), optimizer_updates=0))
            if once or finished:
                return
            time.sleep(config['poll_seconds'])


def main():
    parser = argparse.ArgumentParser(__doc__); sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('register')
    for key in ('plan','asset-registration','output'):
        p.add_argument('--'+key,required=True)
    for mode in ('watch','task','export-group','preflight'):
        p = sub.add_parser(mode);p.add_argument('--registration',required=True)
        if mode == 'watch':
            p.add_argument('--once',action='store_true')
        else:
            p.add_argument('--task',required=True)
            p.add_argument('--attempt',type=int,default=1)
    args = parser.parse_args()
    if args.command == 'register':
        print(json.dumps(register(args)))
    elif args.command == 'watch':
        watch(args.registration,args.once)
    elif args.command == 'task':
        task(args.registration,args.task)
    else:
        try:
            result = export_group(args.registration,args.task,args.attempt,args.command=='preflight')
        except BlockingIOError:
            raise SystemExit(75)
        if isinstance(result,dict):
            print(json.dumps(result))
        else:
            raise SystemExit(result)


if __name__ == '__main__':
    main()
