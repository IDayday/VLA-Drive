"""Host-local finite formal run: wait for complete targets, train fresh, evaluate dev.

Never starts a Navtest observer, edits an older worktree, or signals a foreign job.
The plan is immutable; restarting a controller requires explicit acknowledgement.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
from starVLA.model.modules.vehicle_joint.initialization import identity_hash, file_sha256
from tools.ddpolicy_vehicle.prepare_data import atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def complete_targets(root, expected_identity, scenes):
    root = Path(root)
    if read(root/'identity.json')['identity'] != expected_identity:
        raise ValueError('Registered target identity changed')
    if not (root/'COMPLETE.json').exists():
        return False
    done = read(root/'COMPLETE.json')
    if done['identity'] != expected_identity or done['scenes'] != scenes:
        raise ValueError('Complete marker has a different population')
    return all((root/f'chunk_{i:06d}.json').exists() and
               (root/f'chunk_{i:06d}.safetensors').exists() for i in range(done['chunks']))


def ready_gpus(cards, pressure_script):
    # Auto-yielding pressure is allowed to remain; every other compute PID blocks.
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                    '--format=csv,noheader,nounits'], text=True)
    devices = {uid.strip(): (int(index), int(free))
               for index, uid, free in (row.split(',') for row in rows.splitlines())}
    if any(next((free for index, free in devices.values() if index == card), 0) < 32000
           for card in cards):
        return False
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'], text=True)
    for line in apps.splitlines():
        if not line.strip():
            continue
        uid, pid = line.split(',')
        if devices.get(uid.strip(), (-1, 0))[0] not in cards:
            continue
        try:
            proc = Path('/proc')/str(int(pid))
            args = [x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
            is_yielding_pressure = (pressure_script in args and '--gpu' in args and
                                    proc.stat().st_uid == os.getuid())
        except FileNotFoundError:
            # A driver PID outside our namespace is not evidence of an idle GPU.
            return False
        if not is_yielding_pressure:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--arm', choices=('S0','S1','S2','S3','S4'), required=True)
    parser.add_argument('--resume-controller', action='store_true')
    parser.add_argument('--acknowledge-stop', action='store_true')
    args = parser.parse_args()
    plan = read(args.plan)
    if plan['identity'] != identity_hash({k:v for k,v in plan.items() if k!='identity'}):
        raise ValueError('Campaign plan changed')
    spec = plan['runs'][args.arm]
    if socket.gethostname() != spec['hostname']:
        raise ValueError('Wrong authorized host')
    work = Path(plan['source_worktree'])
    source = subprocess.check_output(['git','rev-parse','HEAD'],cwd=work,text=True).strip()
    if source != plan['training_source_sha'] or subprocess.check_output(['git','status','--porcelain'],cwd=work).strip():
        raise ValueError('Formal source must remain frozen')
    if file_sha256(plan['calibration']) != plan['calibration_sha256']:
        raise ValueError('Training-only calibration changed')
    if file_sha256(plan['base_config']) != plan['base_config_sha256']:
        raise ValueError('Public/random initialization configuration changed')
    root = Path(plan['campaign_root'])
    out = root/'formal_controllers'/spec['run_id']
    out.mkdir(parents=True,exist_ok=True)
    lock = (out/'RUN.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path = out/'status.json'
    if state_path.exists():
        state = read(state_path)
        if not args.resume_controller or state['plan_identity'] != plan['identity']:
            raise ValueError('Existing campaign controller requires explicit unchanged resume')
        if state['status'] == 'COMPLETE':
            return
    else:
        state = {'plan_identity':plan['identity'],'arm':args.arm,'events':[]}
    state.update(pid=os.getpid(),host=socket.gethostname(),status='WAITING_TARGETS')
    stopping = [False]
    signal.signal(signal.SIGINT,lambda *_:stopping.__setitem__(0,True))
    signal.signal(signal.SIGTERM,lambda *_:stopping.__setitem__(0,True))
    children = []
    run = root/'students'/spec['run_id']
    def event(kind, **details):
        state['events'].append(dict(time=time.time(),event=kind,**details))
        atomic_json(state_path,state)
    def stopped():
        return stopping[0] or (root/'STOP_REQUESTED').exists() or (out/'STOP_REQUESTED').exists()
    def launch(label, command, env=None, cwd=None):
        log = out/(label+'_'+str(time.time_ns())+'.log')
        with log.open('x') as stream:
            proc = subprocess.Popen(command,cwd=cwd or work,env=env,
                                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        event('START',label=label,pid=proc.pid,command=command,log=str(log))
        return proc
    try:
        event('WAIT_FOR_COMPLETE_NATIVE_TARGETS',expected_identity=spec['clip_identity'])
        while not complete_targets(spec['clip_root'],spec['clip_identity'],plan['scene_count']):
            if stopped():
                state['status']='PAUSED';event('STOP_BEFORE_TRAINING');return
            time.sleep(10)
        for asset, marker in spec.get('local_asset_markers', {}).items():
            state['status']='WAITING_LOCAL_ASSETS';event('WAIT_LOCAL_ASSET',asset=asset,marker=marker)
            while not Path(marker).exists():
                if stopped():
                    state['status']='PAUSED';event('STOP_BEFORE_TRAINING');return
                time.sleep(10)
        state['status']='WAITING_GPU'
        event('TARGETS_COMPLETE')
        # Release only original ledger-verified reserve parents, never a trainer.
        if plan.get('old_pressure_registration'):
            from tools.full_foresight.navtest_milestones import release_owned_pressure
            prior = read(plan['old_pressure_registration'])['config']
            for model in prior['models']:
                if model['hostname']==socket.gethostname():
                    release_owned_pressure(prior,model,out,reason='Explicit new five-host training authorization; verified owned pressure only')
        while not ready_gpus(spec['gpus'],plan['auto_yield_pressure_script']):
            if stopped():
                state['status']='PAUSED';event('STOP_BEFORE_TRAINING');return
            time.sleep(10)
        command=[sys.executable,'-u','-m','tools.action_video_foresight.run_experiments']
        values={'base-config':plan['base_config'],'data':spec.get('train_data',plan['train_data']),
                'dino-root':spec.get('dino_root',plan['dino_root']),'dino-index':plan['dino_index'],
                'interaction-root':spec.get('interaction_root',plan['interaction_root']),'clip-root':spec['clip_root'],
                'campaign-root':str(root),'run-id':spec['run_id'],'arm':args.arm,
                'calibration':plan['calibration'],'scope':'formal','updates':plan['updates'],
                'schedule-updates':plan['updates'],'gpus':len(spec['gpus']),
                'micro-batch':plan['micro_batch'],'seed':plan['seed'],
                'master-port':spec['master_port'],'milestones':','.join(map(str,plan['milestones']))}
        for key,value in values.items():
            command+=['--'+key,str(value)]
        if spec.get('local_image_root'):
            command+=['--local-image-root',spec['local_image_root']]
        if (run/'status.json').exists():
            previous=read(run/'status.json')
            if previous['status']=='COMPLETE':
                event('TRAINING_ALREADY_COMPLETE')
                # A previous export/scoring failure must remain recoverable without
                # attempting to extend a completed optimizer run.
                for update in plan['evaluation_updates']:
                    children.append(launch(spec['run_id']+'_dev'+str(update),
                        [sys.executable,'-u','-m','tools.action_video_foresight.evaluate_milestone',
                         '--plan',args.plan,'--arm',args.arm,'--update',str(update)]))
                if any(p.wait() for p in children):
                    raise RuntimeError('Completed training still has failed development evaluations')
                state['status']='COMPLETE';event('REGISTERED_DEV_RECOVERED');return
            if previous['status'] not in ('PAUSED','FAILED') or not args.acknowledge_stop:
                raise ValueError('Do not duplicate/resume a model without explicit stop acknowledgement')
            command+=['--resume','--acknowledge-stop']
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,spec['gpus'])),
                         OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
        training=launch('formal_training',command,environment)
        state['status']='TRAINING';event('FORMAL_STARTED')
        launched=set()
        while training.poll() is None:
            if stopped():
                # This marker belongs to the current run and is checked at optimizer boundaries.
                run.mkdir(parents=True,exist_ok=True);(run/'STOP_REQUESTED').touch()
            for update in plan['evaluation_updates']:
                if update in launched or not (run/'checkpoints'/f'milestone_{update:06d}'/'COMPLETE.json').exists():
                    continue
                label=spec['run_id']+'_dev'+str(update)+'_seed42'
                # Complete evaluation can share authorized GPUs; canonical CPU scoring
                # is performed only on the registered reference host/environment.
                eval_command=[sys.executable,'-u','-m','tools.action_video_foresight.evaluate_milestone',
                    '--plan',args.plan,'--arm',args.arm,'--update',str(update)]
                children.append(launch(label,eval_command,environment))
                launched.add(update)
            time.sleep(10)
        event('TRAINING_EXIT',code=training.returncode)
        progress=read(run/'status.json')
        if progress['status']=='PAUSED':
            state['status']='PAUSED';event('SAVED_PAUSE',completed=progress['completed']);return
        if training.returncode or progress['status']!='COMPLETE':
            raise RuntimeError('Formal training failed; checkpoints and failed ledger remain intact')
        for update in plan['evaluation_updates']:
            if update not in launched:
                children.append(launch(spec['run_id']+'_dev'+str(update),
                    [sys.executable,'-u','-m','tools.action_video_foresight.evaluate_milestone',
                     '--plan',args.plan,'--arm',args.arm,'--update',str(update)],environment))
        codes=[p.wait() for p in children]
        if any(codes):
            raise RuntimeError('Training complete, some development evaluations failed; retain all rows')
        state['status']='COMPLETE';event('TRAINING_AND_REGISTERED_DEV_COMPLETE',updates=progress['completed'],exposure=progress['exposure'])
    except BaseException as error:
        state.update(status='FAILED',error=repr(error));event('ERROR');raise


if __name__=='__main__':
    main()
