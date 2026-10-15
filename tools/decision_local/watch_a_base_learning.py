"""Bounded observer for the four registered A-internal checkpoints only.

This starts the existing native inference entry point, never a trainer, scorer,
or checkpoint selector. It uses currently available local GPUs and preserves
all other processes. Native master reconstruction is limited to two loaders at
once; inference continues concurrently. Failed workers are not auto-retried.
"""
import argparse
import fcntl
import importlib.util
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('_curve_watch_helpers', Path(__file__).with_name('common.py'))
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def gpu_snapshot():
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,memory.used',
        '--format=csv,noheader,nounits'], text=True, timeout=15)
    owners = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True, timeout=15)
    visible = {}
    for line in owners.splitlines():
        uuid,pid = [x.strip() for x in line.split(',')]
        visible.setdefault(uuid, []).append((Path('/proc')/pid/'cmdline').exists())
    result = {}
    for line in text.splitlines():
        index,uuid,free,used = [x.strip() for x in line.split(',')]
        # Never classify hidden occupants as idle or as a releasable filler.
        if int(used) > 256 and (not visible.get(uuid) or not all(visible[uuid])):
            continue
        result[uuid] = {'index':int(index),'free_MiB':int(free),'used_MiB':int(used)}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('registration', 'population-contract', 'training-run', 'current-root',
                 'legacy-source', 'profile-receipt', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--max-wait-seconds', type=float, required=True)
    p.add_argument('--gpu-hour-cap', type=float, required=True)
    p.add_argument('--worker-max-seconds', type=float, default=1200)
    a = p.parse_args()
    if min(a.max_wait_seconds, a.gpu_hour_cap, a.worker_max_seconds) <= 0:
        raise ValueError('Finite positive budgets are required')
    h.check_host()
    registration = h.read(a.registration)
    identity = h.read(a.training_run/'identity.json')
    profile = h.read(a.profile_receipt)
    milestones = registration['A_internal_learning_checkpoints']
    if (identity['scope'] != 'formal' or identity['source_sha'] != registration['training_source_sha'] or
            identity['registration_sha256'] != h.digest(a.registration) or
            identity['config']['natural_error_provenance']['population_contract_sha256'] != h.digest(a.population_contract) or
            milestones != registration['milestones'] or milestones[-1] != registration['updates']):
        raise ValueError('Only the registered common A-only training curve is allowed')
    if not profile.get('rows') or profile['peak_allocated_bytes'] <= 0:
        raise ValueError('Actual native FP32 inference memory measurement is required')
    required_MiB = math.ceil((profile['peak_allocated_bytes']+4*2**30)/2**20)
    a.output.mkdir(parents=True, exist_ok=True)
    lock = (a.output/'observer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (a.output/'STATE.json').exists():
        raise FileExistsError('Inspect the existing observer; no implicit restart or retry')
    started = time.monotonic()
    state = dict(schema='fixed_A_internal_curve_observer_v1', pid=os.getpid(),
        source_sha=h.source_sha(ROOT), training_identity=identity['identity'],
        registration_sha256=h.digest(a.registration), milestones=milestones, completed=[],
        GPU_process_seconds=0., GPU_hour_cap=a.gpu_hour_cap, minimum_free_MiB=required_MiB,
        resources=h.check_host(), selection='Fixed terminal only; no B/dev/Navtest selection')
    handles = []

    def write(status):
        state.update(status=status, updated_unix=time.time(), wall_seconds=time.monotonic()-started)
        h.atomic_json(a.output/'STATE.json', state)

    def stop_owned():
        # Each handle is an exact child started in its own process group here.
        # There are no remote signals or matches against other Python jobs.
        for item in handles:
            if item['process'].poll() is None:
                try:
                    os.killpg(item['process'].pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for item in handles:
            try:
                item['process'].wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(item['process'].pid, signal.SIGKILL)
                item['process'].wait()
            item['stream'].close()

    try:
        for update in milestones:
            tag = f'milestone_{update:06d}'
            state['next_tag'] = tag
            while not (a.training_run/'checkpoints'/tag/'COMPLETE.json').exists():
                h.check_host()
                if (a.output/'STOP_REQUESTED').exists():
                    raise RuntimeError('Observer stop requested')
                if time.monotonic()-started >= a.max_wait_seconds:
                    raise TimeoutError('Finite observer time exhausted')
                status = h.read(a.training_run/'status.json')
                if status['status'] != 'RUNNING':
                    raise RuntimeError('Expected checkpoint is absent; base status='+status['status'])
                write('WAITING_FOR_NATIVE_CHECKPOINT')
                time.sleep(30)
            complete = h.read(a.training_run/'checkpoints'/tag/'COMPLETE.json')
            if complete['identity'] != identity['identity'] or complete['completed'] != update:
                raise ValueError('Foreign or incomplete native milestone')
            while True:
                h.check_host()
                if time.monotonic()-started >= a.max_wait_seconds:
                    raise TimeoutError('Finite resource wait exhausted')
                first = gpu_snapshot()
                time.sleep(2)
                second = gpu_snapshot()
                devices = [dict(uuid=uuid, **row) for uuid,row in second.items()
                    if uuid in first and min(row['free_MiB'],first[uuid]['free_MiB']) >= required_MiB]
                if devices:
                    break
                write('WAITING_FOR_LOCAL_GPU_MEMORY')
                time.sleep(30)
            output = a.output/tag
            output.mkdir(exist_ok=False)
            remaining = a.gpu_hour_cap*3600-state['GPU_process_seconds']
            worker_limit = min(a.worker_max_seconds, remaining/len(devices))
            if worker_limit < 600:
                raise TimeoutError('Insufficient measured-load budget for another complete checkpoint')
            state.update(active_tag=tag, devices=devices, workers=[])
            previous_GPU_seconds = state['GPU_process_seconds']
            handles = []
            pending = list(enumerate(devices))
            tick = time.monotonic()
            while pending or any(x['process'].poll() is None for x in handles):
                now = time.monotonic()
                state['GPU_process_seconds'] += (now-tick)*sum(x['process'].poll() is None for x in handles)
                tick = now
                h.check_host()
                if (a.output/'STOP_REQUESTED').exists():
                    raise RuntimeError('Observer stop requested')
                if state['GPU_process_seconds'] >= a.gpu_hour_cap*3600:
                    raise TimeoutError('Finite curve GPU budget reached')
                for item in handles:
                    result = item['process'].poll()
                    if result not in (None, 0):
                        raise RuntimeError('Native curve worker failed; preserve attempt: '+str(item['rank']))
                    if result is None and now-item['started'] > worker_limit:
                        raise TimeoutError('Native read-only curve worker exceeded wall budget')
                loading = sum(x['process'].poll() is None and not (output/f"ready_{x['rank']:03d}.json").exists()
                              for x in handles)
                if pending and loading < 2:
                    rank,device = pending.pop(0)
                    fresh = gpu_snapshot()
                    if fresh.get(device['uuid'],{}).get('free_MiB',0) < required_MiB:
                        raise RuntimeError('GPU memory changed before launch; no blind retry')
                    command = [sys.executable, str(ROOT/'tools/decision_local/measure_scoring_process.py'),
                        '--output', str(output/f'CPU_rank_{rank:03d}.json'), '--', sys.executable,
                        str(ROOT/'tools/decision_local/evaluate_a_base_learning.py'),
                        '--legacy-source',str(a.legacy_source),'--population-contract',str(a.population_contract),
                        '--current-root',str(a.current_root),'--output',str(output),
                        '--training-run',str(a.training_run),'--tag',tag,'--rank',str(rank),
                        '--shards',str(len(devices)),'--max-seconds',str(worker_limit)]
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device['index']), OMP_NUM_THREADS='2',
                               MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
                    stream = (output/f'rank_{rank:03d}.log').open('x')
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    handles.append(dict(process=process,stream=stream,started=now,rank=rank))
                    state['workers'].append(dict(rank=rank,pid=process.pid,device=device,command=command))
                write('INFERRING_A_INTERNAL_ONLY')
                time.sleep(2)
            for item in handles:
                item['process'].wait()
                item['stream'].close()
            meters = [h.read(output/f'CPU_rank_{rank:03d}.json') for rank in range(len(devices))]
            if any(meter['exit_code'] != 0 for meter in meters):
                raise ValueError('A native curve child did not finish successfully')
            # Replace live poll estimates with exact waited-child wall costs.
            state['GPU_process_seconds'] = previous_GPU_seconds+sum(m['wall_seconds'] for m in meters)
            state['CPU_process_seconds'] = state.get('CPU_process_seconds',0.)+sum(
                m['user_seconds']+m['system_seconds'] for m in meters)
            summary = h.read(output/'COMPLETE.json')
            if not summary['full_population'] or summary['engineering_only'] or summary['preflight']:
                raise ValueError('Partial inference cannot be a complete A-internal curve point')
            state['completed'].append(dict(tag=tag,summary=summary,output=str(output)))
            write('CHECKPOINT_EVALUATED')
        write('COMPLETE')
    except BaseException as error:
        stop_owned()
        state['error'] = str(error)
        write('STOPPED_WITH_REMAINING_CHECKPOINTS')
        raise


if __name__ == '__main__':
    main()
