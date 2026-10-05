"""Finite two-node Stage1 extraction -> original Stage2 imitation training.

Never signals other jobs. Smoke weights are discarded as initializers. Each phase
has separate logs/exit codes, costs, source identity and genuine completion checks.
"""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import time

from .assets import atomic_json, read


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--resume-phase', choices=['cache_smoke', 'train_smoke', 'cache_full', 'train_full'])
    a = p.parse_args(); plan = read(a.plan)
    identity = hashlib.sha256(__import__('json').dumps(
        {k:v for k,v in plan.items() if k != 'identity'}, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if plan['identity'] != identity:
        raise ValueError('Frozen ReCogDrive registration changed')
    root = Path(plan['root']); root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'CAMPAIGN.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = Path(plan['source'])
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip() != plan['source_sha']:
        raise ValueError('Campaign source changed')
    phases = ['cache_smoke', 'train_smoke', 'cache_full', 'train_full']
    start = phases.index(a.resume_phase) if a.resume_phase else 0
    costs = read(root/'costs.json') if (root/'costs.json').exists() else []
    state = dict(identity=identity, status='WAITING_MANIFEST', pid=os.getpid())
    atomic_json(root/'status.json', state)
    for phase in phases[start:]:
        remaining = plan['GPU_hours_limit'] - sum(c['GPU_hours'] for c in costs)
        if remaining <= 0 or (root/'STOP_REQUESTED').exists():
            state.update(status='PAUSED', phase=phase); atomic_json(root/'status.json', state); return
        smoke = phase.endswith('smoke')
        manifest = plan['smoke_manifest'] if smoke else plan['manifest']
        while not Path(manifest).exists():
            if (root/'STOP_REQUESTED').exists():
                state.update(status='PAUSED', phase=phase); atomic_json(root/'status.json', state); return
            time.sleep(10)
        cache = str(root/('features_smoke_v1' if smoke else 'features_full_v1'))
        is_cache = phase.startswith('cache')
        module = 'tools.recogdrive_stage2.cache_stage1' if is_cache else 'tools.recogdrive_stage2.train'
        common = ['-u', '-m', module, '--official-source', plan['official_source'],
                  '--official-revision', plan['official_revision'], '--manifest', manifest]
        if is_cache:
            common += ['--output', cache, '--max-seconds', str(remaining*3600/16)]
            if smoke:
                common += ['--limit', '512']
        else:
            output = str(root/('stage2_smoke_v1' if smoke else 'stage2_formal_v1'))
            common += ['--cache', cache, '--output', output, '--gpu-hours-limit', str(remaining)]
            if smoke:
                common += ['--smoke-steps', '2']
            elif a.resume_phase == 'train_full':
                ckpt = Path(output)/'checkpoints/paused.ckpt'
                if not ckpt.exists():
                    ckpt = Path(output)/'checkpoints/last.ckpt'
                common += ['--resume', str(ckpt)]
        processes, commands, begin = [], [], time.time()
        state.update(status='RUNNING', phase=phase, started_at=begin)
        atomic_json(root/'status.json', state)
        for node, host in enumerate(plan['hosts']):
            command = [plan['python'], '-u', '-m', 'torch.distributed.run', '--nnodes=2',
                '--nproc_per_node=8', '--node_rank='+str(node), '--master_addr='+plan['master_addr'],
                '--master_port='+str(plan['ports'][phase]), *common]
            environment = ['env', 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7', 'OMP_NUM_THREADS=2',
                'OPENBLAS_NUM_THREADS=1', 'PYTHONDONTWRITEBYTECODE=1', 'NCCL_IB_DISABLE=0',
                'NCCL_P2P_DISABLE=0', 'NCCL_SHM_DISABLE=0', 'CUDA_LAUNCH_BLOCKING=1',
                'OPENSCENE_DATA_ROOT='+plan['sensors'], 'NUPLAN_MAPS_ROOT='+plan['maps'],
                'NUPLAN_MAP_VERSION=nuplan-maps-v1.0', 'NAVSIM_EXP_ROOT='+str(root)]
            shell = 'cd '+shlex.quote(str(source))+' && '+shlex.join(environment+command)
            ssh = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,shell]
            log = root/'logs'/(phase+'_node'+str(node)+'_'+str(time.time_ns())+'.log')
            with log.open('x') as stream:
                child = subprocess.Popen(ssh, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(child); commands.append(dict(host=host, pid=child.pid, command=ssh, log=str(log)))
        atomic_json(root/(phase+'_launch.json'), dict(identity=identity, commands=commands, source=plan['source_sha']))
        codes = [child.wait() for child in processes]
        costs.append(dict(phase=phase, seconds=time.time()-begin, GPU_hours=16*(time.time()-begin)/3600,
                          exit_codes=codes, timestamp=time.time()))
        atomic_json(root/'costs.json', costs)
        if any(codes):
            state.update(status='FAILED', phase=phase, exit_codes=codes)
            atomic_json(root/'status.json', state); return
        if is_cache:
            complete = [Path(cache)/f'COMPLETE_rank{i:02d}.json' for i in range(16)]
            if not all(x.exists() for x in complete):
                state.update(status='PAUSED', phase=phase, reason='Incomplete extraction is preserved; never start formal on partial labels')
                atomic_json(root/'status.json', state); return
        else:
            trained = read(Path(output)/'status.json')
            if trained['status'] != 'COMPLETE':
                state.update(status='PAUSED', phase=phase); atomic_json(root/'status.json', state); return
        atomic_json(root/(phase+'_COMPLETE.json'), dict(identity=identity, costs=costs[-1]))
    state.update(status='COMPLETE', phase='train_full'); atomic_json(root/'status.json', state)


if __name__ == '__main__':
    main()
