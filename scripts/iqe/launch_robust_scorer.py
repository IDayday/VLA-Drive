#!/usr/bin/env python3
"""One-shot, bounded head-training dispatch on a verified permanent host."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from iqe.robust_scorer.config import load, CELLS
from iqe.io import atomic_json, file_hash, read_json
from iqe.contracts import require


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--cells', nargs='+', choices=CELLS, required=True)
    p.add_argument('--gpus', nargs='+', type=int, required=True)
    p.add_argument('--expect-host', required=True)
    p.add_argument('--share-with-fixed-anchor-cache', action='store_true')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    require(len(a.cells) == len(a.gpus) and len(set(a.cells)) == len(a.cells)
            and len(set(a.gpus)) == len(a.gpus), 'one distinct GPU for each distinct cell')
    policy_path = Path('/mnt/project/server_dispatch_policy.json')
    policy = read_json(policy_path)
    host = socket.gethostname().removesuffix('-worker-0')
    require(host == a.expect_host and host in policy['allowed_hosts'], 'unverified/permanent-disallowed host')
    require(host != 'training-vla-zt2', 'protected Base/G2 host excluded from this launcher')
    cfg = load(a.config)
    require((cfg.root/'EXPERIMENT_BINDING.json').is_file(), 'register experiment before dispatch')
    require((cfg.root/'REAL_CONTEXT_BACKWARD_QUALIFICATION.json').is_file(), 'real cached-context qualification missing')
    folder = cfg.root/'runtime'
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder/'LAUNCH.lock', 'a') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
                                      '--format=csv,noheader,nounits'], text=True)
        inventory = {}
        for line in raw.splitlines():
            index, uuid, free, util = map(str.strip, line.split(','))
            inventory[int(index)] = dict(uuid=uuid, free_mib=int(free), utilization=int(util))
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                                       '--format=csv,noheader,nounits'], text=True)
        owners = {}
        for line in apps.splitlines():
            pid, uuid, memory = map(str.strip, line.split(','))
            proc = Path('/proc')/pid
            try:
                command = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode()
            except OSError:
                command = None
            owners.setdefault(uuid, []).append(dict(pid=int(pid), cmdline=command, memory_mib=memory))
        # Complete preflight for every cell before creating any new child.
        pending = []
        for cell, gpu in zip(a.cells, a.gpus):
            require(gpu in inventory and inventory[gpu]['free_mib'] >= 16384, 'insufficient GPU headroom')
            existing = owners.get(inventory[gpu]['uuid'], [])
            if existing:
                require(a.share_with_fixed_anchor_cache, 'occupied GPU requires explicit qualified sharing')
                require(all(x['cmdline'] and '/tools/decision_local/build_fixed_anchor_cache.py ' in x['cmdline']
                            for x in existing), 'unknown or protected GPU owner; no dispatch')
            else:
                require(inventory[gpu]['utilization'] < 20, 'GPU active without visible owner')
            target = cfg.root/'scorers'/cell
            result = read_json(target/'result.json') if (target/'result.json').is_file() else {}
            if result.get('status') == 'COMPLETE':
                continue
            target.mkdir(parents=True, exist_ok=True)
            with open(target/'RUN.lock', 'a') as run_lock:
                fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            receipt = folder/(cell+'_LAUNCH.json')
            if receipt.exists():
                old = read_json(receipt)
                proc = Path('/proc')/str(old['pid'])
                if old['host'] == host and proc.exists():
                    ticks = (proc/'stat').read_text().rsplit(')', 1)[1].split()[19]
                    require(ticks != old['start_ticks'], 'same cell process is already active')
            require(a.resume or not (target/'RUN_IDENTITY.json').exists(), 'existing run requires --resume')
            pending.append((cell, gpu, existing))
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        for cell, gpu, existing in pending:
            cmd = [sys.executable, '-u', '-m', 'iqe.robust_scorer.cli', 'train', '--config', str(Path(a.config).resolve()),
                   '--cell', cell, '--device', 'cuda:0']
            if a.resume:
                cmd += ['--resume', 'auto']
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            logfile = folder/(cell+'_'+stamp+'.log')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       PYTHONUNBUFFERED='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', IQE_ROBUST_EXPERIMENT=cfg.experiment_id)
            with open(logfile, 'x') as stream:
                child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                         stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            proc = Path('/proc')/str(child.pid)
            info = dict(host=host, pid=child.pid, pgid=os.getpgid(child.pid),
                        start_ticks=(proc/'stat').read_text().rsplit(')', 1)[1].split()[19],
                        launched_utc=stamp, cell=cell, gpu=gpu, command=cmd, cwd=str(ROOT),
                        code_commit=commit, config_hash=cfg.identity, policy_hash=file_hash(policy_path),
                        preexisting_owners=existing, gpu_preflight=inventory[gpu], log=str(logfile),
                        max_optimizer_updates=cfg.updates, temporary_host=False, signals_sent_to_other_tasks=False)
            atomic_json(folder/(cell+'_LAUNCH_'+stamp+'.json'), info, immutable=True)
            atomic_json(folder/(cell+'_LAUNCH.json'), info)
            print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
