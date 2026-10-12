"""One qualified node of a bounded 8/16-GPU profile; reuses benchmark worker.

This is an explicit node launcher, not a queue or a training migration tool.
Both 16-GPU nodes must be launched for the same job during a coordinated idle
window. Planning never touches GPUs. Execution keeps the physical pool lease
through all owned children, as in the existing eight-card benchmark.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.structured_world import benchmark_engineering_execution as bench


def command(manifest, job, node):
    bench.require(manifest['schema'] == 'structured_world_scaling_windows_v1', 'Unknown profile schema')
    world = job['world_size']
    bench.require(world in (8, 16) and 0 <= node < world // 8, 'One or two full8 nodes required')
    bench.require(job['mode'] in ('io_preserving_v1', 'engineering_host_fast'), 'No combined math/precision candidate')
    bench.require(job['updates'] == 220 and job['group'] in ('G1', 'G3'), 'Locked G1/G3 warm20 + stable200')
    root = Path(manifest['source_root'])
    bench.require(root.resolve() == ROOT, 'Run the launcher from the declared frozen source')
    host = manifest['hosts'][node]
    bench.require(len(host['gpu_uuids']) == len(set(host['gpu_uuids'])) == 8, 'Eight unique physical GPUs required')
    run = Path(job['run'])
    assets = manifest['assets']
    args = [manifest['python'], '-B', '-m', 'torch.distributed.run', '--nnodes='+str(world // 8),
        '--nproc-per-node=8', '--node-rank='+str(node), '--master-addr='+manifest['hosts'][0]['address'],
        '--master-port='+str(job['port']), '--rdzv-conf=timeout=120', '--max-restarts=0', '--',
        str(root/'tools/structured_world/benchmark_engineering_execution.py'), '--worker',
        '--trainer', str(root/'tools/structured_world/train_vla.py'), '--run', str(run), '--',
        '--config', str(root/'configs/structured_world'/manifest['configs'][job['group']]),
        '--scope', 'profile', '--world-scaling-profile', '--use-formal-schedule',
        '--debug-updates', '220', '--benchmark-no-end-save', '--micro-batch', str(32 // world),
        '--execution-mode', job['mode'], '--output', str(run)]
    for name in ('cache', 'geometry', 'geometry-identity', 'dino-root', 'dino-identity', 'image-root'):
        args += ['--'+name, assets[name]]
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--job', required=True)
    parser.add_argument('--node-rank', type=int, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    manifest = bench.read(args.manifest)
    job = manifest['jobs'][args.job]
    argv = command(manifest, job, args.node_rank)
    if not args.execute:
        print(json.dumps({'status': 'PLANNED_ONLY', 'command': argv, 'GPU_work_started': False}))
        return
    host = manifest['hosts'][args.node_rank]
    bench.require(socket.gethostname().removesuffix('-worker-0') == host['host'], 'Wrong node host')
    bench.immutable_source(ROOT, manifest['source_sha'])
    bench.require(bench.digest(host['qualification']['path']) == host['qualification']['sha256'], 'Host qualification changed')
    proof = bench.read(host['qualification']['path'])
    bench.require(proof['status'] == 'ENVIRONMENT_AND_OPERATOR_QUALIFIED_NOT_MODEL_ACCEPTANCE', 'Unqualified environment')
    # Check policy BEFORE importing the trainer/model, which can initialize NCCL.
    bench.authorize(host['host'])
    os.environ.update(NCCL_SOCKET_IFNAME='eth0', TORCH_NCCL_ASYNC_ERROR_HANDLING='1',
        TORCH_EXTENSIONS_DIR=host['torch_extensions'], MAX_JOBS='4')
    pool = hashlib.sha256(json.dumps(sorted(host['gpu_uuids'])).encode()).hexdigest()
    with ExitStack() as stack:
        lock = bench.acquire(stack, Path(manifest['physical_locks'])/(pool+'.lock'))
        ready = bench.readiness({'host': host['host'], 'gpu_uuids': host['gpu_uuids'], 'minimum_free_MiB': 68000})
        node_run = Path(job['run']).parent/(Path(job['run']).name+'_node'+str(args.node_rank))
        process = bench.run_job({'run': str(node_run), 'trainer': str(ROOT/'tools/structured_world/train_vla.py'),
            'gpu_uuids': host['gpu_uuids'], 'command': argv}, [lock])
        bench.write(node_run/'PROFILE_NODE_RECEIPT.json', {'manifest_sha256': bench.digest(args.manifest),
            'readiness': ready, 'process': process, 'world': job['world_size'], 'node_rank': args.node_rank,
            'node_process_GPU_hours': process['wall_seconds']*8/3600, 'formal_authorized': False})
        bench.require(process['returncode'] == 0, 'Owned profile node failed; preserve original logs')


if __name__ == '__main__':
    main()
