"""Finite, leased ReCogDrive export followed by unchanged official PDMS replay."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .assets import atomic_json, digest, read
from .evaluate import signature


def gpu_group(plan):
    children = []
    begin = time.time()
    try:
        for rank, uuid in enumerate(plan['gpu_uuids']):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, LOCAL_RANK='0',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            cmd = [sys.executable, '-u', '-m', 'tools.recogdrive_stage2.evaluate', 'export']
            for key in ('official_source', 'official_revision', 'training_run', 'checkpoint', 'manifest'):
                cmd += ['--' + key.replace('_', '-'), str(plan[key])]
            cmd += ['--output', str(Path(plan['output']) / 'bank'), '--sampling-seed', str(plan['sampling_seed']),
                    '--rank', str(rank), '--world-size', str(len(plan['gpu_uuids'])),
                    '--limit', str(plan['limit']), '--max-seconds', str(plan['max_seconds'])]
            log = open(Path(plan['output']) / f'export_rank{rank}.log', 'a')
            child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            children.append((child, log))
        codes = [child.wait() for child, _ in children]
        if any(codes):
            raise RuntimeError('Inference rank failed: ' + repr(codes))
    finally:
        for child, log in children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
            log.close()
        atomic_json(Path(plan['output']) / 'gpu_cost.json', dict(gpu_hours=len(children)*(time.time()-begin)/3600,
            optimizer_updates=0, gpu_uuids=plan['gpu_uuids']))


def run(args):
    plan = read(args.plan)
    source = Path(__file__).resolve().parents[2]
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    if sha != plan['evaluation_source'] or subprocess.check_output(['git', 'status', '--porcelain'], cwd=source, text=True).strip():
        raise ValueError('Evaluation source must be clean and frozen')
    root = Path(plan['output']); root.mkdir(parents=True, exist_ok=True)
    bank = root / 'bank'; bank.mkdir(exist_ok=True)
    if (root / 'registration.json').exists() and read(root / 'registration.json') != plan:
        raise ValueError('Evaluation registration changed')
    atomic_json(root / 'registration.json', plan)
    manifest = read(plan['manifest'])
    identity = dict(schema='official_recogdrive_selected_trajectory_v1',
        model_class='navsim.agents.recogdrive.recogdrive_agent.ReCogDriveAgent',
        training_identity=read(Path(plan['training_run']) / 'identity.json'),
        checkpoint_sha256=digest(plan['checkpoint']), checkpoint=plan['checkpoint'],
        manifest_sha256=digest(plan['manifest']), evaluation_source=sha,
        world_size=len(plan['gpu_uuids']), sampling_seed=plan['sampling_seed'], limit=plan['limit'],
        precision=dict(stage1_weights='BF16', stage1_compute='BF16 official backbone',
                       stage1_hidden_interface='FP32 full sequence', stage2_master='FP32', stage2_compute='FP32', tf32=False),
        protocol=dict(sampler='unchanged official DDIM', steps=5, candidates=1,
                      deterministic=False, learned_scorer=None, future_conditioning=False,
                      noise='new registered SHA256(recogdrive:seed:scene), independent of GPU layout',
                      postprocess='unchanged official clamp and denorm_odo'),
        diagnostic=bool(plan['limit']), scenes=min(plan['limit'], len(manifest['rows'])) if plan['limit'] else len(manifest['rows']))
    if (bank / 'identity.json').exists() and read(bank / 'identity.json') != identity:
        raise ValueError('Checkpoint/export identity changed')
    atomic_json(bank / 'identity.json', identity)
    rows = manifest['rows'][:plan['limit']] if plan['limit'] else manifest['rows']
    metric = {r['token']: r for r in read(plan['metric_index'])}
    index = [metric[r['token']] for r in rows]
    atomic_json(root / 'requested_index.json', index)
    begin = time.time()
    def status(state, phase, **details):
        atomic_json(root / 'status.json', dict(status=state, phase=phase, pid=os.getpid(),
            requested_scenes=len(rows), updated_unix=time.time(), **details))
    try:
        complete = all((bank / f'shard_{i}.json').exists() and
            read(bank / f'shard_{i}.json')['status'] == 'complete' for i in range(len(plan['gpu_uuids'])))
        if not complete:
            status('RUNNING', 'GPU_EXPORT')
            wrapper = [sys.executable, str(Path(plan['skill_snapshot']) / 'with_gpu_lease.py'),
                '--gpus', ','.join(map(str, plan['physical_gpus'])), '--lease-file', plan['host_lease'], '--cwd', str(source)]
            for uuid in plan['gpu_uuids']:
                wrapper += ['--expect-uuid', uuid]
            wrapper += ['--', sys.executable, '-u', '-m', 'tools.recogdrive_stage2.run_evaluation',
                        '--plan', args.plan, '--gpu-group']
            with (root / 'lease.log').open('a') as log:
                code = subprocess.call(wrapper, stdout=log, stderr=subprocess.STDOUT)
            if code == 75:
                status('PAUSED', 'GPU_UNAVAILABLE'); return
            if code:
                raise RuntimeError('Leased export failed: ' + str(code))
        for i in range(len(plan['gpu_uuids'])):
            shard = read(bank / f'shard_{i}.json')
            if shard['identity_sha256'] != signature(identity) or shard['completed'] != len(rows[i::len(plan['gpu_uuids'])]):
                raise ValueError('Incomplete or foreign GPU partition')
            if not read(bank / f'inference_parity_rank{i}.json')['passed']:
                raise ValueError('Original agent inference parity absent')
        helper = Path(plan['skill_snapshot']) / 'score_official.py'
        common = ['--index', str(root / 'requested_index.json'), '--predictions', str(bank), '--devkit', plan['devkit']]
        status('RUNNING', 'OFFICIAL_CPU_SCORING')
        cmd = [sys.executable, str(helper), 'score', *common, '--output', str(root / 'scores'), '--workers', str(plan['cpu_workers'])]
        if not plan['limit']:
            cmd += ['--full-navtest']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        with (root / 'score.log').open('a') as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        # Independently replay this exact bank through the unmodified official
        # submission evaluator; never use the parallel worker as its reference.
        status('RUNNING', 'OFFICIAL_REFERENCE_REPLAY')
        with (root / 'reference.log').open('a') as log:
            subprocess.run([sys.executable, str(helper), 'reference', *common,
                '--output', str(root / 'official_reference')], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        status('RUNNING', 'TOKEN_FACTOR_PARITY')
        with (root / 'parity.log').open('a') as log:
            subprocess.run([sys.executable, str(Path(plan['skill_snapshot']) / 'validate_scores.py'),
                '--candidate', str(root / 'scores/scenes.csv'), '--reference', str(root / 'official_reference/official.csv'),
                '--candidate-identity', str(root / 'scores/identity.json'),
                '--reference-identity', str(root / 'official_reference/identity.json'),
                '--index', str(root / 'requested_index.json'), '--expected-scenes', str(len(rows)),
                '--expected-logs', str(len({r['log'] for r in rows})), '--tolerance', '1e-8',
                '--output', str(root / 'official_parity.json')], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        summary = read(root / 'scores/summary.json')
        atomic_json(root / 'SUMMARY.json', dict(summary=summary, checkpoint=identity,
            inference_parity=[read(bank / f'inference_parity_rank{i}.json') for i in range(len(plan['gpu_uuids']))],
            official_parity=read(root / 'official_parity.json'), elapsed_seconds=time.time()-begin,
            diagnostic=bool(plan['limit']), gpu_cost=read(root / 'gpu_cost.json')))
        status('COMPLETE', 'COMPLETE', elapsed_seconds=time.time()-begin)
    except Exception as error:
        status('FAILED', 'ERROR', error=repr(error)); raise


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--gpu-group', action='store_true')
    args = p.parse_args()
    if args.gpu_group:
        gpu_group(read(args.plan))
    else:
        run(args)


if __name__ == '__main__':
    main()
