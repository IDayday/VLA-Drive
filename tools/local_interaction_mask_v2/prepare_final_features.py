"""Bounded two-host preparation after a completed foundation; no model/test-score selection."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from starVLA.model.modules.joint_world.public_baseline import sha256
from tools.local_interaction_mask_v2.train_foundation import atomic_json
from tools.local_interaction_mask_v2.budget import total_hours


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--role', choices=['train', 'evaluation'], required=True)
    parser.add_argument('--wait-hours', type=float, default=4)
    args = parser.parse_args(); spec = json.loads(Path(args.spec).read_text())
    root = Path(spec['output']); root.mkdir(parents=True, exist_ok=True)
    monitor = root/('pipeline_'+args.role); monitor.mkdir(exist_ok=True)
    deadline = time.monotonic()+args.wait_hours*3600
    def wait_for(predicate, description):
        while not predicate():
            if time.monotonic() > deadline: raise TimeoutError(description)
            atomic_json(monitor/'status.json', {'status': 'waiting', 'for': description})
            time.sleep(10)
    def foundation_finished():
        ledger = json.loads(Path(spec['ledger']).read_text())
        run = next(row for row in ledger['runs'] if row['id'] == spec['foundation_run_id'])
        if run['status'] in ('failed', 'paused'): raise RuntimeError('Foundation did not complete; keep artifacts and review')
        return run['status'] == 'complete'
    def idle_devices(count):
        devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
        if len(devices) != count or any(not item.isdigit() for item in devices):
            raise ValueError('Set exactly the authorized numeric CUDA devices for this host')
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
        uuids = {row.split(',')[1].strip() for row in output.splitlines() if row.split(',')[0].strip() in devices}
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        if any(row.split(',')[0].strip() in uuids for row in processes.splitlines()):
            raise RuntimeError('Assigned device has an existing process; no process is stopped automatically')
    def run_stage(name, module, arguments, hours, gpu_count=8):
        output = root/name; command = [sys.executable]
        if gpu_count > 1: command += ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node='+str(gpu_count)]
        command += ['-m', 'tools.local_interaction_mask_v2.'+module, *map(str, arguments), '--output', str(output)]
        run_id = spec['run_prefix']+'_'+name
        ledger = json.loads(Path(spec['ledger']).read_text())
        previous = next((row for row in ledger['runs'] if row['id'] == run_id), None)
        if previous:
            source_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
            if previous['status'] != 'complete' or previous['metadata']['command'] != command or previous['metadata']['code_sha'] != source_sha:
                raise RuntimeError('Existing stage requires explicit recovery: '+name)
            if json.loads((output/'status.json').read_text())['status'] != 'complete':
                raise RuntimeError('Completed ledger disagrees with artifact')
            return
        if total_hours(ledger)+hours+spec['reserve_after_stage_gpu_hours'] > ledger['gpu_hour_cap']:
            raise RuntimeError('Stage would consume the reserved main experiment/evaluation budget')
        visible = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(visible[:gpu_count]))
        prior = os.environ['CUDA_VISIBLE_DEVICES']; os.environ['CUDA_VISIBLE_DEVICES'] = environment['CUDA_VISIBLE_DEVICES']
        try: idle_devices(gpu_count)
        finally: os.environ['CUDA_VISIBLE_DEVICES'] = prior
        atomic_json(monitor/'status.json', {'status': 'running', 'stage': name})
        subprocess.run([sys.executable, '-m', 'tools.local_interaction_mask_v2.supervise', '--ledger', spec['ledger'],
            '--run-id', run_id, '--output', str(output), '--gpus', str(gpu_count), '--max-gpu-hours', str(hours),
            '--', *command], env=environment, check=True)
        if json.loads((output/'status.json').read_text())['status'] != 'complete':
            raise RuntimeError('Stage paused before completion: '+name)
    try:
        wait_for(foundation_finished, 'foundation supervisor completion')
        foundation = root/'foundation'; frozen = foundation/'checkpoint.pt'
        if args.role == 'train':
            status = json.loads((Path(spec['foundation_run'])/'status.json').read_text())
            if status['status'] != 'complete' or status['epochs_completed'] != spec['foundation_epochs']:
                raise ValueError('Foundation did not reach the registered freeze epoch')
            if not (foundation/'manifest.json').exists():
                foundation.mkdir(exist_ok=False)
                source = Path(spec['foundation_run'])/'checkpoint.pt'
                shutil.copyfile(source, frozen.with_suffix('.tmp')); frozen.with_suffix('.tmp').replace(frozen)
                digest = sha256(frozen)
                if digest != sha256(source): raise ValueError('Frozen foundation copy differs')
                atomic_json(foundation/'manifest.json', dict(source=str(source), sha256=digest, training_status=status,
                    stage='shared final foundation, frozen before graph comparisons and Navtest', Navtest_consulted=False))
        wait_for(lambda: (foundation/'manifest.json').exists(), 'immutable shared foundation')
        if sha256(frozen) != json.loads((foundation/'manifest.json').read_text())['sha256']:
            raise ValueError('Frozen foundation hash changed')
        splits = ['train', 'holdout'] if args.role == 'train' else ['navtest', 'dev']
        for split in splits:
            extra = ['--check-native-prefix'] if split == 'holdout' else []
            run_stage(split+'_current', 'extract_current', ['--checkpoint', frozen, '--public-qwen', spec['public_qwen'],
                '--visual-cache', spec['visual_cache'], '--language-adapter-precision', 'fp32',
                '--dataset', spec['datasets'][split], '--selector', spec['selector'], '--graph-config', spec['graph_config'], *extra],
                spec['extraction_caps'][split])
        if args.role == 'train':
            run_stage('head', 'train_current_heads', ['--foundation', frozen, '--cache', root/'train_current',
                '--targets', spec['train_targets'], '--holdout-cache', root/'holdout_current',
                '--holdout-targets', spec['holdout_targets'], '--epochs', '16', '--schedule-epochs', '16',
                '--batch', '64', '--lr', '0.001', '--workers', '2'], .8, 1)
        def head_finished():
            pipeline = root/'pipeline_train/status.json'
            if pipeline.exists() and json.loads(pipeline.read_text())['status'] == 'failed':
                raise RuntimeError('Training-side feature pipeline failed')
            path = root/'head/status.json'
            if not path.exists(): return False
            status = json.loads(path.read_text())['status']
            if status != 'complete': raise RuntimeError('Shared current head did not complete')
            # Wait until its supervisor releases the GPU and closes its budget record.
            ledger = json.loads(Path(spec['ledger']).read_text())
            return any(row['id'] == spec['run_prefix']+'_head' and row['status'] == 'complete' for row in ledger['runs'])
        wait_for(head_finished, 'completed shared current head')
        for split in splits:
            run_stage(split+'_refined', 'refresh_current_heads', ['--cache', root/(split+'_current'),
                '--perception-checkpoint', root/'head/selected.pt', '--dataset', spec['datasets'][split]],
                spec['refresh_caps'][split])
        atomic_json(monitor/'status.json', {'status': 'complete', 'splits': splits, 'planning_scores_consulted': False})
    except Exception as error:
        atomic_json(monitor/'status.json', {'status': 'failed', 'error': repr(error)})
        raise


if __name__ == '__main__': main()
