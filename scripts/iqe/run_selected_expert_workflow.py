"""Resource-only eight-GPU orchestration of the existing IQE stage commands.

Keeps all model/data/loss/sampler settings and leaves existing jobs untouched.
Training begins only after the fixed expert-data receipt is complete and any
explicit pre-training data revision hold has been cleared.
"""
from pathlib import Path
from datetime import datetime, timezone
import fcntl
import os
import subprocess
import sys
import time

import argparse
parser = argparse.ArgumentParser(description='Continue the selected IQE expert lineage, then its scratch Scorer; preserve existing jobs.')
parser.add_argument('--config', required=True)
parser.add_argument('--host', default='training-vlawm-tsinghua-zt3', choices=['training-vlawm-tsinghua-zt3'])
args = parser.parse_args()
REPO = Path(__file__).resolve().parents[2]
CONFIG = Path(args.config).resolve()
PYTHON = sys.executable
HOST = args.host
sys.path.insert(0, str(REPO))
from iqe.io import atomic_json, read_json, file_hash, digest
from iqe.internal_baseline.expert_data_revision import artifact as expert_artifact, training_root, workflow_identity_path
from iqe.internal_baseline.config import load_config
ROOT = load_config(CONFIG).root


def now():
    return datetime.now(timezone.utc).isoformat()


def policy_check():
    policy = read_json('/mnt/project/server_dispatch_policy.json')
    assert os.uname().nodename.removesuffix('-worker-0') == HOST
    assert HOST in policy['task_authorizations']['iqe_v1']['allowed_hosts']
    assert HOST in policy.get('latest_usage_scope', {}).get('allowed_hosts', [HOST])
    assert policy.get('global_dispatch_prohibitions', {}).get(HOST, {}).get('new_task_dispatch_allowed', True)
    auth = read_json(ROOT/'runtime/SHARED_RESOURCE_AUTHORIZATION.json')
    assert auth['host'] == HOST and auth['shared_GPU_deployment_authorized']


def resource_snapshot(gpus, minimum_free_mib):
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.free,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True)
    rows = {int(fields[0]): {'used_MiB': int(fields[1]), 'free_MiB': int(fields[2]), 'utilization_percent': int(fields[3])}
            for fields in (line.split(',') for line in raw.splitlines())}
    assert all(rows[i]['free_MiB'] >= minimum_free_mib for i in gpus), ('BLOCKED_MEMORY_HEADROOM', rows)
    return {'utc': now(), 'gpus': rows, 'selected': gpus, 'minimum_free_MiB': minimum_free_mib,
            'shared_execution_explicitly_authorized': True, 'idle_GPU_not_required': True}


def main():
    policy_check()
    with open(ROOT/'runtime/HEAD_QUEUE.lock', 'a+') as queue_guard:
        fcntl.flock(queue_guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.time()
        while True:
            policy_check()
            hold_path = ROOT/'runtime/HOLD_HEAD_TRAINING.json'
            hold = read_json(hold_path) if hold_path.exists() else {'active': False}
            preparation = read_json(ROOT/'runtime/MANIFEST_PREPARATION_STATUS.json')
            assert preparation['status'] != 'FAILED', preparation
            if not hold.get('active') and preparation['status'] == 'COMPLETE':
                break
            atomic_json(ROOT/'runtime/HEAD_QUEUE_STATUS.json', {
                'status': 'WAITING_EXPERT_DATA_DECISION' if hold.get('active') else 'WAITING_MANIFEST_PREPARATION',
                'pid': os.getpid(), 'host': HOST, 'updated_utc': now(), 'not_training_yet': True,
                'GPUs': list(range(8)), 'hold': hold, 'preparation': preparation,
                'wait_for_idle_GPU': False})
            assert time.time()-started < 43200, 'bounded preparation wait expired'
            time.sleep(15)
        with open(ROOT/'RUN.lock', 'a+') as run_guard:
            fcntl.flock(run_guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            from iqe.internal_baseline.config import load_config
            from iqe.internal_baseline.budget import training_budget
            config = load_config(CONFIG)
            binding = read_json(ROOT/'BASE_BINDING.json')
            assert binding['config_hash'] == config.identity
            manifest = read_json(expert_artifact(config, "EXPERT_MANIFEST.json"))
            assert manifest['hash'] == digest({k:v for k,v in manifest.items() if k != 'hash'})
            assert manifest['base_binding_hash'] == file_hash(ROOT/'BASE_BINDING.json')
            assert manifest['partition_hash'] == file_hash(ROOT/'DATA_PARTITION.json')
            assert manifest['hard'] and manifest['ordinary'] and manifest['sampling'] == {'hard': .8, 'ordinary': .2}
            budget = training_budget(config)
            identity = {'commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                'config_hash': config.identity, 'config_file_hash': file_hash(CONFIG),
                'base_binding_hash': file_hash(ROOT/'BASE_BINDING.json'),
                'expert_manifest_hash': file_hash(expert_artifact(config, "EXPERT_MANIFEST.json")),
                'expert_training_root': str(training_root(config)),
                'expert_data_transition_hash': file_hash(training_root(config)/'DATA_TRANSITION.json'),
                'scorer_scope_hash': file_hash(ROOT/'SCORER_FIT_SCOPE.json'),
                'source_hashes': {name: file_hash(REPO/'iqe/internal_baseline'/name)
                                  for name in ('training.py','reporting.py','scorer_scope.py','sampling.py','objectives.py','expert_data_revision.py','expert_data_transition.py')},
                'effective_training_budget': budget, 'world_size': 8, 'global_batch': 128,
                'planned_training_interruptions': False, 'phase_switch_is_internal': True,
                'checkpoint_source_hash': file_hash(REPO/'iqe/training/checkpoint.py'),
                'trainer_source_hash': file_hash(REPO/'iqe/training/trainer.py'),
                'orchestrator_sha256': file_hash(__file__), 'resource_authorization_hash': file_hash(ROOT/'runtime/SHARED_RESOURCE_AUTHORIZATION.json')}
            atomic_json(workflow_identity_path(config), identity, immutable=True)
            live = {'pid': os.getpid(), 'host': HOST, 'started_utc': now(), 'experiment': config.experiment_id,
                    'status': 'RUNNING', 'GPUs': list(range(8)), 'world_size': 8, 'global_batch': 128,
                    'effective_training_budget': budget, 'scope': 'one existing expert, scratch scorer, fixed dev; no navtest/ACTIVE'}
            atomic_json(ROOT/'WORKFLOW_LIVE.json', live)
            atomic_json(ROOT/'workflow_stages/build_hard.json', {'status': 'COMPLETE', 'reused': True,
                        'manifest_hash': identity['expert_manifest_hash'], 'preparation': preparation})
            stages = [('expert_finish', 'train-expert', 'expert'), ('pair_bank', 'build-pair-bank', None),
                      ('scorer_finish', 'train-scorer', 'scorer'), ('evaluate', 'evaluate', None),
                      ('diagnostics', 'diagnostics', None), ('export_diagnostic_bundle', 'export-bundle', None),
                      ('audit_actual_training', None, None), ('audit_current_input_labels', 'audit-online', None),
                      ('benchmark_online_bundle', 'benchmark', None)]
            for name, command, trainer in stages:
                policy_check()
                assert file_hash(CONFIG) == identity['config_file_hash'], 'configuration changed during workflow'
                assert all(file_hash(REPO/'iqe/internal_baseline'/k) == v for k,v in identity['source_hashes'].items()), 'imported source changed'
                assert file_hash(REPO/'iqe/training/checkpoint.py') == identity['checkpoint_source_hash']
                assert file_hash(REPO/'iqe/training/trainer.py') == identity['trainer_source_hash']
                gpus = list(range(8)) if trainer else [0]
                snapshot = resource_snapshot(gpus, 8192 if name not in {'audit_current_input_labels','benchmark_online_bundle'} else 24576)
                env = dict(os.environ, PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),
                           OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1',
                           CUBLAS_WORKSPACE_CONFIG=':4096:8')
                for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):
                    env.pop(key, None)
                argv = [PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
                        '-m', 'iqe.internal_baseline.cli'] if trainer else [PYTHON, '-m', 'iqe.internal_baseline.cli']
                if command is not None:
                    argv += [command, '--config', str(CONFIG), '--device', 'cuda:0']
                    if trainer and list(training_root(config,trainer).glob('step_*.pt.COMPLETE.json')):
                        argv += ['--resume', 'auto']
                else:
                    argv = [PYTHON, '-c', 'import json; from iqe.internal_baseline.config import load_config; '
                            'from iqe.internal_baseline.audit_training import audit; '
                            f'print(json.dumps(audit(load_config({str(CONFIG)!r}),"cuda:0")))']
                stage = {'stage': name, 'status': 'RUNNING', 'started_utc': now(), 'command': argv, 'resources': snapshot}
                atomic_json(ROOT/'WORKFLOW_STAGE.json', stage)
                t = time.time()
                log = training_root(config)/'workflow_logs'/f'{name}.log'
                log.parent.mkdir(parents=True, exist_ok=True)
                with open(log, 'a') as output:
                    child = subprocess.Popen(argv, cwd=REPO, env=env, stdout=output, stderr=subprocess.STDOUT)
                    stage.update(pid=child.pid, log=str(log), startticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21])
                    atomic_json(ROOT/'WORKFLOW_STAGE.json', stage)
                    atomic_json(ROOT/'runtime/HEAD_QUEUE_STATUS.json', {**stage, 'host': HOST, 'status': 'WORKFLOW_DISPATCHED', 'coordinator_pid': os.getpid()})
                    result = child.wait()
                assert result == 0, f'{name} failed with exit {result}; see {log}'
                if trainer:
                    outcome = read_json(training_root(config,trainer)/'result.json')
                    if outcome['status'] != 'COMPLETE':
                        atomic_json(ROOT/'WORKFLOW_LIVE.json', {**live,'status':'STOPPED_AT_BOUNDARY','stage':name,'result':outcome,'ended_utc':now()})
                        return
                atomic_json(training_root(config)/'workflow_stages'/f'{name}.json', {**stage, 'status': 'COMPLETE', 'ended_utc': now(),
                            'wall_seconds': time.time()-t, 'log': str(log)})
            atomic_json(ROOT/'WORKFLOW_STAGE.json', {'stage':'ALL','status':'COMPLETE','ended_utc':now()})
            atomic_json(ROOT/'WORKFLOW_LIVE.json', {**live,'status':'COMPLETE','ended_utc':now()})


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        atomic_json(ROOT/'runtime/SHARED_WORKFLOW_ERROR.json', {'status':'FAILED','utc':now(),'pid':os.getpid(),'error':repr(error)})
        raise
