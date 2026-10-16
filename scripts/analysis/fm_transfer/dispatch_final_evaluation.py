"""Dispatch explicit final Navtest shards using the qualified, unchanged worker.

One shared admission lock covers all four models. A retry is explicit, retains
the same global shard and all scene records, and first proves the old worker
has exited. No checkpoint selection, new sampler, automatic retry or training.
"""
import argparse
import fcntl
import inspect
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time
import uuid

from scripts.analysis.fm_transfer.final_lock import authenticated, EVALUATION_SOURCE
from starVLA.rl.fm_transfer.contracts import ROOT, CONFIGS, require_dispatch
from starVLA.rl.simwam_recipe.contracts import POPULATION, file_sha, fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json

PYTHON = '/mnt/project/flow-safe-improvement-20261013/venv/bin/python'
MODELS = tuple(v + '_' + s for v in CONFIGS for s in ('SFT','RL'))
RESERVE_MIB = 24000  # Conservative admission guard, not measured exclusive usage.


def validate_selection(lock_path, model, source):
    selection = authenticated(lock_path)
    source = Path(source).resolve()
    if (selection['schema'] != 'fm_transfer_final_four_models_v1' or
            selection['status'] != 'LOCKED_BEFORE_FINAL_NAVTEST' or
            not selection['formal_budget_complete'] or set(selection['models']) != set(MODELS) or
            selection['inference_seeds'] != [2101] or selection['training_seeds'] != [42] or
            selection['modes'] != ['E_NATIVE','E_SHIFT5'] or
            selection['Navtest_scores_read_for_selection'] or
            source != EVALUATION_SOURCE.resolve() or selection['evaluation_source'] != str(source)):
        raise ValueError('Use the complete registered four-model final lock and qualified worker')
    if selection['population_identity'] != json.loads(Path(POPULATION).read_text())['identity_sha256']:
        raise ValueError('Changed final population')
    for key, relative in [('sampler_sha256','scripts/analysis/simwam_recipe/evaluate_policy.py'),
                          ('transfer_evaluation_sha256','scripts/analysis/fm_transfer/evaluate_policy.py')]:
        if selection[key] != file_sha(source / relative):
            raise ValueError('Evaluation source changed after lock')
    if not json.loads((source / 'SOURCE_COMPLETE.json').read_text())['git_archive_extraction_complete']:
        raise ValueError('Unpublished worker source')
    chosen = selection['models'][model]
    export = Path(chosen['export'])
    receipt = authenticated(export / 'export.json')
    if (file_sha(export / 'export.json') != chosen['export_receipt_sha256'] or
            json.loads((export / 'COMPLETE').read_text()) != receipt or receipt['qualification'] or
            receipt['run_identity'] != chosen['run_identity'] or receipt['new_SFT'] != chosen['new_SFT'] or
            receipt['weight_sha256'] != chosen['weight_sha256'] or receipt['stage'] != chosen['stage'] or
            receipt['updates'] != chosen['updates'] or
            file_sha(export / 'config.yaml') != chosen['config_sha256'] or
            file_sha(export / 'pytorch_model.pt') != chosen['weight_sha256']):
        raise ValueError('Final exported model changed after lock')
    return selection, chosen


def partition(shards, ranks, gpus):
    if not 1 <= shards <= 64 or not ranks or len(ranks) != len(gpus):
        raise ValueError('Explicit equal-length global rank / local GPU lists required')
    if (len(set(ranks)) != len(ranks) or len(set(gpus)) != len(gpus) or
            any(r < 0 or r >= shards for r in ranks) or any(g < 0 or g >= 8 for g in gpus)):
        raise ValueError('Duplicate or invalid rank/GPU; one admitted evaluator per physical GPU')
    return list(zip(ranks,gpus))


def worker_command(source, model, output, lock_path, rank, shards):
    # Seeds, both grids, original observations and canonical scorer are owned by
    # the unchanged worker; do not pass a max-scenes or alternative sampler flag.
    return [PYTHON,'-m','scripts.analysis.fm_transfer.evaluate_policy',
        '--export',model['export'],'--export-sha256',model['weight_sha256'],
        '--output',str(output),'--cohort','NAVTEST','--shard',str(rank),
        '--shards',str(shards),'--final-lock',str(lock_path)]


def local_live(record):
    path = Path('/proc') / str(record['pid'])
    try:
        fields = (path / 'stat').read_text().split()
        if int(fields[21]) != record['start_ticks'] or fields[2] == 'Z':
            return False
        args = (path / 'cmdline').read_bytes().decode().split('\0')
        if 'scripts.analysis.fm_transfer.evaluate_policy' not in args or record['output'] not in args:
            raise RuntimeError('Recorded PID/start now has an unexpected command')
        return True
    except FileNotFoundError:
        return False


def live(record):
    if record['state'] != 'STARTED':
        raise RuntimeError('Unresolved launch receipt requires inspection; never infer it is dead')
    if record['host'] == socket.gethostname().removesuffix('-worker-0'):
        return local_live(record)
    # Administrative PID inspection only, including a previously authorized host
    # if it has since been withdrawn. An SSH timeout is an error, not death.
    # Send the exact same small stdlib probe; importing the model stack for each
    # PID check would add substantial dispatch latency without checking more.
    program = 'import json,sys\nfrom pathlib import Path\n'+inspect.getsource(local_live)+'\nprint(json.dumps(local_live(json.loads(sys.argv[1]))))'
    command = [PYTHON,'-c',program,json.dumps(record)]
    return json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',
        record['host'],shlex.join(command)],text=True,timeout=30))


def previous_attempts(queue, output, contract, ranks, retry, is_live=live):
    recorded = output / 'dispatch_contract.json'
    records = [authenticated(p) for p in sorted((output / 'dispatches').glob('*.json'))]
    if recorded.exists():
        if json.loads(recorded.read_text()) != contract:
            raise ValueError('Cannot change model/source/shard identity during evaluation')
    elif records or (output / 'manifest.json').exists():
        raise ValueError('Untracked existing evaluation; inspect before dispatch')
    parents = {}
    for rank in ranks:
        earlier = [r for r in records if r['rank'] == rank]
        if earlier and not retry:
            raise ValueError('Already dispatched rank; explicit inspected retry required')
        if retry and not earlier:
            raise ValueError('Retry must refer to an existing shard attempt')
        for record in earlier:
            if record['contract_sha256'] != fingerprint(contract) or record['output'] != str(output):
                raise ValueError('Prior receipt belongs to a different evaluation')
            if is_live(record):
                raise RuntimeError('Prior shard is still alive; no duplicate worker')
        if earlier:
            completed = output / f'rank{rank}.json'
            if completed.exists() and json.loads(completed.read_text()).get('status') == 'ASSIGNED_COMPLETE':
                raise ValueError('Completed shard needs analysis, not another inference attempt')
            parents[rank] = [r['sha256'] for r in earlier]
    # The queue-wide lock serializes these reservations across model/host loaders.
    active_uuids = set()
    for path in sorted(queue.glob('workers/*/*.json')):
        record = authenticated(path)
        if is_live(record):
            active_uuids.add(record['GPU_uuid'])
    return parents, active_uuids


def gpu_snapshot():
    text = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free',
                                    '--format=csv,noheader,nounits'],text=True)
    gpus = {int(a):dict(uuid=b.strip(),used_MiB=int(c),free_MiB=int(d))
            for a,b,c,d in (line.split(',') for line in text.splitlines())}
    text = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory',
                                    '--format=csv,noheader,nounits'],text=True)
    owners = []
    for line in text.splitlines():
        pid, gpu, memory = [x.strip() for x in line.split(',')]
        proc = Path('/proc') / pid
        try:
            stat = (proc / 'stat').read_text().split()
            owner = dict(pid=int(pid),GPU_uuid=gpu,memory_MiB=memory,visible=True,
                start_ticks=int(stat[21]),command=(proc / 'cmdline').read_bytes().decode().split('\0'))
        except OSError:
            owner = dict(pid=int(pid),GPU_uuid=gpu,memory_MiB=memory,visible=False)
        owners.append(owner)
    return dict(time=time.time(),gpus=gpus,owners=owners,CPU_affinity=len(os.sched_getaffinity(0)),load=os.getloadavg())


def admit_devices(first, second, mapping, reserved):
    selected = set()
    for _, gpu in mapping:
        a, b = first['gpus'][gpu], second['gpus'][gpu]
        if a['uuid'] != b['uuid'] or b['uuid'] in selected or b['uuid'] in reserved:
            raise RuntimeError('Changed, aliased or already reserved physical GPU')
        for sample in (first,second):
            g = sample['gpus'][gpu]
            owners = [o for o in sample['owners'] if o['GPU_uuid'] == g['uuid']]
            if g['free_MiB'] < RESERVE_MIB or (g['used_MiB'] > 0 and not owners) or any(not o['visible'] for o in owners):
                raise RuntimeError('Insufficient capacity or unidentified owner; preserve the device')
        selected.add(b['uuid'])
    return selected


def publish(path, record):
    value = dict(record);value['sha256'] = fingerprint(value);atomic_json(path,value)
    return value


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--lock',default=str(ROOT/'final_navtest/LOCK.json'))
    parser.add_argument('--source',default=str(EVALUATION_SOURCE))
    parser.add_argument('--model',choices=MODELS,required=True)
    parser.add_argument('--shards',type=int,required=True)
    parser.add_argument('--ranks',required=True,help='Global comma-separated shard indices')
    parser.add_argument('--gpus',required=True,help='Same-length local physical GPU indices')
    parser.add_argument('--retry',action='store_true',help='Explicit existing dead shards only; no data deletion')
    args = parser.parse_args()
    host = require_dispatch();lock_path = Path(args.lock).resolve();queue = lock_path.parent
    mapping = partition(args.shards,[int(x) for x in args.ranks.split(',')],[int(x) for x in args.gpus.split(',')])
    with (queue/'owner.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX | fcntl.LOCK_NB)
        selection, model = validate_selection(lock_path,args.model,args.source)
        output = queue.parent/'evaluations'/(args.model+'_final_NAVTEST')
        output.mkdir(parents=True,exist_ok=True)
        contract = dict(schema='fm_transfer_final_shards_v1',final_lock=selection['sha256'],model=args.model,
            export_receipt_sha256=model['export_receipt_sha256'],worker_source=str(Path(args.source).resolve()),
            shards=args.shards,seed=2101,modes=['E_NATIVE','E_SHIFT5'],output=str(output))
        parents, reserved = previous_attempts(queue,output,contract,[r for r,_ in mapping],args.retry)
        first = gpu_snapshot();time.sleep(2);second = gpu_snapshot()
        admit_devices(first,second,mapping,reserved)
        attempt = uuid.uuid4().hex
        dest = output/'dispatches';dest.mkdir(exist_ok=True)
        global_dest = queue/'workers'/args.model;global_dest.mkdir(parents=True,exist_ok=True)
        atomic_json(dest/(attempt+'.inventory'),dict(first=first,second=second))
        atomic_json(output/'dispatch_contract.json',contract)
        for rank,gpu in mapping:
            require_dispatch()
            current = gpu_snapshot();admit_devices(second,current,[(rank,gpu)],reserved)
            cmd = worker_command(args.source,model,output,lock_path,rank,args.shards)
            path = dest/(attempt+f'_rank{rank}.json');global_path = global_dest/path.name
            logfile = path.with_suffix('.log')
            record = dict(state='LAUNCHING',attempt=attempt,rank=rank,shards=args.shards,host=host,gpu=gpu,
                GPU_uuid=current['gpus'][gpu]['uuid'],output=str(output),start_unix=time.time(),command=cmd,
                source=str(Path(args.source).resolve()),model=args.model,contract_sha256=fingerprint(contract),
                parent_attempts=parents.get(rank,[]),log=str(logfile),dispatcher_sha256=file_sha(__file__),
                reserve_MiB=RESERVE_MIB,shared_device_owners=current['owners'])
            publish(path,record);publish(global_path,record)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),LOCAL_RANK='0',
                OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                CUBLAS_WORKSPACE_CONFIG=':4096:8',NO_ALBUMENTATIONS_UPDATE='1',TOKENIZERS_PARALLELISM='false',
                PYTHONPATH=str(Path(args.source).resolve()),VLM_ATTN_IMPLEMENTATION='flash_attention_2')
            with logfile.open('xb') as log:
                child=subprocess.Popen(cmd,cwd=args.source,env=env,stdout=log,stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,start_new_session=True)
            record.update(state='STARTED',pid=child.pid,start_ticks=int((Path('/proc')/str(child.pid)/'stat').read_text().split()[21]))
            publish(path,record);publish(global_path,record);reserved.add(record['GPU_uuid'])
        print(json.dumps(dict(status='EXPLICIT_FINAL_PARTITION_STARTED',model=args.model,attempt=attempt,
            ranks=[r for r,_ in mapping],shards=args.shards,output=str(output))))


if __name__ == '__main__':
    main()
