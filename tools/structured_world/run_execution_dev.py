"""CPU-default dev128/full1696 plans around unchanged canonical infer/scorer.

--execute is for a later operator-admitted host. This helper has no SSH, Navtest,
weight copying, retries, process termination or automatic training acceptance.
"""
import argparse
from contextlib import ExitStack
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.structured_world import benchmark_engineering_execution as bench

SOURCE_SHA = '0c54f1c49e7ed2a76b22826b654d6c72625f5b6d'
NATIVE_SHA = 'be5e72248a46afd43e1bf279c21134c391c5edad'
INFER_SHA = 'd1132fb66f42b2b70af5467fc347dc3d960cb87af7c5da1fb90bb713401f8333'
SCORE_SHA = '754f9c606a97523dc7ffb2d1a8ea85ff18ca034113a920a9adbc2f9f75ac4372'
BENCH_SHA = 'd358f6969dd78828e705c19f5307c9d26f76b15f37aaadb35930b8cbeb66edee'
CURRENT_SHA = 'ca8980fa3bc7b36ae984bd50dd8ecb5a551db2d4281f39cda8e35b7143122c53'
CANONICAL_ADAPTER_SHA = '7bb032ec7aca4fdd83fd01ceaa4bf388cc94c47952bdc2b578012b654bb525e3'
NAVSIM_TREE = '1468974c9af4405597c9dd3e8db1b68e7d7ab337327a7595fb57c2b3dcd4a84f'
NUPLAN_TREE = '0e83aab8d5b92bf5792d815a1f25686088d1f4b8db70ea604331cd98f1606849'
SCORE_RUNTIME = {'numpy': '1.26.4', 'scipy': '1.13.1', 'shapely': '2.0.7'}
METRICS = ('score', 'no_at_fault_collisions', 'drivable_area_compliance', 'time_to_collision_within_bound',
           'ego_progress', 'comfort', 'driving_direction_compliance')
PROTOCOL = 'locked NAVSIM v1 full-reference-cache PDM; original 8 x 0.5s trajectory, 40 x 0.1s simulation'


def require(value, message):
    if not value:
        raise ValueError(message)


def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': bench.digest(path)}


def bound(record):
    require(bench.digest(record['path']) == record['sha256'], 'Bound artifact changed: '+record['path'])
    return bench.read(record['path'])


def source(root):
    bench.immutable_source(root, SOURCE_SHA)
    require(bench.digest(Path(root)/'tools/structured_world/infer_checkpoint.py') == INFER_SHA and
            bench.digest(Path(root)/'tools/structured_world/score_navsim.py') == SCORE_SHA, 'Canonical entrypoint changed')
    require(bench.digest(bench.__file__) == BENCH_SHA, 'Reviewed benchmark process/lease helper changed')


def runtime(python, packages):
    # Distribution metadata only: never import torch or initialize CUDA here.
    code = 'import json,importlib.metadata as m;print(json.dumps({n:m.version(n) for n in '+repr(sorted(packages))+'}))'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    actual = json.loads(bench.checked([python, '-B', '-c', code], env=env, timeout=30))
    require(actual == packages, 'Qualified runtime versions changed')
    return {'python': python, 'packages': actual, 'CUDA_imported': False}


def population(p):
    require(p['name'] in ('dev128', 'dev1696'), 'Only dev128/debug or full1696/dev; no Navtest')
    expected = 128 if p['name'] == 'dev128' else 1696
    require(p['scenes'] == expected, 'Population count mismatch')
    inputs = Path(p['inputs'])
    for name, sha in p['input_metadata_sha256'].items():
        require(name in ('identity.json', 'index.json', 'COMPLETE.json') and bench.digest(inputs/name) == sha, 'Current-only metadata changed')
    require(set(p['input_metadata_sha256']) == {'identity.json', 'index.json', 'COMPLETE.json'}, 'Bind all current-input metadata')
    identity, complete, current = (bench.read(inputs/n) for n in ('identity.json', 'COMPLETE.json', 'index.json'))
    index = bound(p['score_index'])
    require(complete['identity'] == identity['identity'] == p['input_identity'] and complete['contains_GT'] is False,
            'Actual current-only bank identity required; no GT input')
    require(len(current) == len(index) == complete['scenes'] == expected and len({r['token'] for r in index}) == expected,
            'Actual complete population required')
    require({(r['token'], r['log']) for r in current} == {(r['token'], r['log']) for r in index}, 'Current/score token-log membership differs')
    require(len({r['log'] for r in index}) == 16, 'All16 dev logs required even for fixed dev128')
    return index


def build_plan(manifest, output, *, gpu_uuids=(), groups=('G1', 'G3'), populations=('dev128', 'dev1696'), phases=('infer', 'score'), inference_results_root=None):
    require(manifest['schema'] == 'execution_dev_manifest_v1' and manifest['candidate_mode'] in ('engineering_fast', 'precision_fast'), 'Separate P0/E3 dev manifest required')
    source(manifest['source_root'])
    require(set(groups) <= {'G1', 'G3'} and groups and set(populations) <= {'dev128', 'dev1696'} and populations,
            'Registered dev groups/populations required')
    require(phases in (('infer',), ('score',), ('infer','score')), 'Original infer/score phases only')
    require(inference_results_root is None or phases == ('score',), 'External completed inference root is for a separate score phase')
    require(len(gpu_uuids) == len(set(gpu_uuids)) and all(re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', u) for u in gpu_uuids), 'Explicit unique full physical UUID mask required')
    require(1 <= manifest['CPU_workers'] <= 32 and manifest['minimum_free_MiB'] > 0, 'Explicit CPU pool and GPU memory reserve required')
    rows = manifest['checkpoints']; require(len(rows) == 6 and {(r['group'], r['role']) for r in rows} == {(g,r)for g in ('G1','G3')for r in ('A','AA','B')}, 'Six explicit +500 checkpoint assignments required')
    pops = {p['name']: p for p in manifest['populations']}
    require(set(populations) <= pops.keys(), 'Actual population metadata missing')
    for name in populations:
        population(pops[name])
    jobs = []
    output = Path(output).resolve(); model_root = Path(manifest['source_root']).resolve()
    for row in rows:
        if row['group'] not in groups:
            continue
        require(row['completed'] == (25500 if row['group'] == 'G1' else 50500) and
                row['tag'] == f"milestone_{row['completed']:07d}", 'Original mid+500 immutable tag required')
        require(row['mode'] == ('io_preserving_v1' if row['role'] != 'B' else manifest['candidate_mode']), 'Actual native role mode differs')
        require(Path(row['run']).is_absolute(), 'Actual native run path required')
        for name in populations:
            p = pops[name]; key = row['group']+'_'+row['role']+'_'+name
            result = output/'results'/key
            prediction_result = Path(inference_results_root).resolve()/key if inference_results_root else result
            for phase in phases:
                job = {'key': key+'_'+phase, 'phase': phase, 'group': row['group'], 'role': row['role'], 'population': name,
                       'host': row['native_host'] if phase == 'infer' else manifest['score_host'], 'checkpoint': row,
                       'population_spec': p, 'result': str(result), 'run': str(output/'processes'/(key+'_'+phase)),
                       'prediction_result': str(prediction_result),
                       'gpu_uuids': list(gpu_uuids) if phase == 'infer' else [], 'CPU_workers': manifest['CPU_workers'],
                       'trainer': str(model_root/'tools/structured_world'/('infer_checkpoint.py' if phase == 'infer' else 'score_navsim.py'))}
                if phase == 'infer':
                    job['command'] = ([manifest['inference_python'], '-B', '-u', '-m', 'torch.distributed.run', '--standalone',
                        '--nnodes=1', '--nproc_per_node='+str(len(gpu_uuids)), '--max_restarts=0', '--', job['trainer'],
                        '--run', row['run'], '--tag', row['tag'], '--inputs', p['inputs'], '--output', str(result/'predictions'),
                        '--sampling-seed', '42'] if gpu_uuids else None)
                else:
                    job['command'] = [manifest['scoring_python'], '-B', '-u', job['trainer'], '--predictions', str(prediction_result/'predictions'),
                        '--index', p['score_index']['path'], '--devkit', manifest['devkit'], '--output', str(result/'canonical'),
                        '--population', 'debug' if name == 'dev128' else 'dev', '--workers', str(manifest['CPU_workers'])]
                job['reuse'] = row.get('reuse_dev', {}).get(name)
                require(job['reuse'] is None or row['role'] in ('A','AA'), 'Only explicitly bound IO A/AA dev references can be reused')
                jobs.append(job)
    return {'schema': 'execution_dev_plan_v1', 'source_sha': SOURCE_SHA, 'driver_sha256': bench.digest(__file__),
            'benchmark_helper_sha256': BENCH_SHA, 'jobs': jobs, 'GPU_jobs_started': 0, 'scoring_jobs_started': 0,
            'execution_ready': False, 'commands_populated': bool(gpu_uuids) or phases == ('score',), 'native_checkpoints_verified_here': False,
            'limits': ['Planned +500 paths are not claims that unfinished checkpoints exist', 'No weights copied or loaded by planning',
                       'dev128 is debug only; full1696/16 logs is mandatory for dev evidence', 'Canonical FP32/TF32-off/seed42/original10step/q0&q_final; no acceptance implied']}


def native_binding(job):
    row = job['checkpoint']; run = Path(row['run']); folder = run/'checkpoints'/row['tag']
    identity, complete, done = (bench.read(p) for p in (run/'identity.json', folder/'COMPLETE.json', run/'TRAINING_COMPLETE.json'))
    require(identity['identity'] == complete['identity'] == done['identity'] and complete['completed'] == done['updates'] == row['completed'], 'Actual +500 native endpoint not completed')
    require(identity['training_source_sha'] == NATIVE_SHA and identity['execution_mode'] == row['mode'] and identity['dataset'] == 'navsim', 'Actual native source/mode/dataset mismatch')
    require(identity['config']['structured_world']['group'].startswith(row['group']+'_') and complete['ranks'] == identity['world_size'] == 8 and complete['tag'] == row['tag'], 'Native group/all8/tag mismatch')
    require(identity['training_seed'] == 42 and identity['effective_global_batch'] == 32 and identity['FM_repeat'] == 8, 'Native scientific recipe changed')
    expected = {'mp_rank_00_model_states.pt'} | {f'rng_rank{r}.pt'for r in range(8)} | {f'zero_pp_rank_{r}_mp_rank_00_optim_states.pt'for r in range(8)}
    require(set(complete['file_sha256']) == expected and all((folder/n).is_file() for n in expected), 'Actual complete native payload inventory missing')
    return {'schema': 'execution_dev_native_binding_v1', 'checkpoint': row, 'identity': identity['identity'],
            'identity_metadata': ref(run/'identity.json'), 'COMPLETE': ref(folder/'COMPLETE.json'), 'training_complete': ref(run/'TRAINING_COMPLETE.json'),
            'payload_sha256_from_COMPLETE': complete['file_sha256'], 'payload_hashes_verified_here': False,
            'verification': 'Unchanged infer_checkpoint verifies all native payload bytes before FP32 master restore'}


def selected_resources(pool, selected, minimum_free, *, inventory=None, apps=None):
    require(selected and len(selected) == len(set(selected)) and set(selected) <= set(pool), 'Selected UUIDs outside qualified pool')
    if inventory is None:
        inventory = []
        for line in bench.checked(['nvidia-smi', '--query-gpu=uuid,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], timeout=5).splitlines():
            uuid, used, free, util = [v.strip() for v in line.split(',')]
            inventory.append({'uuid': uuid, 'used_MiB': int(used), 'free_MiB': int(free), 'utilization': int(util)})
    if apps is None:
        apps = bench.checked(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader,nounits'], timeout=5).splitlines()
    rows = {r['uuid']: r for r in inventory}; require(set(pool) <= rows.keys(), 'Qualified UUID inventory changed')
    require(not any(line.split(',')[0].strip() in selected for line in apps if line.strip()), 'Selected GPU has a compute job; no preemption/retry')
    require(all(rows[u]['used_MiB'] <= 128 and rows[u]['free_MiB'] >= minimum_free and rows[u]['utilization'] <= 5 for u in selected), 'Selected GPU not idle/free; no launch')
    return {'selected_gpu_UUIDs': list(selected), 'selected_inventory': [rows[u]for u in selected], 'passed': True}


def gpu_locks(stack, root, pool, selected):
    # The old full8 benchmark takes the hash-of-pool lock. Keep it even for a
    # selected subset, then add per-UUID locks for overlapping dev selections.
    key = hashlib.sha256(json.dumps(sorted(pool)).encode()).hexdigest()
    locks = [bench.acquire(stack, Path(root)/'physical_gpu_locks'/(key+'.lock'))]
    for uuid in sorted(selected):
        locks.append(bench.acquire(stack, Path(root)/'physical_gpu_locks'/(hashlib.sha256(uuid.encode()).hexdigest()+'.lock')))
    return locks


def process_cost(job):
    p = Path(job['run'])/'PROCESS_EXIT.json'
    value = bench.read(p); wall = value['wall_seconds']
    require(type(wall) in (int,float) and math.isfinite(wall) and wall >= 0 and type(value['returncode']) is int, 'Actual process exit cost missing')
    return {'job': job['key'], 'PROCESS_EXIT': ref(p), 'host': job['host'], 'actual_pid': value['pid'],
            'returncode': value['returncode'], 'wall_seconds': wall, 'allocated_gpu_count': len(job['gpu_uuids']),
            'GPU_hours': wall*len(job['gpu_uuids'])/3600, 'charged_despite_failure': value['returncode'] != 0 or value['monitor_error'] is not None,
            'cost_basis': 'Whole process wall includes startup/native hashing/FP32 restore/IO/exit; CPU scoring GPU-hours zero'}


def verify_predictions(job, predictions, native):
    import numpy as np
    p = job['population_spec']; index = population(p); predictions = Path(predictions)
    identity, complete = bench.read(predictions/'identity.json'), bench.read(predictions/'COMPLETE.json')
    row = job['checkpoint']
    expected = native['checkpoint']
    require(all(expected[k] == row[k] for k in ('run','tag','completed','group','role','mode','native_host')), 'Native binding assignment differs')
    require(identity['training_identity'] == native['identity'] and identity['checkpoint_files_sha256'] == native['payload_sha256_from_COMPLETE'], 'Prediction native payload/identity differs')
    require(identity['schema'] == 'structured_world_FP32_predictions_v1' and identity['evaluation_source_sha'] == SOURCE_SHA
            and identity['sampling_seed'] == 42 and identity['updates'] == row['completed'] and identity['checkpoint'] == row['tag']
            and identity['training_source_sha'] == NATIVE_SHA and identity['input_identity'] == p['input_identity'], 'Prediction source/protocol/checkpoint/input mismatch')
    require(identity['precision'] == 'FP32 native ZeRO master restore, strict=True; FP32 inference; TF32 disabled'
            and identity['sampling'] == 'independent CPU scene-bound Gaussian, original 10-step FM, one residual and one final candidate'
            and identity['supervision_heads_removed'] is False and identity['scene_diagnostics'] is False, 'Canonical FP32/sampling/head inventory changed')
    require(identity['evaluation_source_files'] == {'tools/structured_world/infer_checkpoint.py':INFER_SHA,
            'starVLA/dataloader/structured_world/current_inputs.py':CURRENT_SHA}, 'Prediction entrypoint/input source differs')
    require(identity['identity'] == hashlib.sha256(json.dumps({k:v for k,v in identity.items()if k != 'identity'}, sort_keys=True).encode()).hexdigest(), 'Prediction semantic identity invalid')
    require(complete['identity'] == identity['identity'] and complete['scenes'] == len(index), 'Prediction population incomplete')
    tokens = {r['token']for r in index}
    require({p.stem for p in (predictions/'records').glob('*.json')} == tokens and {p.stem for p in (predictions/'predictions').glob('*.npz')} == tokens, 'Actual prediction membership differs')
    for token in tokens:
        path = predictions/'predictions'/(token+'.npz'); record = bench.read(predictions/'records'/(token+'.json'))
        require(record['token'] == token and record['identity'] == identity['identity'] and bench.digest(path) == record['sha256'], 'Actual prediction payload changed')
        with np.load(path, allow_pickle=False) as values:
            require(all(values[k].shape == (8,3) and values[k].dtype == np.float32 and np.isfinite(values[k]).all() for k in ('q0','q_final')), 'Nonfinite/non-FP32/noncanonical trajectory')
    return {'identity': ref(predictions/'identity.json'), 'COMPLETE': ref(predictions/'COMPLETE.json'), 'path': str(predictions),
            'training_identity': identity['training_identity'], 'checkpoint_payload_sha256': identity['checkpoint_files_sha256'], 'scenes': len(index)}


def verify_scores(job, canonical, prediction_identity):
    index = population(job['population_spec']); canonical = Path(canonical)
    identity, complete = bench.read(canonical/'identity.json'), bench.read(canonical/'COMPLETE.json')
    require(identity['protocol'] == PROTOCOL and identity['adapter_sha256'] == SCORE_SHA and identity['index_sha256'] == job['population_spec']['score_index']['sha256'], 'Canonical scoring protocol changed')
    require(identity['prediction_identity'] == prediction_identity and identity['canonical_adapter_sha256'] == CANONICAL_ADAPTER_SHA
            and identity['navsim_python_tree_sha256'] == NAVSIM_TREE and identity['nuplan_python_tree_sha256'] == NUPLAN_TREE
            and identity['runtime_versions'] == SCORE_RUNTIME, 'Canonical backend/runtime/prediction binding differs')
    require(identity['identity'] == hashlib.sha256(json.dumps({k:v for k,v in identity.items()if k != 'identity'}, sort_keys=True).encode()).hexdigest(), 'Scoring semantic identity invalid')
    require(identity['population'] == ('debug' if job['population'] == 'dev128' else 'dev') and complete['identity'] == identity['identity'], 'Scoring population mismatch')
    result = {}
    for variant in ('q0','q_final'):
        bank = canonical/variant; summary = bench.read(bank/'summary.json')
        with (bank/'scenes.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        require(len(rows) == len(index) and {(r['token'],r['log'])for r in rows} == {(r['token'],r['log'])for r in index} and all(r['status'] == 'ok'for r in rows), 'Scoring failed/partial/foreign rows retained; no acceptance')
        require(summary['identity'] == identity['identity'] and summary['valid'] is True and summary['failed'] == 0 and summary['scenes'] == len(index) and complete[variant] == summary, 'Actual valid summary missing')
        for metric in METRICS:
            values = [float(r[metric])for r in rows]
            require(all(math.isfinite(v)for v in values) and math.isclose(math.fsum(values)/len(values), summary['metrics'][metric], rel_tol=1e-12, abs_tol=1e-12), 'Actual CSV mean/summary mismatch')
        result[variant] = {'csv': ref(bank/'scenes.csv'), 'summary': ref(bank/'summary.json')}
    return {'identity': ref(canonical/'identity.json'), 'COMPLETE': ref(canonical/'COMPLETE.json'), 'banks': result}


def reused_reference(job):
    receipt = bound(job['reuse'])
    require(receipt['schema'] == 'execution_dev_receipt_v1' and receipt['canonical_source_sha'] == SOURCE_SHA and receipt['protocol_complete'] is True
            and receipt['role'] == job['role'] and job['role'] in ('A','AA') and receipt['native_mode'] == 'io_preserving_v1'
            and receipt['group'] == job['group'] and receipt['population'] == job['population'], 'Explicit IO reference provenance differs')
    for record in receipt['artifact_refs']:
        require(bench.digest(record['path']) == record['sha256'], 'Reused artifact changed')
    native = bound(receipt['native_binding'])
    pred = verify_predictions(job, receipt['predictions']['path'], native)
    require(pred == receipt['predictions'], 'Reused prediction proof differs')
    scores = verify_scores(job, receipt['canonical_path'], bound(pred['identity'])['identity'])
    require(scores == receipt['scores'], 'Reused score proof differs')
    return receipt


def execute(manifest, plan, output):
    host = socket.gethostname().removesuffix('-worker-0'); bench.authorize(host)
    require(all(j['host'] == host for j in plan['jobs']), 'Run the selected phase on its actual host; parent routes SSH, driver never dispatches remotely')
    driver_sha = bench.checked(['git','rev-parse','HEAD'], ROOT); bench.immutable_source(ROOT, driver_sha)
    require(bench.checked(['git','ls-files','--',str(Path(__file__).relative_to(ROOT))], ROOT), 'Freeze/commit dev driver before execution')
    outputs = []
    for job in plan['jobs']:
        source(manifest['source_root']); result = Path(job['result']); result.mkdir(parents=True, exist_ok=True)
        if job['reuse'] is not None:
            reused_reference(job)
            outputs.append({'job': job['key'], 'reused_IO_reference': job['reuse'], 'new_process_GPU_hours': 0.}); continue
        require(job['command'] is not None, 'Pass --gpu-uuids for later admitted inference')
        with ExitStack() as stack:
            if job['phase'] == 'infer':
                q = bound(manifest['qualifications'][host]); require(q['passed'] is True and q['host'] == host, 'Actual host qualification required')
                versions = q['environment']['packages']
                require(set(bench.PACKAGES) <= versions.keys() and 'flash_attn' in versions, 'Full qualified package inventory including extras required')
                bench.write(output/(job['key']+'_RUNTIME.json'), runtime(manifest['inference_python'], versions))
                pool = q['physical_gpu_UUIDs']; locks = gpu_locks(stack, manifest['artifact_root'], pool, job['gpu_uuids'])
                gates = []
                for sample in range(2):
                    if sample: time.sleep(.25)
                    gates.append(selected_resources(pool, job['gpu_uuids'], manifest['minimum_free_MiB']))
                bench.write(output/(job['key']+'_RESOURCE_GATE.json'), {'samples':gates})
                native = native_binding(job); bench.write(result/'NATIVE_BINDING.json', native)
                require(not (result/'predictions').exists(), 'NEW predictions only; no implicit retry/resume')
            else:
                locks = [bench.acquire(stack, Path(manifest['artifact_root'])/'execution_dev_cpu_pool.lock')]
                bench.write(output/(job['key']+'_RUNTIME.json'), runtime(manifest['scoring_python'], SCORE_RUNTIME))
                prediction_result = Path(job['prediction_result'])
                observation = bench.read(prediction_result/'INFERENCE_RECEIPT.json')
                native = bound(observation['native_binding'])
                pred = verify_predictions(job, prediction_result/'predictions', native)
                require(pred == observation['predictions'], 'Completed inference observation changed')
                require(not (result/'canonical').exists(), 'NEW scores only; failed/old scores preserved')
            try:
                bench.run_job(job, locks)
            finally:
                if (Path(job['run'])/'PROCESS_EXIT.json').exists():
                    bench.write(output/(job['key']+'_PROCESS_COST.json'), process_cost(job))
            if job['phase'] == 'infer':
                proof = verify_predictions(job, result/'predictions', native)
                require(native_binding(job) == native, 'Read-only native checkpoint metadata changed')
                bench.write(result/'INFERENCE_RECEIPT.json', {'schema':'execution_dev_inference_v1','native_binding':ref(result/'NATIVE_BINDING.json'),
                    'predictions':proof,'process_cost':process_cost(job),'passed':False,'formal_authorized':False})
            else:
                proof = verify_scores(job, result/'canonical', bound(pred['identity'])['identity'])
                native_ref = observation['native_binding']
                receipt = {'schema':'execution_dev_receipt_v1','group':job['group'],'role':job['role'],'population':job['population'],
                    'native_mode':job['checkpoint']['mode'],'canonical_source_sha':SOURCE_SHA,'predictions':pred,'canonical_path':str(result/'canonical'),
                    'scores':proof,'native_binding':native_ref,'inference_receipt':ref(prediction_result/'INFERENCE_RECEIPT.json'),
                    'inference_process_cost':observation['process_cost'],'scoring_process_cost':process_cost(job),
                    'artifact_refs':[native_ref,pred['identity'],pred['COMPLETE'],proof['identity'],proof['COMPLETE']]+[r for v in proof['banks'].values()for r in v.values()],
                    'full_dev_evidence':job['population']=='dev1696','protocol_complete':True,'passed':False,'formal_authorized':False}
                bench.write(result/'DEV_RECEIPT.json',receipt)
            outputs.append({'job':job['key'],'actual_protocol_proof':proof,'process_cost':process_cost(job)})
    bench.write(output/'COMPLETE.json',{'schema':'execution_dev_dispatch_v1','outputs':outputs,'passed':False,'formal_authorized':False,
        'GPU_hours':math.fsum(x.get('process_cost',{}).get('GPU_hours',0.)for x in outputs),'scope':'Canonical dev observations only; no acceptance or Navtest'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--groups',default='G1,G3'); parser.add_argument('--populations',default='dev128,dev1696')
    parser.add_argument('--phase',choices=('infer','score','both'),default='both'); parser.add_argument('--gpu-uuids',default='')
    parser.add_argument('--inference-results-root',type=Path,help='Earlier inference output/results for a separate CPU score phase')
    parser.add_argument('--execute',action='store_true'); args=parser.parse_args()
    require(not args.output.exists(),'NEW output directory required')
    manifest=bench.read(args.manifest); selected=tuple(args.gpu_uuids.split(','))if args.gpu_uuids else ()
    phases=('infer','score')if args.phase=='both'else(args.phase,)
    plan=build_plan(manifest,args.output,gpu_uuids=selected,groups=tuple(args.groups.split(',')),populations=tuple(args.populations.split(',')),phases=phases,inference_results_root=args.inference_results_root)
    args.output.mkdir(parents=True); bench.write(args.output/'PLAN.json',plan); bench.write(args.output/'MANIFEST_BINDING.json',ref(args.manifest))
    if args.execute:
        execute(manifest,plan,args.output)
    else:
        print(json.dumps({'status':'CPU_PLANNED_ONLY','jobs':len(plan['jobs']),'GPU_jobs_started':0,'scoring_jobs_started':0}))


if __name__=='__main__':
    main()
