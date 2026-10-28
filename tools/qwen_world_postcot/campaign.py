"""Durable complete-framework IL → DEV selection → canonical RL → final Navtest.

Formal admission requires the actual engineering receipt. GPU jobs use owned
finite-lease workers; only a confirmed exited process and sealed native boundary
may migrate. No module-level PDMS-gain gate, external scorer or extra training seed exists.
"""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time
import math

from starVLA.rl.qwen_world_postcot.contracts import ROOT, authorization, allowed_hosts, file_sha, fingerprint
from starVLA.rl.qwen_world_postcot.resources import remote, probe, inventory, gpu_activity_issue, MODEL_PYTHON, OPERATOR_PYTHON
from starVLA.rl.qwen_world_postcot.checkpoint import atomic_json

IL_MILESTONES = tuple(range(5000, 100001, 5000))
RL_MILESTONES = (5000, 10000, 20000, 35000, 50000)
IL_RUN = ROOT/'runs/IL_full_seed42'
RL_RUN = ROOT/'runs/RL_full_CANONICAL_seed42'
RL_QUAL = ROOT/'qualification/RL_from_selected_full_IL'


def accepted(source):
    path = ROOT/'audits/ENGINEERING_ACCEPTANCE.json'
    if not path.exists(): return False, 'WAITING_ENGINEERING_ACCEPTANCE'
    record = json.loads(path.read_text())
    if record.get('status') != 'PASS': return False, 'ENGINEERING_'+record.get('status', 'UNTESTED')
    for name, digest in record['training_source_files'].items():
        if file_sha(Path(source)/name) != digest:
            raise ValueError('Accepted immutable training source changed: '+name)
    for name in ('real_all_losses', 'post_gradient_to_actor', 'deployment_isolation',
                 'zero2_multigpu', 'native_resume', 'profile_stable'):
        if record['checks'].get(name) != 'PASS': return False, 'WAITING_'+name
    if record['micro_batch'] not in (2, 4) or record['minimum_free_MiB'] < 8192:
        raise ValueError('Missing measured formal topology/headroom')
    return True, record


def _same_source(source, state):
    if state.get('source', source) != source:
        raise ValueError('Campaign source is immutable; no unregistered source migration')


def checkpoint_header(path, kind):
    """Read sealed metadata only; model/native resume loaders rehash real tensors."""
    root = Path(path); complete = json.loads((root/'COMPLETE').read_text())
    files = json.loads((root/'files.json').read_text()); identity = json.loads((root/'identity.json').read_text())
    if complete['files_sha256'] != file_sha(root/'files.json'):
        raise ValueError('Checkpoint inventory seal changed')
    if kind == 'IL':
        if (complete.get('status') != 'COMPLETE' or identity.get('schema') != 'world_postcot_full_IL_v1'
                or complete['identity_sha256'] != fingerprint(identity) or identity['qualification']):
            raise ValueError('Not a complete formal full-framework IL checkpoint')
        model_name = complete['model_file']
        if model_name != 'native/mp_rank_00_model_states.pt' or model_name not in files:
            raise ValueError('Missing full native IL model state')
        model_sha = files[model_name]
    else:
        body = dict(identity); digest = body.pop('sha256', None)
        if (complete.get('schema') != 2 or fingerprint(body) != digest
                or identity.get('schema') != 'world_postcot_cleanx_canonical_RL_v1'
                or identity.get('purpose') != 'formal'):
            raise ValueError('Not a complete formal canonical clean-x RL checkpoint')
        model_sha = files['optimizer.pt']
    for name in files:
        target = root/name
        if not target.resolve().is_relative_to(root.resolve()) or not target.is_file():
            raise ValueError('Unsafe/missing complete checkpoint file: '+name)
    return dict(checkpoint=str(root.resolve()), updates=complete['updates'], model_sha256=model_sha,
                identity=identity, files_sha256=complete['files_sha256'])


def find_milestones(run, kind):
    targets = IL_MILESTONES if kind == 'IL' else RL_MILESTONES
    found = {}
    for path in sorted((Path(run)/'checkpoints').glob('update*')):
        if not (path/'COMPLETE').is_file() or '.incomplete' in path.name: continue
        seal = json.loads((path/'COMPLETE').read_text())
        if seal['updates'] not in targets: continue
        item = checkpoint_header(path, kind)
        old = found.get(item['updates'])
        if old and old['files_sha256'] != item['files_sha256']:
            raise ValueError('Two different checkpoints claim one formal milestone')
        found[item['updates']] = item
    return found


def choose_training_host(minimum, *, excluded_uuids=()):
    """Permanent idle 8-GPU host, temporary idle host, then verified sharing."""
    grant, _ = authorization(); samples = []
    for host in allowed_hosts(admit_long_job=True):
        try:
            before = probe(host); time.sleep(.2); after = probe(host)
            if len(after['gpus']) != 8 or not after['worker_python_exists']: continue
            if [g['uuid'] for g in before['gpus']] != [g['uuid'] for g in after['gpus']]: continue
            if set(excluded_uuids) & {g['uuid'] for g in after['gpus']}: continue
            if min(g['free_MiB'] for g in before['gpus']+after['gpus']) < minimum: continue
            owners = before['owners']+after['owners']
            if any(not owner['visible'] for owner in owners): continue
            if any(gpu_activity_issue(gpu,sample['owners'])for sample in (before,after)for gpu in sample['gpus']):continue
            permanent = host in grant['permanent_hosts']
            if owners and (not permanent or any(not str(owner.get('output', '')).startswith(
                    ('/mnt/project/qwen-sourceflow-', '/mnt/project/qwen-world-postcot-')) for owner in owners)):
                continue
            samples.append((bool(owners), not permanent, host, after))
        except Exception:
            continue
    if not samples: return None
    shared, _, host, info = min(samples, key=lambda value: value[:3])
    return dict(host=host, gpus=[g['index'] for g in info['gpus']],
                uuids=[g['uuid'] for g in info['gpus']], share=minimum if shared else None,minimum_free_MiB=minimum)


def choose_eval_routes(minimum, excluded_uuids=(), maximum=8):
    routes, _ = inventory(minimum_free_MiB=minimum)
    permanent = set(authorization()[0]['permanent_hosts'])
    routes = [r for r in routes if r['uuid'] not in set(excluded_uuids)]
    routes.sort(key=lambda r: (r['host'] not in permanent, r['host'], r['gpu']))
    return [dict(host=r['host'], gpus=[r['gpu']], uuids=[r['uuid']], share=None,minimum_free_MiB=minimum) for r in routes[:maximum]]


def launch(host, source, command, invocation, gpus='', share=None,expected_uuids=None,minimum=None):
    directory = ROOT/'jobs'/invocation; directory.mkdir(parents=True, exist_ok=True)
    if (directory/'DISPATCH.json').exists():
        raise ValueError('Invocation identity already dispatched')
    args = [MODEL_PYTHON, '-m', 'tools.qwen_world_postcot.job_worker', '--receipt-root', str(directory),
            '--gpus', gpus, '--command-json', json.dumps(command), '--cwd', source]
    if share: args += ['--share-min-free-mib', str(share)]
    if expected_uuids is not None:args+=['--expected-gpu-uuids-json',json.dumps(expected_uuids)]
    if minimum:args+=['--minimum-free-mib',str(minimum)]
    # Namespace explicit new runs on remote containers as well as locally.
    # Do not rely on SSH propagating environment variables.
    runtime_env={key:os.environ[key] for key in ('WORLD_POSTCOT_ROOT','WORLD_POSTCOT_TASK') if key in os.environ}
    runtime_env['PYTHONPATH']=source
    code = ('import subprocess,json,os;from pathlib import Path;'
        'p=subprocess.Popen('+repr(args)+',cwd='+repr(source)+',stdout=open('
        +repr(str(ROOT/'jobs'/f'{invocation}.launcher.log'))+',"ab"),stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,**'+repr(runtime_env)+'));'
        'print(json.dumps(dict(worker_pid=p.pid,worker_start_ticks=int(Path(f"/proc/{p.pid}/stat").read_text().split()[21]))))')
    started = json.loads(remote(host, [MODEL_PYTHON, '-c', code]))
    atomic_json(directory/'DISPATCH.json', dict(host=host, source=source, command=command,
                invocation=invocation, dispatched_unix=time.time(), **started))
    return str(directory)


def owned_exit_verified(record):
    """Read-only administrative check remains allowed after a temporary lease."""
    if not record.get('child_pid') or not record.get('child_start_ticks'): return False
    code = ('import json;from pathlib import Path;'
            'p=Path('+repr('/proc/'+str(record['child_pid']))+');'
            'print(json.dumps(dict(exited=not p.exists() or int((p/"stat").read_text().split()[21])!='
            +repr(int(record['child_start_ticks']))+')))')
    host = record['host']; local = socket.gethostname().removesuffix('-worker-0')
    command = [OPERATOR_PYTHON, '-c', code] if host == local else [
        'ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', host, shlex.join([OPERATOR_PYTHON, '-c', code])]
    try:
        return json.loads(subprocess.check_output(command, text=True, timeout=12))['exited'] is True
    except Exception:
        return False


def poll_jobs(state):
    for path in (ROOT/'jobs').glob('*/CAMPAIGN_INTENT.json'):
        intent = json.loads(path.read_text())
        if intent.get('source') == state['source']:
            state.setdefault('jobs', {}).setdefault(intent['invocation'], intent['job'])
    for job in state.setdefault('jobs', {}).values():
        if job['status'] not in ('RUNNING', 'WAITING_OWNED_EXIT'): continue
        path = Path(job['receipt'])/'RESULT.json'
        if not path.exists():
            dispatched = Path(job['receipt'])/'DISPATCH.json'
            if (not (Path(job['receipt'])/'RUNNING.json').exists()
                    and time.time()-job.get('dispatch_unix', time.time()) > 180):
                job.update(status='FAILED', error='Worker startup not durably confirmed; inspect owned invocation, never duplicate')
            continue
        result = json.loads(path.read_text())
        refused=result.get('status')=='REFUSED_RESOURCE' and result.get('child_not_started')is True and not result.get('child_pid')
        if (result.get('host') != job['host'] or result.get('command') != job['command']
                or result.get('cwd') != state['source'] or result.get('gpus') != job['gpus']
                or set(result.get('requested_gpu_uuids' if refused else 'gpu_uuids', [])) != set(job['uuids'])):
            job.update(status='FAILED', error='Owned worker result does not bind this exact host/command/source/device invocation')
            continue
        if refused:
            job.update(status='REFUSED_RESOURCE',result=result,child_not_started=True)
            count=state.setdefault('resource_refusals',{}).get(job['name'],0)+1
            state['resource_refusals'][job['name']]=count
            state.setdefault('resource_retry_after',{})[job['name']]=time.time()+min(30*2**min(count,4),300)
            continue
        if not owned_exit_verified(result):
            job['status'] = 'WAITING_OWNED_EXIT'; continue
        job.update(status='COMPLETE' if result['returncode'] == 0 and result['status']=='EXITED' else 'FAILED',
                   result=result, old_process_exit_verified=True)
        if job['status']=='COMPLETE':
            state.setdefault('resource_refusals',{}).pop(job['name'],None)
            state.setdefault('resource_retry_after',{}).pop(job['name'],None)


def claimed_uuids(state):
    return {uuid for job in state.get('jobs', {}).values()
            if job['status'] in ('RUNNING', 'WAITING_OWNED_EXIT') for uuid in job.get('uuids', [])}


def dispatch(state, source, name, command, route, *, role):
    attempts = 1+sum(job['name'] == name for job in state['jobs'].values())
    invocation = name+f'_attempt{attempts:03d}'
    receipt = ROOT/'jobs'/invocation; receipt.mkdir(parents=True, exist_ok=True)
    job = dict(name=name, role=role, receipt=str(receipt), host=route['host'],
        uuids=route['uuids'], gpus=route['gpus'], command=command, status='RUNNING', dispatch_unix=time.time())
    # Persist intent before the remote fork. A controller crash cannot erase
    # ownership and silently create a second long-running lineage.
    atomic_json(receipt/'CAMPAIGN_INTENT.json', dict(source=source, invocation=invocation, job=job))
    state['jobs'][invocation] = job
    launch(route['host'], source, command, invocation,
        gpus=','.join(map(str, route['gpus'])), share=route.get('share'),
        expected_uuids=route['uuids'],minimum=route.get('minimum_free_MiB'))
    return invocation


def active(state, name=None, role=None):
    return [job for job in state['jobs'].values() if job['status'] in ('RUNNING', 'WAITING_OWNED_EXIT')
            and (name is None or job['name'] == name) and (role is None or job['role'] == role)]


def progress(run):
    path = Path(run)/'progress.json'
    return json.loads(path.read_text()) if path.exists() else None


def ensure_training(state, source, receipt, kind, il_selection=None):
    name = 'formal_'+kind; run = IL_RUN if kind == 'IL' else RL_RUN
    endpoint = 100000 if kind == 'IL' else 50000
    if active(state, name=name)or state.get('resource_retry_after',{}).get(name,0)>time.time(): return
    current = progress(run)
    if current and current['updates'] == endpoint:
        checkpoint_header(current['checkpoint'], kind); return
    route = choose_training_host(receipt['minimum_free_MiB'] if kind == 'IL' else
        receipt.get('rl_minimum_free_MiB', 48000), excluded_uuids=claimed_uuids(state))
    if not route: state['capacity_wait_'+kind] = time.time(); return
    command = [MODEL_PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8', '-m',
        'tools.qwen_world_postcot.train_il' if kind == 'IL' else 'tools.qwen_world_postcot.train_rl',
        '--config', 'configs/qwen_world_postcot/il.yaml' if kind == 'IL' else 'configs/qwen_world_postcot/rl.yaml',
        '--output', str(run)]
    if kind == 'IL':
        command += ['--micro-batch', str(receipt['micro_batch']), '--candidate-bank',
                    str(ROOT/'candidates/full_train_initial_v1'), '--teacher-cache',
                    str(ROOT/'teachers/TRAIN_resolved_v3')]
    else:
        command += ['--manifest', str(ROOT/'data/RL_HARD.json'), '--data', str(ROOT/'data/TRAIN'),
            '--il-checkpoint', il_selection['selected']['checkpoint'],
            '--parent-selection', str(ROOT/'campaign/IL_SELECTION.json')]
    if current:
        header = checkpoint_header(current['checkpoint'], kind)
        if header['updates'] != current['updates']: raise ValueError('Resume checkpoint is not the saved data/optimizer boundary')
        command += ['--resume', current['checkpoint']]
    dispatch(state, source, name, command, route, role='training')


def verified_export(state, path):
    """Hash immutable model bytes once, then check their verified file version."""
    path=Path(path);stat=(path/'model.pt').stat()
    version=[stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns]
    metadata_sha=file_sha(path/'export.json');seal_sha=file_sha(path/'COMPLETE')
    cache=state.setdefault('verified_exports',{});previous=cache.get(str(path))
    if previous is not None:
        if (previous['model_version']!=version or previous['metadata_sha256']!=metadata_sha
                or previous['seal_sha256']!=seal_sha):
            raise ValueError('Previously verified immutable export changed; do not rescore under the old identity')
        for name,old in previous['processor_versions'].items():
            current=(path/'processor'/name).stat()
            if [current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns,current.st_ctime_ns]!=old:
                raise ValueError('Previously verified deployment processor changed')
        return previous['metadata']
    from starVLA.rl.qwen_world_postcot.deployment import inspect_export
    metadata=inspect_export(path)
    processor={}
    for name in metadata['processor']:
        current=(path/'processor'/name).stat()
        processor[name]=[current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns,current.st_ctime_ns]
    cache[str(path)]=dict(metadata=metadata,model_version=version,metadata_sha256=metadata_sha,
        seal_sha256=seal_sha,processor_versions=processor)
    return metadata


def ensure_export(state, source, name, il_checkpoint, output, rl_checkpoint=None):
    path = Path(output)
    if (path/'COMPLETE').exists():
        meta = verified_export(state,path)
        if meta['qualification']: raise ValueError('Formal evaluation cannot use a qualification export')
        if Path(meta['IL_parent']['path']).resolve() != Path(il_checkpoint).resolve():
            raise ValueError('Export checkpoint lineage changed')
        return meta
    if active(state, name=name) or active(state, role='export')or state.get('resource_retry_after',{}).get(name,0)>time.time(): return None
    host = socket.gethostname().removesuffix('-worker-0')
    if host not in authorization()[0]['permanent_hosts'] or host not in allowed_hosts():
        host = next((h for h in allowed_hosts() if h in authorization()[0]['permanent_hosts']), None)
    if host is None: return None
    if probe(host)['MemAvailable_bytes'] < 40*1024**3: return None
    command = [MODEL_PYTHON, '-m', 'tools.qwen_world_postcot.export', '--checkpoint', il_checkpoint, '--output', str(path)]
    if rl_checkpoint: command += ['--rl-checkpoint', rl_checkpoint]
    dispatch(state, source, name, command, dict(host=host, gpus=[], uuids=[], share=None), role='export')
    return None


def _scene_file_version(root):
    # Shared filesystems can preserve size and coarse mtime/ctime across an
    # in-place mutation. Hash the small scene receipts themselves; never hash
    # giant model/teacher shards here. Full decoding/bootstrap remains cached.
    files=[Path(root)/'identity.json']+sorted((Path(root)/'scenes').glob('*.json'))
    rows=[]
    for path in files:
        stat=path.stat();rows.append((path.name,stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,file_sha(path)))
    return fingerprint(rows)


def _sealed_summary(path):
    value=json.loads(Path(path).read_text());body=dict(value);seal=body.pop('sha256',None)
    if fingerprint(body)!=seal or value.get('status')!='COMPLETE':raise ValueError('Changed complete evaluation summary')
    return value


def ensure_evaluation(state, source, name, export, cohort, receipt, *, dev_binding=None,seed=2101,seed_variation=False):
    from tools.qwen_world_postcot.evaluate import sampling_role
    sampling_role(cohort,seed,seed_variation,False)
    output = ROOT/'evaluations'/name
    group = state.setdefault('evaluations', {}).setdefault(name, dict(output=str(output), rounds=[]))
    if any(value>time.time()for key,value in state.get('resource_retry_after',{}).items()if key.startswith(name+'_rank')):return None
    pending = [state['jobs'][job] for round in group['rounds'] for job in round
               if state['jobs'][job]['status'] in ('RUNNING', 'WAITING_OWNED_EXIT')]
    if pending: return None
    if (output/'identity.json').exists():
        from tools.qwen_world_postcot.summarize import summarize
        identity = json.loads((output/'identity.json').read_text())
        if identity['model_sha256'] != export['model_sha256']or identity['seed']!=seed:
            raise ValueError('Evaluation belongs to a different export or sampling seed')
        if len(list((output/'scenes').glob('*.json'))) == identity['count']:
            version=_scene_file_version(output);target=output/'SUMMARY.json'
            if group.get('fully_audited_scene_file_version')==version and target.exists():
                result=_sealed_summary(target)
                if result['sha256']!=group['fully_audited_summary_sha256']:raise ValueError('Previously audited summary changed')
            else:
                result = summarize(output)
                if target.exists() and json.loads(target.read_text()) != result: raise ValueError('Changed sealed summary')
                if not target.exists(): atomic_json(target, result)
                if _scene_file_version(output)!=version:raise ValueError('Evaluation files changed during full audit')
                group.update(fully_audited_scene_file_version=version,fully_audited_summary_sha256=result['sha256'])
            group.update(status='COMPLETE', summary=str(target)); return result
    routes = choose_eval_routes(receipt.get('evaluation_minimum_free_MiB', 32768), claimed_uuids(state))
    if not routes: return None
    commands = []
    for rank, route in enumerate(routes):
        data = ROOT/'data'/cohort
        if cohort == 'NAVTEST' and not (data/'COMPLETE').exists():
            # Reuse the already sealed fixed test observations only after all
            # training/DEV selection stages above are complete.
            data = Path('/mnt/project/qwen-sourceflow-native-20261018/data/NAVTEST')
        command = [MODEL_PYTHON, '-m', 'tools.qwen_world_postcot.evaluate', '--export', export['path'],
            '--output', str(output), '--cohort', cohort, '--data', str(data),
            '--rank', str(rank), '--world', str(len(routes)), '--seed', str(seed)]
        if seed_variation:command+=['--post-selection-seed-variation']
        if cohort == 'NAVTEST':
            command += ['--parent-selection', str(ROOT/'campaign/IL_SELECTION.json'), '--dev-evaluation', dev_binding]
        commands.append(dispatch(state, source, name+f'_rank{rank:03d}', command, route, role='evaluation'))
    group['rounds'].append(commands); group['status'] = 'RUNNING'
    return None


def register_sampling_seeds():
    value=dict(schema='world_postcot_preregistered_sampling_seed_plan_v1',training_seed=42,
        primary_sampling_seed=2101,post_selection_sampling_seeds=[2102,2103],
        all_sampling_seeds=[2101,2102,2103],additional_training_seeds=0,
        inference='single_candidate_cleanx_K10_shift5',cohort='NAVTEST',
        selected_models_frozen_from_DEV_seed2101=True,Navtest_used_for_selection=False,
        summary='primary2101 separate; descriptive mean/population_std(ddof0) over three sampling seeds separate')
    path=ROOT/'campaign/PREDECLARED_SAMPLING_SEEDS.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists()and json.loads(path.read_text())!=value:raise ValueError('Preregistered inference seed plan changed')
    if not path.exists():atomic_json(path,value)
    return value


def sampling_summary(results):
    if sorted(results)!=[2101,2102,2103]:raise ValueError('All three full-population sampling seeds must complete')
    values=[results[seed]['PDMS']for seed in (2101,2102,2103)]
    if any(not math.isfinite(value)for value in values):raise ValueError('Invalid sampling variation result')
    mean=sum(values)/3
    return dict(training_seed=42,sampling_seeds=[2101,2102,2103],PDMS_by_seed=dict(zip((2101,2102,2103),values)),
        mean_PDMS=mean,population_std_PDMS=math.sqrt(sum((value-mean)**2 for value in values)/3),
        minimum_PDMS=min(values),maximum_PDMS=max(values),independent_training_seeds=1,
        interpretation='Inference-sampling variability at one fixed trained model; not multi-training-seed evidence')


def select_checkpoint(rows, *, kind, completed_updates):
    expected = IL_MILESTONES if kind == 'IL' else RL_MILESTONES
    if sorted(row['updates'] for row in rows) != list(expected):
        raise ValueError('Full fixed DEV checkpoint window is incomplete')
    if any(row['qualification'] or row['cohort'] != 'DEV' or not 0 <= row['PDMS'] <= 100 for row in rows):
        raise ValueError('Checkpoint selection requires complete valid formal DEV results')
    best = max(rows, key=lambda row: (row['PDMS'], row['updates']))
    value = dict(schema='world_postcot_DEV_selection_v1' if kind == 'IL' else 'world_postcot_RL_DEV_selection_v1',
        completed_IL_updates=completed_updates if kind == 'IL' else 100000,
        completed_RL_updates=None if kind == 'IL' else completed_updates,
        Navtest_used=False, criterion='maximum full DEV canonical PDMS; exact tie prefers later update',
        evaluated_candidates=rows, selected=best)
    value['sha256'] = fingerprint(value)
    return value


def _save_selection(kind, rows, completed_updates):
    value = select_checkpoint(rows, kind=kind, completed_updates=completed_updates)
    target = ROOT/'campaign'/f'{kind}_SELECTION.json'
    if target.exists() and json.loads(target.read_text()) != value: raise ValueError('Do not change frozen DEV selection')
    if not target.exists(): atomic_json(target, value)
    return value


def advance_dev_window(state, source, receipt, kind, il_selection=None):
    run = IL_RUN if kind == 'IL' else RL_RUN
    rows = state.setdefault('dev_results', {}).setdefault(kind, {})
    for step, header in find_milestones(run, kind).items():
        if str(step) in rows: continue
        name = f'{kind}_update{step:06d}'
        parent = header['checkpoint'] if kind == 'IL' else il_selection['selected']['checkpoint']
        path = ROOT/'exports'/name
        meta = ensure_export(state, source, 'export_'+name, parent, path,
                             rl_checkpoint=header['checkpoint'] if kind == 'RL' else None)
        if meta is None: continue
        meta = dict(meta, path=str(path))
        result = ensure_evaluation(state, source, 'DEV_'+name, meta, 'DEV', receipt)
        if result is None: continue
        rows[str(step)] = dict(checkpoint=header['checkpoint'], model_sha256=header['model_sha256'],
            updates=step, PDMS=result['PDMS'], qualification=result['qualification'], cohort='DEV',
            export=str(path), export_model_sha256=meta['model_sha256'],
            evaluation=result['evaluation'], summary=str(ROOT/'evaluations'/('DEV_'+name)/'SUMMARY.json'),
            evaluation_identity_sha256=result['evaluation_identity_sha256'])
    return list(rows.values())


def advance_refreshes(state,source,receipt):
    """At most nine asynchronous full-TRAIN current-policy banks; never gate IL."""
    from tools.qwen_world_postcot.refresh_candidates import REFRESH_STEPS,snapshot_binding
    count=json.loads((ROOT/'data/TRAIN/manifest.json').read_text())['count']
    registry_path=ROOT/'candidate_refresh/REGISTRY.json'
    registered={}
    if registry_path.exists():
        value=json.loads(registry_path.read_text());body=dict(value);seal=body.pop('sha256',None)
        if value['schema']!='world_postcot_bank_registry_v1'or(seal is not None and fingerprint(body)!=seal):
            raise ValueError('Current-policy registry changed')
        registered={value['source_updates']:value for value in body['entries']}
    for step,header in find_milestones(IL_RUN,'IL').items():
        if step not in REFRESH_STEPS:continue
        task=state.setdefault('candidate_refreshes',{}).setdefault(str(step),dict(source_updates=step,stage='WAITING_EXPORT'))
        if step in registered:
            if registered[step]['source_model_sha256']!=header['model_sha256']:raise ValueError('Registered refresh changed its source policy')
            task.update(stage='COMPLETE',registry_entry=registered[step]);continue
        export_path=ROOT/'exports'/f'IL_update{step:06d}'
        metadata=ensure_export(state,source,f'export_IL_update{step:06d}',header['checkpoint'],export_path)
        if metadata is None:continue
        binding=snapshot_binding(header['checkpoint'],metadata)
        if 'binding'in task and task['binding']!=binding:raise ValueError('Queued refresh source/actual saved parent changed')
        task['binding']=binding;output=ROOT/'candidate_refresh'/f'update{step:06d}'/'bank'
        task['bank_root']=str(output)
        failures=[job for job in state['jobs'].values()if job['role'].startswith('candidate_refresh')
            and job['name'].startswith(f'refresh_{step:06d}_')and job['status']=='FAILED'
            and not job.get('result',{}).get('migration_required')]
        if len(failures)>=3:
            task['stage']='FAILED_RETRY_BUDGET_EXHAUSTED';continue
        world=task.get('world')
        if world and all((output/'parts'/f'rank{rank:05d}'/'COMPLETE.json').exists()for rank in range(world)):
            name=f'refresh_{step:06d}_finalize'
            if active(state,name=name):continue
            if state.get('resource_retry_after',{}).get(name,0)>time.time():continue
            host=next((host for host in allowed_hosts()if host in authorization()[0]['permanent_hosts']),None)
            if host:
                command=[MODEL_PYTHON,'-m','tools.qwen_world_postcot.refresh_candidates','--phase','finalize',
                    '--output',str(output),'--data',str(ROOT/'data/TRAIN'),'--world',str(world)]
                dispatch(state,source,name,command,dict(host=host,gpus=[],uuids=[],share=None),role='candidate_refresh_finalize')
                task['stage']='PUBLISHING_COMPLETE_BANK'
            continue
        routes=choose_eval_routes(receipt.get('refresh_minimum_free_MiB',32768),claimed_uuids(state),maximum=8)
        if not routes:task['stage']='WAITING_SPARE_GPU';continue
        if world is None:world=task['world']=len(routes)
        for rank in range(world):
            part=output/'parts'/f'rank{rank:05d}';name=f'refresh_{step:06d}_rank{rank:03d}'
            if(part/'COMPLETE.json').exists()or active(state,name=name):continue
            if state.get('resource_retry_after',{}).get(name,0)>time.time():continue
            if state.get('resource_refusals',{}).get(name,0)>=8:
                task['stage']='RESOURCE_RETRY_BUDGET_EXHAUSTED';continue
            if not routes:break
            assigned=(count+world-1-rank)//world
            command=[MODEL_PYTHON,'-m','tools.qwen_world_postcot.refresh_candidates',
                '--source-checkpoint',header['checkpoint'],'--export',str(export_path),'--data',str(ROOT/'data/TRAIN'),
                '--output',str(output),'--rank',str(rank),'--world',str(world),
                '--max-new-candidates',str(2*assigned),'--max-simulation-attempts',str(6*assigned)]
            if(part/'BUILD_IDENTITY.json').exists():command+=['--resume']
            dispatch(state,source,name,command,routes.pop(0),role='candidate_refresh')
            task['stage']='RUNNING_ASYNCHRONOUS_REFRESH'


def rl_qualification_passed():
    rows = []
    for rank in range(8):
        path = RL_QUAL/f'result_rank{rank}.json'
        if not path.exists(): return False
        value = json.loads(path.read_text())
        if value['qualification'] and value['updates'] < 16 and value['status'] == 'STOPPED_AT_SAVED_BOUNDARY':
            return False  # migrate/resume this diagnostic without resetting it
        if (not value['qualification'] or value['updates'] != 16
                or value['status'] != 'STOPPED_AT_SAVED_BOUNDARY'
                or value['frozen_SHA_before'] != value['frozen_SHA_after']
                or not any(value['changed_adapters'].values())
                or value['frozen_observation_before'] != value['frozen_observation_after']):
            raise ValueError('Actual selected-parent canonical RL qualification failed')
        rows.append(value)
    atomic_json(ROOT/'campaign/RL_ENGINEERING_ACCEPTANCE.json', dict(status='PASS',
        scope='actual16updates8ranks clean-x realcanonicalPDM; source runtime validates sampler/recompute means',
        qualification_weights_used_for_formal_initialization=False,
        result_files={f'result_rank{rank}.json':file_sha(RL_QUAL/f'result_rank{rank}.json') for rank in range(8)}))
    return True


def verify_complete_inputs(state):
    paths = dict(data=ROOT/'data/TRAIN/manifest.json', teacher=ROOT/'teachers/TRAIN_resolved_v3/MANIFEST.json',
                 candidate=ROOT/'candidates/full_train_initial_v1/MANIFEST.json')
    hashes = {name: file_sha(path) for name, path in paths.items()}
    if state.get('inputs_verified'):
        if hashes != state['inputs_verified']['manifests']:
            raise ValueError('A complete input version changed during the formal lineage')
        return
    manifest = json.loads(paths['data'].read_text())
    seal = json.loads((paths['data'].parent/'COMPLETE').read_text())
    if (manifest['cohort'] != 'TRAIN' or seal['manifest_sha256'] != hashes['data']
            or file_sha(paths['data'].parent/'records.pkl') != manifest['records_sha256']):
        raise ValueError('TRAIN population/observations are not sealed')
    from starVLA.model.modules.qwen_world_postcot.teacher_targets import TeacherCacheReader
    from starVLA.model.modules.qwen_world_postcot.candidate_bank import CandidateBank
    teacher = TeacherCacheReader(paths['teacher'].parent, expected_cohort='TRAIN')
    bank = CandidateBank(paths['candidate'].parent, expected_identity=dict(
        source_data_manifest_sha256=hashes['data'], source_records_sha256=manifest['records_sha256']))
    try:
        if (set(teacher.records) != set(manifest['tokens'])
                or set(bank.manifest['scene_ids']) != set(manifest['tokens'])
                or teacher.identity['records_sha256'] != manifest['records_sha256']):
            raise ValueError('Complete teacher/candidate coverage must match every fixed TRAIN scene')
        validity=Counter();timing=Counter()
        for record in teacher.records.values():
            metadata=record['metadata'];views=metadata['valid_future_views']
            if views not in (0,1,2,3) or type(metadata['teacher_clip_timing_valid']) is not bool:
                raise ValueError('Teacher full-cache validity accounting is missing or malformed')
            if views and not metadata['teacher_clip_timing_valid']:
                raise ValueError('Future teacher admitted invalid clip timestamps')
            validity[views]+=1;timing[metadata['teacher_clip_timing_valid']]+=1
    finally:
        teacher.close(); bank.close()
    state['inputs_verified'] = dict(manifests=hashes, count=len(manifest['tokens']), complete_coverage=True,
        resolved_teacher_root=str(paths['teacher'].parent),valid_future_views_histogram=dict(validity),
        clip_timing_valid_histogram=dict(timing),validity_counts_verified=True)


def tick(source, state):
    _same_source(source, state); state.setdefault('jobs', {}); poll_jobs(state)
    from tools.qwen_world_postcot.monitor import snapshot as monitor_snapshot
    state.setdefault('monitors',{})
    for run in (IL_RUN,RL_RUN):
        result=monitor_snapshot(run)
        state['monitors'][Path(run).name]=result
        if Path(run).exists():atomic_json(Path(run)/'MONITOR.json',result)
    failures = [name for name, job in state['jobs'].items() if job['status'] == 'FAILED'and not job['role'].startswith('candidate_refresh')]
    if failures: state.update(stage='OWNED_JOB_FAILED_REVIEW_REQUIRED', failed_jobs=failures); return state
    optional={job['name']for job in state['jobs'].values()if job['role'].startswith('candidate_refresh')}
    exhausted=[name for name,count in state.get('resource_refusals',{}).items()if count>=8 and name not in optional]
    if exhausted:state.update(stage='RESOURCE_ADMISSION_RETRY_BUDGET_EXHAUSTED',resource_review_required=exhausted);return state
    okay, receipt = accepted(source)
    if not okay: state['stage'] = receipt; return state
    for path in (ROOT/'teachers/TRAIN_resolved_v3/MANIFEST.json', ROOT/'candidates/full_train_initial_v1/COMPLETE.json',
                 ROOT/'data/TRAIN/COMPLETE', ROOT/'data/DEV/COMPLETE'):
        if not path.exists(): state['stage'] = 'WAITING_COMPLETE_INPUT_'+str(path.relative_to(ROOT)); return state
    verify_complete_inputs(state)
    ensure_training(state, source, receipt, 'IL')
    il_rows = advance_dev_window(state, source, receipt, 'IL')
    try:advance_refreshes(state,source,receipt)
    except Exception as error:
        state['candidate_refresh_controller_error']=repr(error)  # retain prior complete bank; IL continues
    il_progress = progress(IL_RUN)
    if not il_progress or il_progress['updates'] != 100000 or active(state, name='formal_IL'):
        state['stage'] = 'IL_AND_PERIODIC_FULL_DEV'; return state
    if len(il_rows) != len(IL_MILESTONES): state['stage'] = 'WAITING_COMPLETE_IL_DEV_WINDOW'; return state
    il_selection = _save_selection('IL', il_rows, 100000)
    if not rl_qualification_passed():
        if not active(state, name='selected_parent_RL_qualification')and state.get('resource_retry_after',{}).get('selected_parent_RL_qualification',0)<=time.time():
            route = choose_training_host(receipt.get('rl_minimum_free_MiB', 48000), excluded_uuids=claimed_uuids(state))
            if route:
                command = [MODEL_PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
                    '-m', 'tools.qwen_world_postcot.train_rl', '--config', 'configs/qwen_world_postcot/rl.yaml',
                    '--manifest', str(ROOT/'data/RL_HARD.json'), '--data', str(ROOT/'data/TRAIN'),
                    '--il-checkpoint', il_selection['selected']['checkpoint'], '--output', str(RL_QUAL),
                    '--qualification', '--stop-after-update', '16', '--save-every', '8']
                current = progress(RL_QUAL)
                if current: command += ['--resume', current['checkpoint']]
                dispatch(state, source, 'selected_parent_RL_qualification', command, route, role='qualification')
        state['stage'] = 'REAL_SELECTED_PARENT_RL_QUALIFICATION'; return state
    if active(state, name='selected_parent_RL_qualification'):
        state['stage'] = 'WAITING_QUALIFICATION_WORKER_EXIT'; return state
    ensure_training(state, source, receipt, 'RL', il_selection)
    rl_rows = advance_dev_window(state, source, receipt, 'RL', il_selection)
    rl_progress = progress(RL_RUN)
    if not rl_progress or rl_progress['updates'] != 50000 or active(state, name='formal_RL'):
        state['stage'] = 'RL_AND_PERIODIC_FULL_DEV'; return state
    if len(rl_rows) != len(RL_MILESTONES): state['stage'] = 'WAITING_COMPLETE_RL_DEV_WINDOW'; return state
    rl_selection = _save_selection('RL', rl_rows, 50000)
    register_sampling_seeds()
    state['stage'] = 'FINAL_DEV_LOCKED_NAVTEST'
    final = {}
    for kind, selection in (('IL', il_selection), ('RL', rl_selection)):
        selected = selection['selected']
        meta = dict(verified_export(state,selected['export']), path=selected['export'])
        result = ensure_evaluation(state, source, 'NAVTEST_final_'+kind, meta, 'NAVTEST', receipt,
                                   dev_binding=selected['evaluation'])
        if result is not None: final[kind] = result
    if len(final) == 2:
        from tools.qwen_world_postcot.summarize import summarize
        baseline_binding=ROOT/'campaign/BASELINE_BINDING.json'
        if not baseline_binding.exists():raise ValueError('Own native references must be pinned before any new-method Navtest')
        main_path=ROOT/'campaign/MAIN_SEED2101_COMPARISON.json'
        fixed=dict(IL_selection_sha256=il_selection['sha256'],RL_selection_sha256=rl_selection['sha256'],
            native_baseline_binding_sha256=file_sha(baseline_binding),source=source)
        if main_path.exists():
            main_result=_sealed_summary(main_path)
            if any(main_result[key]!=value for key,value in fixed.items()):raise ValueError('Frozen primary-result source/selection changed')
            il_main,paired=main_result['IL'],main_result['RL']
        else:
            il_main=summarize(ROOT/'evaluations/NAVTEST_final_IL',legacy_binding=baseline_binding)
            paired = summarize(ROOT/'evaluations/NAVTEST_final_RL', ROOT/'evaluations/NAVTEST_final_IL',legacy_binding=baseline_binding)
            main_result=dict(status='COMPLETE',primary_sampling_seed=2101,IL=il_main,RL=paired,**fixed,
                single_candidate_K10_shift5=True,main_training_seed=42,Navtest_used_for_selection=False)
            main_result['sha256']=fingerprint(main_result);atomic_json(main_path,main_result)
        state['stage']='POST_SELECTION_SAMPLING_VARIATION'
        variants={'IL':{2101:il_main},'RL':{2101:paired}}
        for seed in (2102,2103):
            for kind,selection in (('IL',il_selection),('RL',rl_selection)):
                selected=selection['selected'];meta=dict(verified_export(state,selected['export']),path=selected['export'])
                result=ensure_evaluation(state,source,f'NAVTEST_seed{seed}_{kind}',meta,'NAVTEST',receipt,
                    dev_binding=selected['evaluation'],seed=seed,seed_variation=True)
                if result is not None:variants[kind][seed]=result
        if any(len(value)!=3 for value in variants.values()):
            state['sampling_results_complete']={kind:sorted(value)for kind,value in variants.items()}
            return state
        seed_pairs={2101:paired}
        for seed in (2102,2103):
            seed_pairs[seed]=summarize(ROOT/f'evaluations/NAVTEST_seed{seed}_RL',ROOT/f'evaluations/NAVTEST_seed{seed}_IL')
        atomic_json(ROOT/'campaign/FINAL_COMPARISON.json',dict(status='COMPLETE',primary_seed2101=main_result,
            inference_sampling_variation={kind:sampling_summary(value)for kind,value in variants.items()},
            paired_results_by_sampling_seed=seed_pairs,models_frozen_before_all_Navtest=True,
            additional_training_seeds=0,Navtest_used_for_selection=False))
        state['stage'] = 'COMPLETE'; state['final_comparison'] = str(ROOT/'campaign/FINAL_COMPARISON.json')
    return state


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', required=True); parser.add_argument('--execute-ready-stages', action='store_true')
    parser.add_argument('--loop', action='store_true'); parser.add_argument('--interval', type=int, default=60)
    args = parser.parse_args()
    if not args.execute_ready_stages: raise ValueError('Explicitly arm authorized complete main experiment')
    directory = ROOT/'campaign'; directory.mkdir(parents=True, exist_ok=True)
    lock = (directory/'owner.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = directory/'STATE.json'; source = str(Path(args.source).resolve())
    state = json.loads(path.read_text()) if path.exists() else dict(schema='world_postcot_campaign_v2', source=source, jobs={})
    _same_source(source, state)
    while True:
        try:
            state = tick(source, state)
        except Exception as error:
            state.update(stage='CONTROLLER_ERROR', error=repr(error))
        state['last_tick_unix'] = time.time(); atomic_json(path, state)
        if not args.loop or state['stage'] == 'COMPLETE': break
        time.sleep(max(5, args.interval))


if __name__ == '__main__':
    main()
