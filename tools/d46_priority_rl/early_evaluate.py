"""Finite, independently locked full-NAVTEST batch on user-released GPUs.

Uses the frozen native evaluator and existing UUID/lease/owner admission. It
does not edit the scientific campaign, stop learners, or return labels to them.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_world_compare.contracts import allowed_hosts, file_sha, fingerprint
from tools.d44_native_rl.launch import launch, DispatchPreflightFailed
from tools.d44_native_rl.qualification_queue import capacity

ROOT = Path('/mnt/project/d46-priority-rl-20261103')
BATCH = ROOT / 'operations/EARLY_EVAL_20261103'
PYTHON = '/mnt/project/flow-safe-improvement-20261013/venv/bin/python'
SOURCE = Path(__file__).resolve().parents[2]
HOSTS = ('training-vlawm-tsinghua-zt', 'training-rl-zt2', 'training-vlawm-zt2')


def read(path):
    return json.loads(Path(path).read_text())


def policy_signature(identity, files):
    return fingerprint(dict(parent=identity['parent_model_sha256'],
                            trainable_payload=files['optimizer.pt'], scope=identity['scope']))


def pending_shards(cases, world):
    """Round robin across policies; no sequential-policy starvation."""
    return [(case, rank) for rank in range(world) for case in cases]


def snapshot():
    import torch
    from starVLA.rl.d44_native_rl.deployment import inspect
    if (BATCH / 'PLAN.json').exists():
        raise FileExistsError('Keep this finite batch immutable')
    campaign = read(ROOT / 'CAMPAIGN.json')
    latest = {arm: Path(read(Path(run['output']) / 'latest.json')['checkpoint'])
              for arm, run in campaign['runs'].items()}
    common = min(read(latest[arm] / 'COMPLETE')['updates'] for arm in ('D0', 'D1', 'D2'))
    for arm in ('D0', 'D1', 'D2'):
        latest[arm] = Path(campaign['runs'][arm]['output']) / 'checkpoints' / f'update{common:06d}_inner0'
    cases = []
    references = []
    signatures = {}
    for arm in ('D0', 'D5', 'D3_FPO', 'D3_AWM', 'D1', 'D2'):
        cp = latest[arm]
        identity, seal, files = inspect(cp)
        if identity['qualification'] or identity['recipe']['parent'] != 'D_IL46K' or seal['next_inner'] != 0:
            raise ValueError('Requires sealed formal IL46 checkpoint at a collection boundary')
        state = torch.load(cp / 'rank0.pt', map_location='cpu', weights_only=False)['state']
        if state['updates'] != seal['updates']:
            raise ValueError('Checkpoint state/seal update mismatch')
        entry = dict(arm=arm, checkpoint=str(cp), updates=seal['updates'],
                     scene_exposures=state['valid_scene_exposures'],
                     exposure_epoch=state['valid_scene_exposures'] / 103288,
                     policy_signature=policy_signature(identity, files),
                     identity_sha256=identity['sha256'], payload_sha256=files['optimizer.pt'],
                     source=campaign['runs'][arm]['source'])
        sig = entry['policy_signature']
        if sig in signatures:
            references.append(dict(entry, reference_case=signatures[sig],
                                   mode='exact_payload_reference_not_independent_evaluation'))
        else:
            entry['id'] = f'EARLY_{arm}_u{seal["updates"]:06d}'
            entry['output'] = str(ROOT / 'evaluations' / entry['id'] / 'evaluation')
            signatures[sig] = entry['id']
            cases.append(entry)
    plan = dict(schema='D46_finite_early_full_NAVTEST_plan_v1', created_at=time.time(),
                reason='User requested early full evaluation before preferential resource allocation',
                cases=cases, equivalent_references=references, world=32, hosts=list(HOSTS),
                parent_PDMS=89.4955209758546,
                baseline=read(ROOT / 'PARENT_CONTRACT.json')['complete_navtest']['original_evaluation'],
                training_continues=True, training_labels_exported=False,
                protocol='B1 single clean-x ODE K10 shift5 seed2101;12146 scenes/44 physical logs/invalid0',
                source=str(SOURCE), source_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip(),
                selection_policy='Early scores allocate extra resources; dormant mechanisms are not ruled out')
    plan['sha256'] = fingerprint(plan)
    BATCH.mkdir(parents=True, exist_ok=True)
    atomic_json(BATCH / 'PLAN.json', plan)
    return plan


def initial_state(plan):
    return dict(schema='D46_finite_early_full_NAVTEST_state_v1', plan_sha256=plan['sha256'],
                cases={c['id']: dict(status='PENDING', jobs={}, output=c['output'],
                                    completed_shards=0, active_shards=0) for c in plan['cases']},
                complete=False)


def reconcile(case, world):
    completed = 0
    active = 0
    for rank in range(world):
        jobs = case['jobs'].get(str(rank), [])
        if not jobs:
            continue
        result = Path(jobs[-1]['job']) / 'RESULT.json'
        if not result.exists():
            active += 1
            continue
        receipt = read(result)
        if receipt['returncode'] == 0:
            report = Path(case['output']) / f'rank{rank}.json'
            if report.exists() and read(report).get('status') == 'COMPLETE':
                completed += 1
            else:
                case.update(status='FAILED_REQUIRES_REVIEW', error='Exit0 without complete shard', failed_rank=rank)
        elif receipt['returncode'] != 75:
            case.update(status='FAILED_REQUIRES_REVIEW', failed_rank=rank, failed_result=str(result))
    case.update(completed_shards=completed, active_shards=active)
    return completed == world


def summarize_case(case, baseline):
    from tools.qwen_world_compare.evaluate import summarize
    from tools.d44_native_rl.analyze import analyze
    output = Path(case['output'])
    summary = summarize(output, baseline)
    # summarize checks complete population, row seals, protocol and invalids.
    atomic_json(output.parent / 'SUMMARY.json', summary)
    paired = analyze(output, baseline)
    paired['analysis_schema'] = 'D46_complete_B1_paired_gain_loss_v1'
    gain = paired['gain_loss']
    gain['fixed_reference_cohorts'] = {k.replace('IL44', 'IL46'): v
                                      for k, v in gain['fixed_reference_cohorts'].items()}
    gain['limits'] = gain['limits'].replace('IL44', 'IL46')
    paired.pop('sha256', None)
    paired['sha256'] = fingerprint(paired)
    atomic_json(output.parent / 'PAIRED_ANALYSIS.json', paired)
    case.update(status='COMPLETE', summary=str(output.parent / 'SUMMARY.json'),
                paired_analysis=str(output.parent / 'PAIRED_ANALYSIS.json'))


def tick(plan, state):
    allowed = set(allowed_hosts(admit_long_job=True))
    reserved = set()
    for spec in plan['cases']:
        case = state['cases'][spec['id']]
        if case['status'] != 'COMPLETE' and reconcile(case, plan['world']):
            try:
                summarize_case(case, plan['baseline'])
            except Exception:
                case.update(status='FAILED_SUMMARY_REQUIRES_REVIEW', error=traceback.format_exc())
        for jobs in case['jobs'].values():
            if jobs and not (Path(jobs[-1]['job']) / 'RESULT.json').exists():
                reserved.update((jobs[-1]['host'], g) for g in jobs[-1]['gpus'])
    pool = []
    for host in plan['hosts']:
        if host not in allowed:
            continue
        try:
            # Initially vacant GPUs only. Live worker independently rechecks
            # owner visibility, unexplained allocations, UUID and both leases.
            pool.extend((host, gpu) for gpu in capacity(host, 79000)
                        if (host, gpu) not in reserved)
        except (OSError, ValueError, subprocess.SubprocessError, DispatchPreflightFailed) as error:
            state.setdefault('inventory_errors', {})[host] = dict(at=time.time(), error=repr(error))
    specs = {c['id']: c for c in plan['cases']}
    for case_id, rank in pending_shards(list(specs), plan['world']):
        case = state['cases'][case_id]
        if case['status'].startswith(('COMPLETE', 'FAILED', 'DISPATCH_REQUIRES')):
            continue
        jobs = case['jobs'].setdefault(str(rank), [])
        if jobs:
            result = Path(jobs[-1]['job']) / 'RESULT.json'
            if not result.exists() or read(result)['returncode'] == 0:
                continue
            if time.time() < read(result)['end_unix'] + 300:
                continue
        if not pool:
            break
        host, gpu = pool.pop(0)
        spec = specs[case_id]
        job = ROOT / 'jobs/EARLY_EVALUATE' / case_id / f'rank{rank:02d}_attempt{len(jobs)+1:03d}'
        command = [PYTHON, '-m', 'tools.d44_native_rl.evaluate', '--checkpoint', spec['checkpoint'],
                   '--output', spec['output'], '--rank', str(rank), '--world', str(plan['world'])]
        try:
            receipt = launch(host, spec['source'], job, command, [gpu],
                             share_min_free_mib=None, worker_source=SOURCE,
                             task_root=ROOT, task='d46_priority_rl_20261103')
        except DispatchPreflightFailed as error:
            case['last_prechild_error'] = repr(error)
            continue
        except Exception:
            # A lost SSH response can still have started a child. Freeze this
            # case for reconciliation instead of creating a duplicate worker.
            case.update(status='DISPATCH_REQUIRES_RECONCILIATION', error=traceback.format_exc(), ambiguous_job=str(job))
            continue
        jobs.append(receipt)
        case['status'] = 'RUNNING'
        atomic_json(BATCH / 'STATE.json', state)
    for case in state['cases'].values():
        reconcile(case, plan['world'])
    state.update(at=time.time(), owner_pid=os.getpid(),
                 complete=all(c['status'] == 'COMPLETE' for c in state['cases'].values()),
                 terminal=all(c['status'] == 'COMPLETE' or c['status'].startswith(('FAILED', 'DISPATCH_REQUIRES'))
                              for c in state['cases'].values()))
    return state


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=('snapshot', 'run', 'once'))
    parser.add_argument('--interval', type=int, default=60)
    args = parser.parse_args()
    if args.action == 'snapshot':
        print(json.dumps(snapshot(), indent=2)); return
    plan = read(BATCH / 'PLAN.json')
    body = dict(plan); digest = body.pop('sha256')
    if fingerprint(body) != digest:
        raise ValueError('Immutable early evaluation plan changed')
    with (BATCH / 'OWNER.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(BATCH / 'STATE.json') if (BATCH / 'STATE.json').exists() else initial_state(plan)
        if state['plan_sha256'] != digest:
            raise ValueError('Early evaluation state/plan mismatch')
        while True:
            state = tick(plan, state)
            atomic_json(BATCH / 'STATE.json', state)
            if args.action == 'once' or state['complete'] or (state['terminal'] and not any(c['active_shards'] for c in state['cases'].values())):
                break
            time.sleep(max(10, args.interval))


if __name__ == '__main__':
    main()
