"""Four independent generic-init IL screens; sealed Test selection; native continuation.

Uses the existing lease-backed campaign workers, trainer, checkpoint, refresh,
and native export/evaluator. No new optimizer or scoring implementation.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from starVLA.rl.qwen_world_postcot.contracts import ROOT, TASK, file_sha, fingerprint, allowed_hosts
from starVLA.rl.qwen_world_postcot.checkpoint import atomic_json
from starVLA.rl.qwen_world_postcot.fresh_loss import ARMS, read_registration, select_survivors
from tools.qwen_world_postcot import campaign as c

SOURCE = Path(__file__).resolve().parents[2]
CONTROLLER_SOURCE = SOURCE
RECOVER = False
BASE = Path('/mnt/project/qwen-world-postcot-20261025')
CONTROL = BASE/'original_vs_optimized_labels_NAVTEST_20261029T0500Z/ORIGINAL_GT_IL_010000/evaluation'


def read(path): return json.loads(Path(path).read_text())


def held_shared_uuids(root='/mnt/project/runtime_gpu_shared_workers'):
    """Only inspect advisory leases; never evict/overwrite a live owner."""
    held = set()
    for path in Path(root).glob('*.lock'):
        with path.open('a') as stream:
            try: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: held.add(path.stem)
    return held


def training_route(state):
    shared=held_shared_uuids();primary=held_shared_uuids('/mnt/project/runtime_gpu_leases')
    excluded=c.claimed_uuids(state)|(shared&primary)
    routing=ROOT/'EXECUTION_ROUTING.json'
    eligible=None
    if routing.exists():
        directive=read(routing)
        if (directive.get('schema')!='world_IL_execution_routing_v1' or directive.get('arm')!=ROOT.name
                or directive.get('world')!=8 or not isinstance(directive.get('training_hosts'),list)):
            raise ValueError('Invalid explicit training migration route')
        eligible=directive['training_hosts']
    # Entire eight-GPU placement must use one vacant slot. Mixed busy patterns
    # are skipped, never repaired by taking another task's lease.
    for _ in range(len(allowed_hosts())):
        route=c.choose_training_host(49152,excluded_uuids=excluded,eligible_hosts=eligible)
        if route is None:return None
        uuids=set(route['uuids'])
        if route.get('share') and not uuids&shared:route['lease_slot']='shared';return route
        if not uuids&primary:route['lease_slot']='primary';return route
        excluded|=uuids
    return None


def process_alive(receipt):
    path=Path('/proc')/str(receipt['pid'])/'stat'
    try:
        fields=path.read_text().rsplit(')',1)[1].split()
        return fields[0]!='Z'and int(fields[19])==receipt['start_ticks']
    except FileNotFoundError:return False


class AdoptedProcess:
    """Observe a verified existing owner; never invent its process exit code."""
    def __init__(self,receipt,completion,success_states,*,require_workers_exited=False):
        self.receipt,self.completion,self.success_states=receipt,Path(completion),success_states
        self.require_workers_exited=require_workers_exited
    def poll(self):
        if process_alive(self.receipt):return None
        if self.completion.exists():
            done=read(self.completion)
            if done.get('status')in self.success_states and(not self.require_workers_exited or done.get('all_owned_workers_exited')is True):return 0
        return 1
    @property
    def returncode(self):
        # Existing owner code uses Popen.poll() followed by .returncode.
        # This value is explicitly completion-derived, not an observed OS code.
        return self.poll()


def existing_arm(arm):
    path=ROOT/arm;receipt=read(path/'OWNER.json')
    if not process_alive(receipt):return None
    cmd=(Path('/proc')/str(receipt['pid'])/'cmdline').read_bytes().decode().split('\0')
    if (receipt.get('source')!=str(SOURCE)or receipt.get('arm')!=arm
            or 'tools.qwen_world_postcot.fresh_loss_owner'not in cmd
            or '--arm'not in cmd or cmd[cmd.index('--arm')+1]!=arm):
        raise ValueError('Cannot adopt an unrelated arm CPU owner')
    return AdoptedProcess(receipt,path/'COMPLETE.json',('STOPPED_AFTER_COMMON10K_SCREEN','FULL100K_AND_NAVTEST_COMPLETE'))


def existing_test(step):
    root=ROOT/'NAVTEST'/f'update{step:06d}';receipt=read(root/'LAUNCH.json');owner=read(root/'owner.json')
    if any(receipt[key]!=owner[key]for key in('pid','start_ticks'))or owner.get('source')!=str(SOURCE):
        raise ValueError('Active Test owner ancestry/source mismatch')
    pins=read(root/'PINNED.json')['models']
    if len(pins)!=1 or next(iter(pins.values()))['updates']!=step:raise ValueError('Changed active Test checkpoint')
    if process_alive(receipt):
        cmd=(Path('/proc')/str(receipt['pid'])/'cmdline').read_bytes().decode().split('\0')
        if ('tools.qwen_world_postcot.latest_snapshot_owner'not in cmd or '--root'not in cmd
                or Path(cmd[cmd.index('--root')+1]).resolve()!=root.resolve()):
            raise ValueError('Active Test PID no longer owns this request')
    elif not(root/'COMPLETE.json').exists():raise RuntimeError('Old Test owner exited without a complete request')
    previous=root/'CPU_PARENT_ADOPTION.json'
    if previous.exists():
        prior=read(previous);ancestor=root/f'CPU_PARENT_ADOPTION_ancestor_{prior["new_parent_pid"]}.json'
        if not ancestor.exists():atomic_json(ancestor,prior)
        elif read(ancestor)!=prior:raise ValueError('Changed previous Test adoption receipt')
    atomic_json(previous,dict(at=time.time(),new_parent_pid=os.getpid(),
        owner=owner,previous_parent_replaced=True,no_evaluation_worker_restarted=True))
    return AdoptedProcess(receipt,root/'COMPLETE.json',('COMPLETE',),require_workers_exited=True)


def recover_state():
    if not RECOVER: return None
    old = read(ROOT/'OWNER.json'); proc = Path('/proc')/str(old['pid'])/'stat'
    if proc.exists():
        fields = proc.read_text().split()
        if int(fields[21]) == old['start_ticks'] and fields[2] != 'Z':
            raise RuntimeError('Previous CPU owner is still alive; no duplicate handoff')
    if Path(old['source']).resolve() != SOURCE.resolve():
        raise ValueError('CPU handoff must preserve the immutable scientific worker source')
    from tools.qwen_world_postcot.train_il import source_identity
    hashes = source_identity()
    for rel, digest in hashes.items():
        if file_sha(SOURCE/rel) != digest:
            raise ValueError('CPU repair changes a scientific source dependency: '+rel)
    state = read(ROOT/'STATE.json')
    if state['source'] != str(SOURCE): raise ValueError('Changed native worker source in state')
    folder = ROOT/'owner_migrations'; folder.mkdir(exist_ok=True)
    receipt = dict(at=time.time(), previous_owner=old, previous_owner_sha256=file_sha(ROOT/'OWNER.json'),
        previous_state_sha256=file_sha(ROOT/'STATE.json'), runtime_source=str(SOURCE),
        controller_source=str(CONTROLLER_SOURCE), controller_sha256=file_sha(__file__),
        scientific_source_files=hashes, scientific_workers_signalled=False,
        reason='CPU-only shared-lease-aware admission repair; preserve every existing worker invocation')
    atomic_json(folder/(str(old['pid'])+'.json'), receipt)
    return state


def spawn_arm(arm, *, recover=False):
    path = ROOT/arm; env = dict(os.environ, WORLD_POSTCOT_ROOT=str(path))
    log = (path/('owner_admission_recovery.log' if recover else 'owner.log')).open('ab' if recover else 'xb')
    argv = [c.MODEL_PYTHON, '-u', '-m', 'tools.qwen_world_postcot.fresh_loss_owner', '--arm', arm,
            '--runtime-source', str(SOURCE)]
    if recover: argv += ['--recover-owner']
    process = subprocess.Popen(argv, cwd=CONTROLLER_SOURCE, env=env, stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    log.close(); previous = path/'LAUNCH.json'
    if previous.exists():
        old = read(previous); atomic_json(path/('LAUNCH_ancestor_'+str(old['pid'])+'.json'), old)
    atomic_json(previous, dict(pid=process.pid, start_ticks=int(Path(f'/proc/{process.pid}/stat').read_text().split()[21]),
        source=str(CONTROLLER_SOURCE), runtime_source=str(SOURCE)))
    return process


def command(config, arm, *, qualification=False, resume=None, full=False):
    args = [c.MODEL_PYTHON, '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
        '-m', 'tools.qwen_world_postcot.train_il', '--config', str(config), '--output',
        str(ROOT/'qualification/joint3' if qualification else c.IL_RUN),
        '--fresh-loss-registration', str(ROOT.parent/'REGISTRATION.json'), '--fresh-loss-arm', arm]
    if qualification: args += ['--qualification', '--qualification-joint', '--stop-after', '3']
    if resume: args += ['--resume', str(resume)]
    if full: args += ['--fresh-loss-stage', 'full', '--fresh-selection', str(ROOT.parent/'SELECTION.json')]
    return args


def qualified():
    q = ROOT/'qualification/joint3'; progress = read(q/'progress.json')
    if progress['updates'] != 3: raise ValueError('Incomplete real joint qualification')
    for rank in range(8):
        rows = [json.loads(line) for line in (q/f'metrics_rank{rank}.jsonl').read_text().splitlines()]
        if [r['updates'] for r in rows] != [1, 2, 3] or any(r['optimizer_skipped_steps'] or r['grad_norm'] <= 0 for r in rows):
            raise ValueError('Qualification needs three real finite optimizer updates on every rank')
        dtype = read(q/f'DTYPE_update000001_rank{rank}.json')
        if any(v is None for v in dtype['incoming_gradients'].values()):
            raise ValueError('Missing Qwen/memory/action/readout gradient')
    return dict(status='PASS', world=8, joint_updates=3, qualification_weights_used_for_formal=False,
        mean_update_seconds=sum(r['seconds'] for r in rows)/len(rows),
        max_peak_allocated_bytes=max(r['peak_allocated'] for r in rows), original_global_batch=32)


def launch_test(step, arm):
    checkpoint = c.IL_RUN/'checkpoints'/f'update{step:06d}'
    if not (checkpoint/'COMPLETE').exists(): return None
    root = ROOT/'NAVTEST'/f'update{step:06d}'; label = f'{arm}_IL_{step:06d}'
    if (root/'COMPLETE.json').exists(): return None
    if root.exists(): raise RuntimeError('Existing Test invocation needs explicit recovery; never duplicate')
    header = c.checkpoint_header(checkpoint, 'IL'); root.mkdir(parents=True)
    pin = dict(label=label, kind='IL', updates=step, checkpoint=str(checkpoint), parent_checkpoint=str(checkpoint),
        files_sha256=header['files_sha256'], native_identity_sha256=fingerprint(header['identity']),
        source=str(SOURCE), user_instruction='新损失从头IL；完整Navtest用于有限筛选；单候选seed2101')
    atomic_json(root/'PINNED.json', dict(schema='world_checkpoint_sweep_NAVTEST_v1', models={label: pin},
        max_concurrent_models=1, reference_evaluation=str(CONTROL), seed=2101, full_population=12146,
        Test_used_for_loss_method_selection=True, no_actor_update_from_Test=True))
    log = (root/'owner.log').open('xb')
    process = subprocess.Popen([c.MODEL_PYTHON, '-u', '-m', 'tools.qwen_world_postcot.latest_snapshot_owner',
        '--root', str(root), '--report-dir', str(root/'analysis'), '--asset-only'],
        cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    log.close(); atomic_json(root/'LAUNCH.json', dict(pid=process.pid,
        start_ticks=int(Path(f'/proc/{process.pid}/stat').read_text().split()[21])))
    return process


def arm_owner(arm):
    if ROOT.name != arm: raise ValueError('Explicit independent arm ROOT required')
    lock = (ROOT/'owner.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (ROOT/'OWNER.json').exists() and not RECOVER: raise ValueError('Owner already registered; no duplicate')
    recovered = recover_state()
    config = SOURCE/f'configs/qwen_world_postcot/fresh_loss_{arm.lower()}.yaml'
    registration = read_registration(ROOT.parent/'REGISTRATION.json', arm, config)
    atomic_json(ROOT/'OWNER.json', dict(pid=os.getpid(), start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
        source=str(SOURCE), registration_sha256=registration['sha256'], arm=arm))
    state = recovered or dict(source=str(SOURCE), stage='QUALIFICATION', jobs={}, tests_done=[]); test = None
    if recovered and state.get('active_Test') is not None:
        test=existing_test(state['active_Test'])
    try:
        while True:
            c.poll_jobs(state)
            failed = [j for j in state['jobs'].values() if j['status'] == 'FAILED']
            if failed: raise RuntimeError('Scientific/worker failure requires exact native review: '+str(failed[-1]))
            if any(value>=8 for value in state.get('resource_refusals',{}).values()):
                raise RuntimeError('Bounded resource admission retries exhausted; inspect leases before resuming')
            if state['stage'] == 'QUALIFICATION':
                complete = [j for j in state['jobs'].values() if j['name'] == 'qualification' and j['status'] == 'COMPLETE']
                if complete:
                    atomic_json(ROOT/'QUALIFICATION.json', qualified()); state['stage'] = 'PILOT'
                elif not c.active(state, name='qualification') and time.time() >= state.get('resource_retry_after', {}).get('qualification', 0):
                    route = training_route(state)
                    if route: c.dispatch(state, str(SOURCE), 'qualification', command(config, arm, qualification=True), route, role='qualification')
            if state['stage'] in ('PILOT', 'FULL'):
                full = state['stage'] == 'FULL'; limit = 100000 if full else 10000
                p = c.progress(c.IL_RUN)
                if not c.active(state, role='training') and (not p or p['updates'] < limit):
                    name = 'full_IL' if full else 'pilot_IL'
                    if time.time() >= state.get('resource_retry_after', {}).get(name, 0):
                        route = training_route(state)
                        if route: c.dispatch(state, str(SOURCE), name, command(config, arm, resume=p['checkpoint'] if p else None, full=full), route, role='training')
                if full: c.advance_refreshes(state, str(SOURCE), dict(refresh_minimum_free_MiB=32768))
                if test is not None and test.poll() is not None:
                    root = ROOT/'NAVTEST'/f'update{state["active_Test"]:06d}'
                    if test.returncode or not (root/'COMPLETE.json').exists(): raise RuntimeError('Full Test failed: '+str(root))
                    state['tests_done'].append(state.pop('active_Test')); test = None
                for step in ((5000, 10000, 20000, 40000, 60000, 80000, 100000) if full else (5000, 10000)):
                    if step in state['tests_done'] or test is not None: continue
                    test = launch_test(step, arm)
                    if test is not None: state['active_Test'] = step
                if not full and p and p['updates'] == 10000 and all(s in state['tests_done'] for s in (5000, 10000)):
                    state['stage'] = 'WAITING_SELECTION'
                elif full and p and p['updates'] == 100000 and all(s in state['tests_done'] for s in (5000, 10000, 20000, 40000, 60000, 80000, 100000)):
                    atomic_json(ROOT/'COMPLETE.json', dict(status='FULL100K_AND_NAVTEST_COMPLETE', state=state)); return
            if state['stage'] == 'WAITING_SELECTION' and (ROOT.parent/'SELECTION.json').exists():
                selection = read(ROOT.parent/'SELECTION.json')
                if arm in selection['selected']:
                    state['stage'] = 'FULL'
                else:
                    atomic_json(ROOT/'COMPLETE.json', dict(status='STOPPED_AFTER_COMMON10K_SCREEN', arm=arm,
                        checkpoint=read(c.IL_RUN/'LATEST.json'), selection_sha256=selection['sha256'])); return
            state['at'] = time.time(); atomic_json(ROOT/'STATE.json', state); time.sleep(60)
    except BaseException:
        atomic_json(ROOT/'FAILURE.json', dict(at=time.time(), error=traceback.format_exc(), state=state, automatic_retry=False)); raise


def main_owner():
    if ROOT.name != 'fresh_loss_IL_20261029' or TASK != 'qwen_world_fresh_loss_IL_20261029':
        raise ValueError('Explicit fresh-study root and task required')
    ROOT.mkdir(parents=True, exist_ok=True); lock = (ROOT/'owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (ROOT/'OWNER.json').exists() and not RECOVER: raise ValueError('No duplicate screen owner')
    recovered = recover_state()
    atomic_json(ROOT/'OWNER.json', dict(pid=os.getpid(), start_ticks=int(Path('/proc/self/stat').read_text().split()[21]), source=str(SOURCE)))
    state = recovered or dict(source=str(SOURCE), stage='CALIBRATION', jobs={}); children = {}
    if recovered and state['stage'] in ('FOUR_INDEPENDENT_SCREENS','SELECTED_FULL_CONTINUATIONS'):
        for arm in ARMS:
            if not (ROOT/arm/'COMPLETE.json').exists():
                children[arm]=existing_arm(arm)or spawn_arm(arm,recover=True)
    try:
        while True:
            c.poll_jobs(state)
            if any(j['status'] == 'FAILED' for j in state['jobs'].values()): raise RuntimeError('Calibration failure; preserve evidence')
            calibration = ROOT/'calibration/CALIBRATION.json'
            if state['stage'] == 'CALIBRATION':
                completed = [j for j in state['jobs'].values() if j['name'] == 'calibration' and j['status'] == 'COMPLETE']
                if completed:
                    cal = read(calibration); body = dict(cal)
                    if body.pop('sha256') != fingerprint(body) or cal['status'] != 'PASS': raise ValueError('Bad calibration seal')
                    reg = dict(schema='world_fresh_loss_IL_v1', seed=42, evaluation_seed=2101, pilot_updates=10000,
                        max_updates=100000, interface_updates=1000, global_batch=32,
                        initialization='official_generic_Qwen_plus_new_World_seed42', initial_driving_checkpoint=None,
                        initialization_student_sha256=cal['initialization_student_sha256'], arms=cal['arms'],
                        calibration_sha256=cal['sha256'], calibration_file=str(calibration),
                        configs={a: dict(path=str(SOURCE/f'configs/qwen_world_postcot/fresh_loss_{a.lower()}.yaml'),
                            sha256=file_sha(SOURCE/f'configs/qwen_world_postcot/fresh_loss_{a.lower()}.yaml')) for a in ARMS},
                        selection=dict(max_survivors=2, primary='full_NAVTEST_10000', exclude_if_delta_points_lte=-.5,
                            exclude_if_paired_CI_upper_lt=0., ties='fixed_arm_order', no_decision_from_DEV=True),
                        control_evaluation=str(CONTROL), observed_Test_not_blind=True,
                        common_bank='Same initial full TRAIN bank through10k; own-policy refresh begins only for selected continuations.',
                        budget=dict(pilot_updates_per_arm=10000, pilot_scene_exposures_per_arm=320000,
                            selected_full_cap=100000, qualification_and_evaluation_separate=True))
                    reg['sha256'] = fingerprint(reg); atomic_json(ROOT/'REGISTRATION.json', reg)
                    for arm in ARMS:
                        path = ROOT/arm; path.mkdir(exist_ok=False)
                        for name in ('data', 'assets', 'teachers'):
                            (path/name).symlink_to(BASE/name, target_is_directory=True)
                        children[arm] = spawn_arm(arm)
                    state['stage'] = 'FOUR_INDEPENDENT_SCREENS'
                elif not c.active(state, name='calibration') and time.time() >= state.get('resource_retry_after', {}).get('calibration', 0):
                    routes = c.choose_eval_routes(32768, maximum=48)
                    routes.sort(key=lambda r: (r['host'] != 'training-rl-zt2', r['host'], r['gpus']))
                    if routes: c.dispatch(state, str(SOURCE), 'calibration', [c.MODEL_PYTHON, '-u', '-m',
                        'tools.qwen_world_postcot.calibrate_fresh_losses', '--config', str(SOURCE/'configs/qwen_world_postcot/il.yaml'),
                        '--output', str(ROOT/'calibration')], routes[0], role='calibration')
            if state['stage'] == 'FOUR_INDEPENDENT_SCREENS':
                # Failed arms do not silently select the remaining variants.
                for arm in ARMS:
                    if (ROOT/arm/'FAILURE.json').exists(): raise RuntimeError('Arm requires review: '+arm)
                    if children[arm].poll() not in (None,0): raise RuntimeError('Arm owner exited unexpectedly: '+arm)
                ready = all((ROOT/arm/'NAVTEST/update010000/COMPLETE.json').exists() for arm in ARMS)
                if ready and not (ROOT/'SELECTION.json').exists():
                    from tools.qwen_world_postcot.summarize import summarize
                    reg = read(ROOT/'REGISTRATION.json'); results = {}; parents = {}
                    for arm in ARMS:
                        evaluation = ROOT/arm/f'NAVTEST/update010000/{arm}_IL_010000/evaluation'
                        result = summarize(evaluation, CONTROL)
                        atomic_json(ROOT/(arm+'_SCREEN_RESULT.json'), result)
                        results[arm] = dict(PDMS=result['PDMS'], scenes=result['scenes'], invalid_scenes=result['invalid_scenes'],
                            delta_points=100*result['paired']['mean_delta'], paired_CI_points=[100*x for x in result['paired']['CI95']])
                        cp = ROOT/arm/'runs/IL_full_seed42/checkpoints/update010000'
                        parents[arm] = dict(checkpoint=str(cp), files_sha256=file_sha(cp/'files.json'))
                    row = dict(status='LOCKED', registration_sha256=reg['sha256'], selected=select_survivors(results, reg),
                        results=results, parents=parents, at=time.time(), Test_selection_not_blind=True)
                    row['sha256'] = fingerprint(row); atomic_json(ROOT/'SELECTION.json', row)
                    state['stage'] = 'SELECTED_FULL_CONTINUATIONS'
            if state['stage'] == 'SELECTED_FULL_CONTINUATIONS' and all((ROOT/arm/'COMPLETE.json').exists() for arm in ARMS):
                atomic_json(ROOT/'COMPLETE.json', dict(status='SCREEN_AND_SELECTED_CONTINUATIONS_COMPLETE', selection=read(ROOT/'SELECTION.json'))); return
            state['at'] = time.time(); atomic_json(ROOT/'STATE.json', state); time.sleep(60)
    except BaseException:
        atomic_json(ROOT/'FAILURE.json', dict(at=time.time(), error=traceback.format_exc(), state=state, automatic_retry=False)); raise


def main():
    global SOURCE, RECOVER
    p = argparse.ArgumentParser(__doc__); p.add_argument('--arm', choices=ARMS)
    p.add_argument('--recover-owner',action='store_true');p.add_argument('--runtime-source');a = p.parse_args()
    if a.runtime_source: SOURCE = Path(a.runtime_source).resolve()
    RECOVER = a.recover_owner
    arm_owner(a.arm) if a.arm else main_owner()


if __name__ == '__main__': main()
