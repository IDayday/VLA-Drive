#!/usr/bin/env python3
"""Lead-only fixed official T1 2 -> 4 adapter; never registers or allocates work."""
import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import selectors
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

RUNTIME = Path('/mnt/project/VLA-Drive-st-temporal-readiness-inventory-20261017')
SOURCE = 'e85d11810bdf0240e7295d0994534f1df4ed21a4'
ROOT = Path('/mnt/project/st-world-policy-20261016/smoke/official_history_T1_native4_v1')
PYTHON = '/root/miniconda3/envs/ddp/bin/python'
REL = Path('reports/st_world_policy/20261017')
OBSERVER = Path('/mnt/project/st-world-policy-20261016/audits/correctness/audit_live_official_native_load.py')
OBSERVER_SHA = '5fce51c4c3e682ff4d3c33c958d7cacad49d05d372a7c46a9d39123a8dea5b5d'
HELPERS = {
    'full_future_metadata_preparation_v1/run_build_once.py': '8674aea1f2a3df894bb3c451287e6a14b5c20d59b01efe4883158f73a3690644',
    'spatial_probe_actual_preparation_v1/prepare_bindings.py': '31d2d302d5dd441df9251762e34cd9d1762afc3463cdabfe8dbf5b1f12577c84',
    'spatial_probe_actual_preparation_v1/supervise_once.py': 'd829cf48764b23b5fce08cdd6ea45c6d3e61e633763ad2cb876e5ab8d96d8902',
}
GiB = 1 << 30
LIMITS = dict(cpu_affinity=[2, 3, 4, 5], world_size=2, max_invocations=2,
              total_updates=4, total_scene_exposures=16, rss_bytes=96*GiB,
              cumulative_rchar_bytes=512*GiB, new_checkpoint_bytes=120*GiB,
              invocation_seconds=3600, save_cleanup_seconds=180)


def check(value, message):
    if not value:
        raise ValueError(message)


def pin_read(pin, decode=True):
    check(isinstance(pin, dict) and set(pin) == {'path', 'sha256'}, 'Exact pin required')
    p = Path(pin['path'])
    check(p.is_absolute() and '..' not in p.parts and not any(x.is_symlink() for x in (p, *p.parents)), 'Non-symlink absolute control required')
    before = p.stat()
    check(p.is_file() and before.st_size <= 4 << 20, 'Control size limit')
    raw = p.read_bytes()
    version = lambda v: (v.st_dev, v.st_ino, v.st_size, v.st_mtime_ns, v.st_ctime_ns)
    check(version(before) == version(p.stat()) and hashlib.sha256(raw).hexdigest() == pin['sha256'], 'Control/source changed')
    return json.loads(raw) if decode else raw


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_once(path, value):
    # One invocation marker, including failed invocations, cannot be reused.
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    check(len(raw) <= 4 << 20, 'Result control exceeds bound')
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('xb') as out:
        out.write(raw); out.flush(); os.fsync(out.fileno())
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()


def validate_recipe(reg, cfg, phase, proposed):
    check(phase in (2, 4), 'Only phase2 and phase4')
    expected = json.loads(json.dumps(proposed))
    expected['execution']['stop_after'] = phase
    actual = dict(cfg); continuation = actual.pop('continuation', None)
    check(actual == expected, 'Exact reviewed recipe required; no altered training timeline')
    check(reg.get('source_sha') == SOURCE and reg.get('execution') == cfg['execution']
          and reg.get('weights_usable_for_formal_init') is False
          and reg.get('run_id') == f'T1_official_real64_phase{phase}_v1', 'Wrong T1 identity/source/scope')
    check((phase == 2 and continuation is None) or (phase == 4 and isinstance(continuation, dict)), 'Explicit real phase2 parent required only in phase4')


def check_prior(a, cfg):
    if a['phase'] == 2:
        check('previous_phase' not in a and 'parent_receipt' not in a, 'Phase2 cannot import a parent')
        return 0, None
    check(a['previous_phase']['path'] == str(ROOT/'phase2/RESULT.json')
          and Path(a['parent_receipt']['path']).parent.parent == ROOT/'runs/T1_official_real64_phase2_v1/checkpoints', 'Only this qualification phase2 namespace is a parent')
    previous = pin_read(a['previous_phase']); receipt = pin_read(a['parent_receipt'])
    check(previous.get('schema') == 'st_T1_native4_execution_result_v1'
          and previous.get('status') == 'COMPLETE' and previous.get('phase') == 2
          and previous.get('cleanup', {}).get('live_owned_remaining') == 0
          and previous.get('completed') == 2 and previous.get('native_receipt') == a['parent_receipt'], 'Actual successful phase2 adapter/native receipts required')
    check(previous['registration'] == cfg['continuation']['parent_registration']
          and cfg['continuation']['parent_receipt_sha256'] == a['parent_receipt']['sha256']
          and cfg['continuation']['parent_tag'] == Path(a['parent_receipt']['path']).parent.name
          and receipt.get('state', {}).get('completed') == 2
          and receipt.get('world_size') == 2, 'Phase4 parent mismatch')
    used = previous['cumulative_rchar_bytes']
    check(type(used) is int and 0 <= used < LIMITS['cumulative_rchar_bytes'], 'No remaining shared read budget')
    return used, receipt


def controls(args, child=False):
    a = pin_read({'path': args.admission, 'sha256': args.admission_sha256})
    check(a.get('schema') == 'st_T1_native4_execution_admission_v1'
          and a.get('status') == 'APPROVED_BOUNDED_T1_ONLY' and a.get('limits') == LIMITS
          and a.get('hostname') == socket.gethostname(), 'Fresh exact Lead admission required')
    check(a.get('runtime') == {'path': str(RUNTIME), 'commit': SOURCE}, 'Frozen e85 runtime required')
    check(a['adapter']['path'] == str(Path(__file__).absolute()), 'Reviewed adapter path required')
    pin_read(a['adapter'], False)
    now = time.time()
    check(all(type(a.get(k)) in (int, float) and math.isfinite(a[k])
              for k in ('issued_at_unix', 'latest_start_unix', 'valid_until_unix')), 'Finite resource window required')
    check(type(a.get('valid_until_unix')) in (int, float)
          and a.get('issued_at_unix', float('inf')) <= now
          and now <= a.get('latest_start_unix', 0)
          and now <= a['valid_until_unix']
          and (child or now+3780 <= a['valid_until_unix']), 'Lease does not cover save/cleanup')
    check(set(LIMITS['cpu_affinity']).issubset(os.sched_getaffinity(0)), 'CPUs 2-5 unavailable')
    for role in ('correctness', 'resource'):
        review = pin_read(a['reviews'][role])
        check(review.get('status') == 'PASS_BOUNDED_T1_ADAPTER_ONLY'
              and review.get('adapter_sha256') == a['adapter']['sha256']
              and review.get('reviewer') not in (None, '', '/root/st_temporal_readiness_inventory'), 'Independent adapter reviews required')
        if role == 'resource':
            check(review.get('new_checkpoint_reservation_bytes') == LIMITS['new_checkpoint_bytes']
                  and review.get('checkpoint_root') == str(ROOT/'runs')
                  and review.get('uses_old_profile_reservation') is False
                  and review.get('valid_until_unix', 0) >= a['valid_until_unix'], 'Separate new 120GiB reservation must cover both phases')
    reg = pin_read(a['registration'])
    # Existing trainer still performs complete registration/environment/lease verification.
    for pin in reg['artifacts'].values():
        pin_read(pin)
    cfg = pin_read(reg['artifacts']['config'])
    proposed = json.loads((RUNTIME/REL/'official_T1_native4_preparation_v1/PROPOSED_CONFIG_STEP2.json').read_text())
    validate_recipe(reg, cfg, a['phase'], proposed)
    lease = pin_read(reg['artifacts']['resource_lease'])
    check(lease.get('world_size') == 2 and lease.get('valid_until_unix', 0) >= a['valid_until_unix'], 'Trainer lease must cover adapter window')
    devices = lease['cuda_visible_devices_by_host'].get(socket.gethostname(), '')
    check(len(devices.split(',')) == 2 and len(set(devices.split(','))) == 2, 'Exactly two leased devices required')
    prior, parent = check_prior(a, cfg)
    for name, digest in HELPERS.items():
        pin_read({'path': str(RUNTIME/REL/name), 'sha256': digest}, False)
    pin_read({'path': str(OBSERVER), 'sha256': OBSERVER_SHA}, False)
    check(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=RUNTIME, text=True).strip() == SOURCE, 'Runtime source changed')
    check(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=RUNTIME).strip(), 'Runtime is dirty')
    return a, reg, cfg, prior, parent, devices


@contextmanager
def observe_restore(trainer, engine_type, audit, observer, capture_rng):
    """Reuse existing live comparator exactly; never replace backward or step."""
    original_load, original_restore = engine_type.load_checkpoint, trainer.restore_rng
    signature = inspect.signature(original_load)
    pending_restore = [False]
    def loaded(engine, *pos, **kw):
        values = signature.bind(engine, *pos, **kw); values.apply_defaults(); v = values.arguments
        check(v['load_module_strict'] and v['load_optimizer_states'] and v['load_lr_scheduler_states']
              and not v['load_module_only'] and v['custom_load_fn'] is None, 'Incomplete native restore')
        result = original_load(engine, *pos, **kw)
        observer.collective_audit(lambda: audit.after_load(engine, Path(v['load_dir'])/v['tag'], result), 'T1 live native restore before update3')
        pending_restore[0] = True
        return result
    def restored(state, generators):
        original_restore(state, generators)
        if pending_restore[0]:
            observer.collective_audit(lambda: audit.after_restore(state, generators, capture_rng), 'T1 restored RNG before update3')
            pending_restore[0] = False
    engine_type.load_checkpoint, trainer.restore_rng = loaded, restored
    try:
        yield
    finally:
        engine_type.load_checkpoint, trainer.restore_rng = original_load, original_restore


def rank_main(args):
    a, reg, cfg, prior, parent, devices = controls(args, child=True)
    check(os.environ.get('CUDA_VISIBLE_DEVICES') == devices and os.environ.get('WORLD_SIZE') == '2', 'Wrong torchrun topology')
    sys.path.insert(0, str(RUNTIME))
    from tools.st_world_policy import train_world as trainer
    check(Path(trainer.__file__).resolve().is_relative_to(RUNTIME), 'Foreign trainer')
    call = SimpleNamespace(registration=a['registration']['path'], approval=None, resume=False, acknowledge_stop=False)
    if a['phase'] == 2:
        trainer.run(call)
        return 0
    observer = load_module('t1_existing_native_observer', OBSERVER)
    from deepspeed.runtime.engine import DeepSpeedEngine
    from tools.foresight.student_state import capture_rng
    audit = observer.LiveAudit(ROOT/'phase4/LIVE_RESTORE', rank=int(os.environ['RANK']),
        run=ROOT/'runs'/reg['run_id'], expected_step=2, identity=parent['state']['identity'], world_size=2)
    try:
        with observe_restore(trainer, DeepSpeedEngine, audit, observer, capture_rng):
            trainer.run(call)
        check(audit.load_calls == audit.restore_calls == 1
              and audit.report['status'] == 'LIVE_MODEL_OPTIMIZER_AND_RNG_MATCH', 'Actual pre-update3 evidence incomplete')
        audit.report.update(observer_scope='Actual restored state and RNG before update3; subsequent updates3-4 allowed',
            source_payload_checksum_validation='Original e85 trainer resolve_extension verifies every native file before load; reused observer compares actual live tensors and restored RNG.',
            zero_update_exit_audit='NOT_RUN_OUTSIDE_TWO_INVOCATION_BUDGET')
        audit.persist()
        # Deliberately no observer.finish(): its zero-update contract is different.
    except BaseException as exc:
        audit.report.update(status='FAILED', error=repr(exc)); audit.persist(); raise
    return 0


def proc_io(pid=None):
    with open('/proc/'+str(pid or os.getpid())+'/io') as stream:
        return {k: int(v) for k, v in (line.split(':') for line in stream)}


def live_owned_usage(owner, s):
    """Torchrun's unreaped rank IO is not yet in its anchor: include live members.

    A reap concurrent with sampling can over-count (safe early stop). Final charge
    uses the reused post-wait accounting, which includes descendants exactly once.
    """
    rss = reads = 0
    for known, _fd in list(owner.members.values()):
        try:
            before = s.proc_identity(known['pid'])
            check(owner.same(known, before), 'Owned member identity changed')
            value = proc_io(known['pid'])
            with open(f"/proc/{known['pid']}/statm") as stream:
                memory = int(stream.read(4096).split()[1])*os.sysconf('SC_PAGE_SIZE')
            check(owner.same(known, s.proc_identity(known['pid'])), 'Owned member changed during monitor')
            reads += value['rchar']; rss += memory
        except (FileNotFoundError, ProcessLookupError):
            continue
    with open('/proc/self/statm') as stream:
        rss += int(stream.read(4096).split()[1])*os.sysconf('SC_PAGE_SIZE')
    return rss, reads


def checkpoint_bytes(root):
    total = count = 0
    if not root.exists():
        return 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in (*dirs, *files):
            path = Path(directory)/name; count += 1
            check(count <= 10000 and not path.is_symlink(), 'Checkpoint tree bound/symlink')
            if path.is_file(): total += path.stat().st_size
    return total


def monitor(command, env, fd, s, core, remaining):
    """Narrow adaptation of 8674 monitor: topology, 180s grace, native log format."""
    baseline = proc_io(); started = time.monotonic(); buffers = {'stdout': bytearray(), 'stderr': bytearray()}
    child = owner = leader = None; finished = False; failure = None; peak = 0; term_at = None
    selector = selectors.DefaultSelector()
    with s.lifetime_scope() as stopped:
        try:
            child = subprocess.Popen(['taskset', '-c', '2-5', *command], cwd=RUNTIME, env=env,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=True, pass_fds=(fd,))
            leader = s.pidfd_open(child.pid); owner = s.OwnedSession(child)
            for name, stream in (('stdout', child.stdout), ('stderr', child.stderr)):
                os.set_blocking(stream.fileno(), False); selector.register(stream, selectors.EVENT_READ, name)
            next_sample = 0
            while True:
                active = owner.discover(); done = owner.leader_done()
                if done and active:
                    active = owner.discover()
                    if active: failure = failure or 'OWNED_DESCENDANTS_AFTER_LEADER_EXIT'
                if stopped['signal']: failure = failure or 'SUPERVISOR_SIGNAL'
                if not active and not selector.get_map():
                    terminal = core.owned_proc_io(owner, s); break
                for key, _ in selector.select(.25):
                    data = os.read(key.fd, 65536)
                    if not data: selector.unregister(key.fileobj); continue
                    if not core.append_pipe(buffers, key.data, data): failure = failure or 'LOG_LIMIT'
                now = time.monotonic()
                if now >= next_sample:
                    next_sample = now+1
                    memory, reads = live_owned_usage(owner, s); peak = max(peak, memory)
                    if memory > LIMITS['rss_bytes']: failure = failure or 'RSS_LIMIT'
                    if reads+proc_io()['rchar']-baseline['rchar'] > remaining: failure = failure or 'READ_LIMIT'
                    if checkpoint_bytes(ROOT/'runs') > LIMITS['new_checkpoint_bytes']: failure = failure or 'CHECKPOINT_LIMIT'
                    if now-started >= LIMITS['invocation_seconds']: failure = failure or 'WALL_LIMIT'
                if failure and term_at is None: owner.send(signal.SIGTERM); term_at = now
                if term_at and now-term_at >= 170: owner.send(signal.SIGKILL)
                if term_at and now-term_at >= 180: raise RuntimeError('Owned cleanup exceeded 180s')
            before = proc_io()
            code = owner.finish(terminate=bool(failure), deadline=term_at+180 if term_at else None)
            finished = True
            accounting = core.terminal_read_accounting(baseline, before, proc_io(), terminal['rchar'], remaining, final_child_available=True)
            if accounting['status'] != 'COMPLETE': failure = failure or 'IO_ACCOUNTING_INCOMPLETE'
            return dict(returncode=code, failure=failure, peak_rss_bytes=peak,
                elapsed_seconds=time.monotonic()-started, accounting=accounting,
                stdout=buffers['stdout'].decode('utf8', 'replace'), stderr=buffers['stderr'].decode('utf8', 'replace'),
                cleanup={'verified_processes': owner.verified, 'live_owned_remaining': 0})
        finally:
            try:
                if owner is not None and not finished:
                    owner.finish(terminate=True, deadline=time.monotonic()+180)
                elif owner is None and child is not None: s.abort_spawn_setup(child, leader)
            finally:
                selector.close()
                if owner is not None: owner.close()
                if leader is not None: os.close(leader)
                if child is not None: child.stdout.close(); child.stderr.close()


def parent_main(args):
    start_io = proc_io()
    a, reg, cfg, prior, parent, devices = controls(args)
    helper_dir = RUNTIME/REL/'spatial_probe_actual_preparation_v1'
    sys.path.insert(0, str(helper_dir)); import supervise_once as s
    core = load_module('t1_accounting_core', RUNTIME/REL/'full_future_metadata_preparation_v1/run_build_once.py')
    s.shared_mount()  # original fixed shared-mount/global-flock check
    os.sched_setaffinity(0, LIMITS['cpu_affinity'])
    phase_root = ROOT/f"phase{a['phase']}"
    s.check_ancestry(phase_root)
    phase_root.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock = s.acquire(ROOT/'T1_NATIVE4_EXECUTION.lock', stack)
        for gpu in devices.split(','):
            s.acquire(Path('/tmp')/f'foresight_gpu_{int(gpu)}.lock', stack)
        lease = pin_read(reg['artifacts']['resource_lease'])
        uuids = lease['gpu_uuids_by_host'][socket.gethostname()]
        check(len(uuids) == 2, 'Two current leased UUIDs required')
        for gpu, uuid in zip(devices.split(','), uuids):
            s.fresh_gpu({'index': int(gpu), 'uuid': uuid})
        check(not (ROOT/'runs'/reg['run_id']).exists(), 'Fresh invocation only; no automatic retry/resume')
        if a['phase'] == 2:
            check(not (ROOT/'phase4').exists() and checkpoint_bytes(ROOT/'runs') == 0, 'Phase2 requires empty new qualification run namespace')
        marker = dict(schema='st_T1_native4_execution_result_v1', phase=a['phase'], registration=a['registration'],
            admission={'path': args.admission, 'sha256': args.admission_sha256}, source_sha=SOURCE, limits=LIMITS,
            scientific_result=False, weights_usable_for_formal_init=False, automatic_retry=False)
        atomic_once(phase_root/'STARTED.json', marker)
        observed = None
        try:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices, PYTHONPATH=str(RUNTIME),
                PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                TOKENIZERS_PARALLELISM='false', CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUNBUFFERED='1')
            command = [PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                str(Path(__file__).absolute()), '--rank-child', '--admission', args.admission, '--admission-sha256', args.admission_sha256]
            overhead = proc_io()['rchar']-start_io['rchar']
            check(prior+overhead < LIMITS['cumulative_rchar_bytes'], 'Read budget exhausted before launch')
            observed = monitor(command, env, lock.fileno(), s, core, LIMITS['cumulative_rchar_bytes']-prior-overhead)
            result_path = ROOT/'runs'/reg['run_id']/'RESULT.json'
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            check(observed['returncode'] == 0 and observed['failure'] is None, 'Bounded invocation unsuccessful')
            check(result.get('completed') == a['phase'] and result.get('new_optimizer_updates') == 2
                  and result.get('exposure') == a['phase']*4, 'Wrong actual updates/exposure')
            native = ROOT/'runs'/reg['run_id']/'checkpoints'/result['tag']/'COMPLETE.json'
            raw = native.read_bytes(); receipt = json.loads(raw)
            check(receipt['state']['completed'] == a['phase'] and receipt['world_size'] == 2, 'Missing exact native endpoint')
            if a['phase'] == 4:
                for rank in range(2):
                    report = json.loads((phase_root/'LIVE_RESTORE'/f'rank{rank}.json').read_text())
                    check(report['status'] == 'LIVE_MODEL_OPTIMIZER_AND_RNG_MATCH' and report['expected_step'] == 2, 'Pre-update3 live audit absent')
            charged = overhead+core.charge_after_reap(observed['accounting'], proc_io())
            check(prior+charged <= LIMITS['cumulative_rchar_bytes'] and checkpoint_bytes(ROOT/'runs') <= LIMITS['new_checkpoint_bytes'], 'Cumulative budget exceeded')
            value = {**marker, 'status': 'COMPLETE', 'completed': result['completed'], 'new_optimizer_updates': 2,
                'cumulative_rchar_bytes': prior+charged, 'resources': observed, 'cleanup': observed['cleanup'],
                'native_receipt': {'path': str(native), 'sha256': hashlib.sha256(raw).hexdigest()},
                'terminal_reload': 'NOT_RUN_OUTSIDE_TWO_INVOCATION_BUDGET'}
            atomic_once(phase_root/'RESULT.json', value)
            return 0
        except BaseException as exc:
            # Failure consumes this invocation; retained output is not an automatic retry ticket.
            atomic_once(phase_root/'RESULT.json', {**marker, 'status': 'FAILED_NO_AUTO_RETRY', 'error': repr(exc),
                'resources': observed, 'cleanup': observed['cleanup'] if observed else {'status': 'SEE_RETAINED_EXCEPTION'},
                'cumulative_rchar_bytes': LIMITS['cumulative_rchar_bytes'], 'terminal_reload': 'NOT_RUN'})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission', required=True); parser.add_argument('--admission-sha256', required=True)
    parser.add_argument('--rank-child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    return rank_main(args) if args.rank_child else parent_main(args)


if __name__ == '__main__':
    raise SystemExit(main())
