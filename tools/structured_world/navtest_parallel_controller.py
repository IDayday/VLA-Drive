"""One queue lock, concurrent independent Navtest banks, unchanged native workers.

Adoption keeps already-running GPU workers and their one canonical CPU scorer.
New evaluations recheck measured capacity and exact occupant ownership. Counts
of physical GPUs and model processes are reported separately when sharing.
"""
import concurrent.futures
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import socket
import threading
import time


class PreparationReservations:
    """Reserve GPUs during CPU loading, before NVML can observe the model.

    Once all workers have completed one scene, measured CUDA occupancy takes
    over admission. This does not serialize the independent inference banks.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.pending = {}
        self.scorers = {}

    def select(self, key, scan, unavailable):
        with self.lock:
            # Scan under the same lock as release: a snapshot taken before a
            # peer allocated CUDA memory must not be admitted after its release.
            routes = scan()
            occupied = {uuid for other, value in self.pending.items() if other != key
                        for uuid in value['uuids']}
            routes = [route for route in routes if route['uuid'] not in occupied]
            if not routes:
                raise unavailable('Eligible GPUs are currently preparing another evaluation')
            self.pending[key] = dict(uuids={route['uuid'] for route in routes}, attempt=None)
            return routes

    def launched(self, key, attempt):
        with self.lock:
            if key in self.pending:
                self.pending[key]['attempt'] = attempt

    def release_ready(self, output):
        with self.lock:
            for key, value in list(self.pending.items()):
                if not value['attempt']:
                    continue
                root = output/'jobs'/key/'attempts'/value['attempt']/'shards'
                reports = [json.loads(path.read_text()) for path in root.glob('*.json')]
                if (len(reports) == len(value['uuids'])
                        and {r.get('actual_CUDA_UUID') for r in reports} == value['uuids']
                        and all(r.get('status') == 'COMPLETE' or
                                (r.get('status') == 'INFERENCE' and r.get('scenes_done', 0) >= 1)
                                for r in reports)):
                    del self.pending[key]

    def release(self, key):
        with self.lock:
            self.pending.pop(key, None)
            self.scorers.pop(key, None)

    def score_budget(self, key, probe, unavailable):
        with self.lock:
            budget = probe()
            peers = sum(count for other, count in self.scorers.items() if other != key)
            available = int(budget['effective_capacity_CPUs']-budget['reserved_host_CPUs']-peers)
            if available < 1:
                raise unavailable('Active canonical scorer pools fill the current CPU budget')
            budget['other_active_score_workers'] = peers
            budget['workers'] = min(budget['workers'], available)
            self.scorers[key] = budget['workers']
            return budget


def process_present(pid, start):
    try:
        fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == str(start)
    except FileNotFoundError:
        return False


def accepted_receipt(output, request):
    """Reuse the sole owner's prior verified completion, not repeated file scans.

Every new bank still goes through valid_score_completion before publication.
Explicit --verify-complete retains full payload/CSV verification on demand.
"""
    path = output/'completed'/(request['key']+'.json')
    if not path.exists(): return False
    saved = json.loads(path.read_text())
    if saved.get('valid') is not True or saved['request'] != request:
        raise ValueError('Completed request receipt differs')
    return True


def adopt(args, m, record):
    req = record['request']; destination = args.output/'jobs'/req['key']
    attempt = record['attempt']; count = len(record['routes'])
    paths = [destination/'attempts'/attempt/'shards'/f'shard_{i:03d}.json' for i in range(count)]
    while True:
        if (destination/'INFERENCE_FAILED.json').exists():
            raise RuntimeError('Adopted inference has a recorded failure')
        reports = [json.loads(p.read_text()) for p in paths if p.exists()]
        if len(reports) == count and all(r['status'] == 'COMPLETE' for r in reports):
            break
        m.atomic(args.output/'STATE.json', dict(status='ADOPTED_INFERENCE_RUNNING', request=req,
                  workers=count, scenes_done=sum(r['scenes_done'] for r in reports), observed_utc=m.now()))
        time.sleep(5)
    # Remote CUDA process identities and local SSH launcher identities were
    # captured before the handoff. The former owner alone was terminated.
    while any(process_present(r['pid'], r['start_ticks']) for r in record['launchers']):
        time.sleep(1)
    proofs = m.worker_exit_proofs(destination)
    if not (destination/'COMPLETE.json').exists():
        m.finish_predictions(destination, count, attempt)
    m.atomic(args.output/'STATE.json', dict(status='ADOPTED_CANONICAL_SCORING',request=req,observed_utc=m.now()))
    while process_present(record['scorer']['pid'], record['scorer']['start_ticks']):
        time.sleep(2)
    if not m.valid_score_completion(destination, req):
        raise ValueError('Adopted native workers/scorer did not produce a valid complete bank')
    m.atomic(destination/'OWNED_WORKERS_EXITED.json', dict(key=req['key'],
             training_identity=req['training_identity'],checkpoint_files_sha256=req['checkpoint_files_sha256'],
             all_gpu_launchers_exited=True,scorer_exited=True,worker_exit_proofs=proofs,
             launchers=record['launchers'],scorer=record['scorer'],
             evidence='Intentional CPU-owner handoff: complete native reports, exact remote exit proofs and launcher/scorer PID-start absence; former parent exit statuses unavailable',
             observed_utc=m.now(),routing_tool_sha256=m.EXECUTED_TOOL_SHA256))
    m.atomic(args.output/'completed'/(req['key']+'.json'),dict(request=req,
             score_complete=str(destination/'scores/COMPLETE.json'),valid=True,completed_utc=m.now(),
             adoption_record=str(args.adopt_running)))


def serve_parallel(args, m):
    import navtest_cadence as cadence
    m.authorized(); policy = cadence.load_policy(args.cadence_policy)
    local = threading.local(); original_atomic = m.atomic
    reservations = PreparationReservations()
    original_refresh = m.refresh_routes
    def reserved_refresh(options, request):
        return reservations.select(request['key'], lambda: original_refresh(options, request),
                                   m.ResourcesUnavailable)
    m.refresh_routes = reserved_refresh
    original_budget = m.cpu_budget
    def combined_budget(requested):
        return reservations.score_budget(local.key, lambda: original_budget(requested),
                                         m.ResourcesUnavailable)
    m.cpu_budget = combined_budget
    def routed_atomic(path, value):
        key = getattr(local, 'key', None)
        if key and Path(path) == args.output/'STATE.json':
            path = args.output/'job_states'/(key+'.json')
            if value.get('status') == 'INFERENCE_RUNNING':
                reservations.launched(key, value['attempt'])
        original_atomic(path,value)
    m.atomic = routed_atomic
    with (args.output/'controller.lock').open('a') as lock, ExitStack() as cleanup:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original_atomic(args.output/'CONTROLLER.json', dict(pid=os.getpid(),host=socket.gethostname(),
            started_utc=m.now(),tool=str(m.SCRIPT),tool_sha256=m.EXECUTED_TOOL_SHA256,
            cadence_policy=str(args.cadence_policy),cadence_policy_sha256=m.digest(args.cadence_policy),
            routes_file=str(args.routes),mode='parallel_checkpoint_evaluations',
            helper_sha256=m.digest(__file__)))
        observer = cadence.Observer(policy,args.output,m.request,m.valid_score_completion)
        cleanup.callback(observer.close)
        # Pending banks are few (four registered G groups); CPU threads do not
        # fix GPU/shard counts. Actual GPU admission remains per request.
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        cleanup.callback(pool.shutdown, wait=True)
        active = {}; failed = set(); next_retry = {}; preparations = {}
        def run(req, inherited=None):
            local.key = req['key']
            try:
                dependency = preparations.get(req['key'])
                if dependency:
                    while not Path(dependency['receipt']).exists():
                        if not process_present(dependency['pid'], dependency['start_ticks']):
                            raise RuntimeError('Existing qualification owner exited without acceptance')
                        m.atomic(args.output/'STATE.json',dict(status='EXISTING_QUALIFICATION_RUNNING',
                                 request=req,observed_utc=m.now(),dependency=dependency))
                        time.sleep(5)
                if inherited: adopt(args,m,inherited)
                else: m.execute(args,req)
                return 'COMPLETE'
            finally:
                reservations.release(req['key'])
                local.key = None
        if args.adopt_running:
            record = json.loads(args.adopt_running.read_text())
            assert record['old_controller_exited'] is True
            preparations.update(record.get('preparations',{}))
            req = record['request']
            if not accepted_receipt(args.output,req):
                budget = json.loads((args.output/'jobs'/req['key']/'CPU_BUDGET.json').read_text())
                reservations.scorers[req['key']] = budget['workers']
                active[req['key']] = pool.submit(run,req,record)
        while True:
            m.authorized()
            reservations.release_ready(args.output)
            if observer.error:
                raise RuntimeError('Cadence capture failed') from observer.error
            for key, future in list(active.items()):
                if not future.done(): continue
                del active[key]
                try:
                    future.result()
                except m.ResourcesUnavailable as error:
                    next_retry[key] = time.monotonic()+30
                    original_atomic(args.output/'job_states'/(key+'.json'),dict(
                        status='WAITING_FOR_AVAILABLE_RESOURCES',reason=str(error),observed_utc=m.now()))
                except BaseException as error:
                    failed.add(key)
                    original_atomic(args.output/'job_states'/(key+'.json'),dict(
                        status='FAILED',error=repr(error),observed_utc=m.now()))
            requests = sorted((json.loads(p.read_text()) for p in (args.output/'requests').glob('*.json')),
                              key=lambda r:(r['priority'],r['key']))
            for req in requests:
                key = req['key']
                if key in active or key in failed or not cadence.eligible(policy,req): continue
                if accepted_receipt(args.output,req): continue
                if time.monotonic() < next_retry.get(key,0): continue
                # A foreign/scientific failure is never an automatic retry.
                if (args.output/'jobs'/key/'INFERENCE_FAILED.json').exists():
                    failed.add(key); continue
                active[key] = pool.submit(run,req)
            original_atomic(args.output/'STATE.json',dict(
                status='PARALLEL_EVALUATIONS' if active else 'IDLE_AWAITING_REQUESTS',
                active_keys=sorted(active),failed_keys=sorted(failed),observed_utc=m.now(),
                GPU_count='dynamic; distinct physical UUIDs and model processes counted separately',
                per_job_states=str(args.output/'job_states')))
            time.sleep(5)
