"""Bounded off-host FP32 pool tools. No automatic Base stop or migration."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from ..contracts import require
from ..io import atomic_json, file_hash, read_json
from .plan import load_session, prepare_session
from .pool import Pool


def parser():
    p = argparse.ArgumentParser(__doc__); sub = p.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('plan', help='bind parent state and bounded original-sampler lookahead')
    for name in ('full-plan','parent-checkpoint','root','qualification'): plan.add_argument('--'+name, required=True)
    plan.add_argument('--window-updates', type=int, default=64); plan.add_argument('--lookahead', type=int, default=3)
    plan.add_argument('--capacity-gib', type=float, default=512); plan.add_argument('--reserve-gib', type=float, default=1700)
    plan.add_argument('--reuse-cache', action='append', default=[])
    start = sub.add_parser('prewarm', help='prepare windows from the bound checkpoint; no training launch')
    start.add_argument('--root', required=True)
    produce = sub.add_parser('produce', help='remote frozen-only builder; automatic bounded owned-file eviction')
    for name in ('root','config','device','hold-file'): produce.add_argument('--'+name, required=True)
    produce.add_argument('--max-windows', type=int, required=True)
    produce.add_argument('--max-seconds', type=int, required=True)
    produce.add_argument('--max-gib-per-window', type=float, required=True)
    produce.add_argument('--rate-mib-s', type=float, default=64)
    produce.add_argument('--worker', type=int, default=0)
    produce.add_argument('--workers', type=int, default=1)
    produce.add_argument('--share-with-pid', type=int)
    produce.add_argument('--share-start-ticks', type=int)
    produce.add_argument('--minimum-free-mib', type=int, default=30000)
    status = sub.add_parser('status', help='read current window coverage and original-sequence cache hit counts')
    status.add_argument('--root', required=True)
    verify = sub.add_parser('verify', help='bounded record/ordering/resume IO checks, not training throughput')
    verify.add_argument('--root', required=True); verify.add_argument('--max-updates', type=int, required=True)
    verify.add_argument('--output', required=True)
    for name in ('train','resume'):
        t = sub.add_parser(name, help='existing Base trainer with remote cache reads only; opt-in qualification')
        for arg in ('config','runtime','rolling-config','output','hold-file'): t.add_argument('--'+arg, required=True)
        t.add_argument('--parent-checkpoint', required=name=='train')
        t.add_argument('--mode', choices=['bounded','continuation'], required=True)
        t.add_argument('--stop-after', type=int, required=True)
        t.add_argument('--warmup', type=int, default=10)
        t.add_argument('--qualification'); t.add_argument('--save', action='store_true')
        t.add_argument('--fingerprint', action='store_true'); t.add_argument('--probe', action='store_true')
        t.add_argument('--allow-fallback', action='store_true')
        t.add_argument('--checkpoint-every', type=int, default=1000)
    return p


def make_pool(root, config=None):
    session, full = load_session(root)
    # Explicit source contract, also used when only inspecting a cache. The
    # original function is short but must not be reimplemented with a new RNG.
    from ..config import load_config
    source_config = load_config(config or str(Path(__file__).resolve().parents[2]/'configs/iqe/fp32_cache/source.yaml'))
    contract = read_json(source_config['s0']['contract'])
    require(contract['framework_contract_hash'] == full['framework_contract_hash'], 'source contract mismatch')
    from ..query_base import bind_source
    bind_source(contract['source_root'], contract['source_commit'])
    from tools.ddpolicy_vehicle.training_state import epoch_batches
    return Pool(root, session, full, epoch_batches)


def admit_gpu(args):
    from .placement import producer_placement
    result = producer_placement()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    require(visible.isdigit() and args.device == 'cuda:0', 'one explicit physical GPU per producer required')
    lines = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,uuid','--format=csv,noheader,nounits'], text=True)
    index, free, uuid = next(line.split(',') for line in lines.splitlines() if line.split(',')[0].strip() == visible)
    require(int(free) >= args.minimum_free_mib >= 24000, 'producer memory headroom unavailable')
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'], text=True)
    owners = {int(line.split(',')[0]) for line in apps.splitlines() if line.split(',')[1].strip() == uuid.strip()}
    if owners:
        require(owners == {args.share_with_pid} and args.share_start_ticks is not None, 'unknown GPU owner; no sharing')
        process = Path('/proc') / str(args.share_with_pid)
        ticks = int((process/'stat').read_text().rsplit(')', 1)[1].split()[19])
        require(ticks == args.share_start_ticks, 'shared owner PID reused')
        command = (process/'cmdline').read_bytes().replace(b'\x00', b' ').decode()
        require('tools/structured_world/train_vla.py' in command and 'G2_EVENT_UNIFORM' in command,
                'only explicitly authorized G2/cache sharing accepted')
        policy = read_json('/mnt/project/server_dispatch_policy.json')
        require(policy['task_authorizations']['iqe_v1'].get('shared_GPU_deployment_authorized') is True,
                'shared GPU authorization absent')
        result['shared_owner'] = {'pid':args.share_with_pid,'start_ticks':ticks,'cmdline':command}
    result.update(physical_gpu=visible, free_mib=int(free), max_rate_mib_s=args.rate_mib_s)
    return result


def run_producer(args, pool):
    from .placement import producer_placement
    from .pool import build_window
    from ..vision_cache.cli import load_front, hold_check
    from ..config import load_config
    require(1 <= args.max_windows <= 10000 and 1 <= args.max_seconds <= 86400 and
            0 < args.rate_mib_s <= 128 and 0 < args.max_gib_per_window <= 512, 'bounded producer budget required')
    producer_placement(); hold_check(args.hold_file)
    began = time.monotonic(); completed = 0; holder = []
    config = load_config(args.config)
    atomic_json(pool.root/f'PRODUCER_{os.getpid()}_RESOLVED.json', {
        'args':vars(args),'config':config,'session_hash':pool.session['content_hash'],
        'source_hashes':{p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')},
        'status':'BOUNDED_PRODUCER','training_started':False,'base_process_signals':False}, immutable=True)
    def load(native):
        if not holder:
            admission = admit_gpu(args)
            atomic_json(pool.root/f'PRODUCER_{os.getpid()}_ADMISSION.json', admission, immutable=True)
            holder.append(load_front(config, native, args.device, frozen_only=True))
        return holder[0]
    # Load once before waiting for the consumer, so the next window does not
    # incur cold model loading on the training critical path.
    load(pool.full)
    while completed < args.max_windows and time.monotonic() - began < args.max_seconds:
        producer_placement(); hold_check(args.hold_file)
        desired, request = pool.desired()
        keep = set(range(max(0, (desired[0] if desired else math.ceil((pool.session['max_optimizer_steps']-
            pool.session['parent_step'])/pool.session['window_updates'])) - 1),
            (desired[-1]+1) if desired else math.ceil((pool.session['max_optimizer_steps']-
                pool.session['parent_step'])/pool.session['window_updates'])))
        paths = {p.parent:p for name in ('READY.json','EVICTING.json') for p in (pool.root/'windows').glob('*/'+name)}
        for path in paths.values():
            number = int(path.parent.name)
            if number not in keep or (path.parent / 'EVICTING.json').exists():
                try: pool.evict(number)
                except BlockingIOError: continue
        partials = {p.parent for name in ('BUILDING.json','ABANDONING.json')
                    for p in (pool.root/'windows').glob('*/'+name)}
        for folder in partials:
            if (folder/'READY.json').exists(): continue
            if int(folder.name) not in keep or (folder/'ABANDONING.json').exists():
                try: pool.discard_stale_partial(int(folder.name))
                except BlockingIOError: continue
        pending = [n for n in desired if pool.ready(n) is None]
        progressed = False
        for number in pending:
            if completed >= args.max_windows or time.monotonic()-began >= args.max_seconds: break
            try:
                result = build_window(pool, number, model_loader=load, rate_mib_s=args.rate_mib_s,
                    reserve_gib=pool.session['reserve_bytes']/1024**3,
                    max_gib=args.max_gib_per_window, hold_file=args.hold_file,
                    deadline=began+args.max_seconds,worker=args.worker,workers=args.workers)
            except BlockingIOError: continue
            print(json.dumps({'event':'window','number':number,'status':result['status']}), flush=True)
            if result.get('stop_requested'): return {**result,'status':'OPERATOR_PAUSED'}
            if result['status'] == 'SKIPPED_REPLAN': break
            if result['status'] == 'PARTIAL': continue
            if result['status'] != 'WINDOW_COMPLETE': return result
            completed += 1; progressed = True
        if not progressed: time.sleep(min(1., max(.001, args.max_seconds-(time.monotonic()-began))))
    return {'status':'BOUNDED_PRODUCER_FINISHED','windows_committed':completed,
            'seconds':time.monotonic()-began,'base_gpu_compute':False,'full_cache_complete':False}


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == 'plan':
            result = prepare_session(args.full_plan,args.parent_checkpoint,args.root,
                window_updates=args.window_updates,lookahead=args.lookahead,capacity_gib=args.capacity_gib,
                reserve_gib=args.reserve_gib,reuse_roots=args.reuse_cache,qualification=args.qualification)
            result = {k:result[k] for k in ('schema','content_hash','parent_step','capacity_bytes','worst_case_pool_bytes')}
        elif args.command in {'train','resume'}:
            from ..vision_cache.cli import hold_check
            from .runtime import RollingRuntime
            from ..performance.run import main as train
            hold_check(args.hold_file)
            runtime = RollingRuntime(args.rolling_config,resume_from_run=args.command=='resume',allow_fallback=args.allow_fallback)
            cli = ['--config',args.config,'--runtime',args.runtime,'--checkpoint',args.parent_checkpoint or 'auto',
                '--output',args.output,'--mode',args.mode,'--updates',str(args.stop_after),'--warmup',str(args.warmup),
                '--checkpoint-every',str(args.checkpoint_every)]
            if args.qualification: cli += ['--qualification',args.qualification]
            for name in ('save','fingerprint','probe'):
                if getattr(args,name): cli += ['--'+name]
            train(cli,production=runtime); return 0
        else:
            pool = make_pool(args.root,getattr(args,'config',None))
            try:
                if args.command == 'prewarm':
                    from .plan import checked
                    result = checked(pool.root/'REQUEST.json') if (pool.root/'REQUEST.json').exists() else pool.start_attempt(pool.session['parent_step'])
                elif args.command == 'produce': result = run_producer(args,pool)
                elif args.command == 'status':
                    desired, _ = pool.desired(); windows = []
                    for n in desired:
                        plan = pool.plan(n); refs, missing = pool.split_sources(plan)
                        windows.append({'number':n,'first_step':plan['first_step'],'last_step':plan['last_step'],
                            'observations':len(plan['rows']),'permanent_hits':len(refs),'to_compute':len(missing),
                            'ready':pool.ready(n) is not None})
                    result={'status':'INSPECTED','windows':windows,'owned_bytes':pool.resident_owned_bytes()}
                else:
                    from .reader import RollingReader
                    from .plan import checked
                    require(1 <= args.max_updates <= 200, 'bounded IO verification only')
                    request = checked(pool.root/'REQUEST.json'); readers = []
                    from ..vision_cache.storage import encode_record
                    from ..io import digest
                    import hashlib
                    started = time.monotonic(); hashes = []
                    # Sequential simulation of each rank's original IDs. This
                    # is real cached-tensor IO, explicitly NOT GPU/DDP evidence.
                    for rank in range(request['world_size']):
                        reader=RollingReader(pool,request,rank=rank,epoch=pool.session['epoch'],
                            offset=pool.session['offset'],step=pool.session['parent_step'],wait_seconds=1)
                        for _ in range(args.max_updates):
                            a=reader.expected.at(reader.epoch,reader.offset)
                            ids=[pool.full['rows'][i]['scene_id'] for i in a.mine]
                            for sid in ids: hashes.append((sid,hashlib.sha256(encode_record(reader.read(sid))).hexdigest()))
                            reader.prepared(ids)
                        readers.append(dict(reader.stats));reader.close()
                    result={'status':'PASS','scope':'REAL_PAYLOAD_IO_AND_ORIGINAL_SAMPLER_NOT_TRAINING',
                        'reads':sum(r['reads'] for r in readers),'bytes':sum(r['bytes'] for r in readers),
                        'seconds':time.monotonic()-started,'record_hashes':hashes,'reader_stats':readers}
                    atomic_json(args.output,result)
                    result={k:v for k,v in result.items() if k not in {'record_hashes','reader_stats'}}
            finally: pool.close()
        print(json.dumps(result,sort_keys=True),flush=True)
        return 0 if result.get('status') not in {'PARTIAL','OPERATOR_PAUSED'} else 3
    except Exception as error:
        print(json.dumps({'status':'FAILED','error_type':type(error).__name__,'error':str(error)}),flush=True)
        return 2


if __name__=='__main__': sys.exit(main())
