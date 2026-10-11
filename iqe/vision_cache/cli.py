"""Strict production FP32 cache and continuation entrypoints. No implicit launch."""
import argparse
from pathlib import Path
import json
import os
import sys
import time
import torch
from ..config import load_config
from ..contracts import require
from ..io import read_json, atomic_json, file_hash, digest


def load_front(config, plan, device, *, frozen_only=False):
    from omegaconf import OmegaConf
    from ..query_base import build_query_framework, bind_source
    from ..training.runtime_identity import configure_runtime
    configure_runtime({'deterministic_backward':True})
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = True
    contract = read_json(config['s0']['contract'])
    bind_source(contract['source_root'], contract['source_commit'])
    for name, expected in contract['query_model_code_hashes'].items():
        require(file_hash(name) == expected, 'protected Base source changed')
    state = torch.load(plan['parent_checkpoint'], map_location='cpu', mmap=True, weights_only=False)
    require(state['optimizer_step'] == plan['parent_step'], 'plan/parent step mismatch')
    require(read_json(plan['parent_checkpoint']+'.COMPLETE.json')['checksum'] == plan['parent_sha256'], 'plan/parent receipt mismatch')
    # Only visual weights are needed; a shared staged checkpoint may avoid repeat
    # physical IO, but we still check the real parent checksum once per process.
    if not frozen_only: require(file_hash(plan['parent_checkpoint']) == plan['parent_sha256'], 'parent checksum mismatch')
    model = build_query_framework(OmegaConf.create(contract['source_config']), contract['query_architecture'],
                                  contract['source_root'], contract['source_commit']).float().to(device)
    if frozen_only:
        from .identity import checkpoint_visual_hash
        require(checkpoint_visual_hash(state) == plan['visual_state_sha256'], 'frozen checkpoint state changed after plan verification')
        prefix='qwen_vl_interface.model.model.visual.'
        model.qwen_vl_interface.model.model.visual.load_state_dict(
            {k[len(prefix):]:v for k,v in state['model'].items() if k.startswith(prefix)}, strict=True)
    else: model.load_state_dict(state['model'], strict=True)
    model.eval()
    return model, contract


def parser():
    p = argparse.ArgumentParser(__doc__); sub = p.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('cache-plan', help='lock exact parent fit list or next original sampler batches')
    plan.add_argument('--config', required=True); plan.add_argument('--parent-checkpoint', required=True)
    plan.add_argument('--output', required=True); plan.add_argument('--scope', choices=['pilot','representative','full'], required=True)
    plan.add_argument('--updates', type=int, default=200); plan.add_argument('--max-samples', type=int, default=64)
    plan.add_argument('--records-per-shard', type=int, default=24)
    build = sub.add_parser('cache-build', help='frozen-only bounded shard builder; idempotent continuation')
    build.add_argument('--config', required=True); build.add_argument('--plan', required=True); build.add_argument('--root', required=True)
    build.add_argument('--device', required=True); build.add_argument('--worker', type=int, default=0); build.add_argument('--workers', type=int, default=1)
    build.add_argument('--max-shards', type=int, required=True); build.add_argument('--max-gib', type=float, required=True)
    build.add_argument('--reserve-gib', type=float, default=500.); build.add_argument('--rate-mib-s', type=float, default=256.)
    build.add_argument('--qualification'); build.add_argument('--hold-file', required=True)
    build.add_argument('--reuse-cache',action='append',default=[],help='verified nonoverlapping cache on same mount; hardlink immutable shards')
    verify = sub.add_parser('cache-verify', help='verify immutable index/records; optionally all shard hashes')
    verify.add_argument('--root', required=True); verify.add_argument('--max-samples', type=int, required=True)
    verify.add_argument('--full-shards', action='store_true'); verify.add_argument('--output', required=True)
    bench = sub.add_parser('cache-benchmark', help='bounded shared-record read benchmark, never training throughput')
    bench.add_argument('--root', required=True); bench.add_argument('--max-samples', type=int, required=True)
    bench.add_argument('--repetitions', type=int, default=1); bench.add_argument('--output', required=True)
    bench.add_argument('--cold-advised', action='store_true')
    for name in ('train-base-cached','resume-base-cached'):
        train = sub.add_parser(name, help='existing Query trainer, original schedule, required parent/lineage')
        train.add_argument('--config', required=True); train.add_argument('--production-config', required=True)
        train.add_argument('--runtime', required=True); train.add_argument('--parent-checkpoint', required=name=='train-base-cached')
        train.add_argument('--output', required=True); train.add_argument('--mode', choices=['bounded','continuation'], required=True)
        train.add_argument('--stop-after', type=int, required=True, help='ADDITIONAL updates; never rescales original100k scheduler')
        train.add_argument('--warmup', type=int, default=10); train.add_argument('--qualification')
        train.add_argument('--checkpoint-every', type=int, default=1000); train.add_argument('--probe', action='store_true')
        train.add_argument('--fingerprint', action='store_true'); train.add_argument('--save', action='store_true')
        train.add_argument('--allow-fallback', action='store_true'); train.add_argument('--hold-file', required=True)
        train.add_argument('--local-checkpoint-root', default='/var/tmp/iqe-fp32-continuation-checkpoints')
    cp = sub.add_parser('checkpoint-roundtrip', help='read-only full-state schema/hash and original loading audit')
    cp.add_argument('--checkpoint', required=True); cp.add_argument('--output', required=True)
    return p


def hold_check(path):
    hold = read_json(path)
    state = hold.get('status', hold.get('mode'))
    require(state in {'conditional','CONDITIONAL','RELEASED','released','CONDITIONAL_CAPACITY','CONDITIONAL_UNTIL_CAPACITY_SUFFICIENT'},
            'cache writes/training held by current operator instruction')
    return file_hash(path)


def main(argv=None):
    a = parser().parse_args(argv)
    try:
        if a.command == 'cache-plan':
            from .plan import make_plan
            result = make_plan(load_config(a.config), a.parent_checkpoint, a.output, scope=a.scope,
                               updates=a.updates, max_samples=a.max_samples, records_per_shard=a.records_per_shard)
        elif a.command == 'cache-build':
            from .builder import build_cache
            from .plan import load_plan, capacity_check
            hold_hash = hold_check(a.hold_file)
            # Qualified host/idle physical GPU checks occur before model allocation.
            from ..resources import qualify
            qualify(a.device)
            root = Path(a.root); root.mkdir(parents=True, exist_ok=True)
            capacity = capacity_check(root, int(a.max_gib*1024**3), reserve_bytes=int(a.reserve_gib*1024**3), temporary_bytes=2*1024**3)
            atomic_json(root/f'CAPACITY_{a.worker}.json', capacity)
            require(capacity['sufficient'], 'BLOCKED_SHARED_CAPACITY: explicit phase budget does not fit protected reserve')
            config = load_config(a.config); plan = load_plan(a.plan)
            atomic_json(root/f'BUILD_RESOLVED_{a.worker}.json', {'args':vars(a),'config':config,'plan_hash':plan['plan_hash'],'hold_sha256':hold_hash})
            model, contract = load_front(config, plan, a.device, frozen_only=True)
            result = build_cache(model, contract, a.plan, root, worker=a.worker, workers=a.workers,
                max_shards=a.max_shards, max_gib=a.max_gib, reserve_gib=a.reserve_gib,
                rate_mib_s=a.rate_mib_s, qualification=a.qualification,hold_file=a.hold_file,reuse_roots=a.reuse_cache)
        elif a.command in {'cache-verify','cache-benchmark'}:
            from .storage import CacheReader
            require(a.max_samples > 0, 'explicit positive sample budget')
            reader = CacheReader(a.root); ids = sorted(reader.allowed)[:a.max_samples]
            if a.command == 'cache-verify':
                result = reader.verify(full_shards=a.full_shards, scene_ids=ids)
            else:
                from .plan import mount_info
                require(1 <= a.repetitions <= 10, 'bounded IO repetitions')
                if a.cold_advised:
                    for name in reader.index['shards']:
                        fd=os.open(Path(a.root)/name,os.O_RDONLY)
                        try: os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
                        finally: os.close(fd)
                times=[]; begin=time.perf_counter()
                for _ in range(a.repetitions):
                    for token in ids:
                        t=time.perf_counter();reader.read(token);times.append(time.perf_counter()-t)
                import numpy as np
                elapsed=time.perf_counter()-begin
                result={'status':'PASS','scope':'IO_ONLY_NOT_UPDATE_THROUGHPUT','samples':len(times),'wall_seconds':elapsed,
                    'mean':float(np.mean(times)),'p50':float(np.quantile(times,.5)),'p95':float(np.quantile(times,.95)),
                    'bytes_per_second':reader.stats['bytes']/elapsed,'reader':reader.stats,'mount':mount_info(a.root),
                    'page_cache':'cold-advised, physical cold not guaranteed' if a.cold_advised else 'OS page cache uncontrolled'}
            atomic_json(a.output,result); reader.close()
        elif a.command in {'train-base-cached','resume-base-cached'}:
            hold_check(a.hold_file)
            from .runtime import ProductionRuntime
            from ..performance.run import main as train
            runtime=ProductionRuntime(a.production_config,resume_from_run=a.command=='resume-base-cached',allow_fallback=a.allow_fallback)
            argv=['--config',a.config,'--runtime',a.runtime,'--checkpoint',a.parent_checkpoint or 'auto',
                  '--output',a.output,'--mode',a.mode,'--updates',str(a.stop_after),'--warmup',str(a.warmup),
                  '--checkpoint-every',str(a.checkpoint_every),'--local-checkpoint-root',a.local_checkpoint_root]
            if a.qualification:argv+=['--qualification',a.qualification]
            for name in ('probe','fingerprint','save'):
                if getattr(a,name):argv+=['--'+name]
            train(argv,production=runtime);return
        else:
            from ..training.resume import inspect_checkpoint
            result=inspect_checkpoint(a.checkpoint)
            state=torch.load(a.checkpoint,map_location='cpu',mmap=True,weights_only=False)
            result.update(status='PASS',learned_driving_weights_loaded=state.get('learned_driving_weights_loaded'),
                inference_visual_source=state.get('inference_visual_source'),
                training_visual_source=state.get('training_visual_source'),mode=state.get('mode'),
                scope='CPU_FULL_STATE_SCHEMA_AND_CHECKSUM; use evaluate-base for real model forward')
            atomic_json(a.output,result)
        print(json.dumps(result,sort_keys=True),flush=True)
        if result.get('status') in {'PARTIAL','OPERATOR_PAUSED'}: return 3
        return 0
    except Exception as error:
        print(json.dumps({'status':'FAILED','error_type':type(error).__name__,'error':str(error)}),flush=True)
        return 2


if __name__=='__main__':sys.exit(main())
