"""Cache only frozen S0 W/H_A and two genuine ten-step plans, from real images.

This process imports the pinned S runtime first. It never loads G or calls a
trainer. All new helpers are local file-loaded to avoid shadowing that runtime.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('decision_local_cache_helpers', Path(__file__).with_name('common.py'))
helpers = importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anchor', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'dev', 'navtest'), required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--diagnostic-train', action='store_true')
    parser.add_argument('--role', choices=('A_train','A_calibration','B_predictor','B_refiner'))
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--chunk-size', type=int, default=128)
    parser.add_argument('--max-seconds', type=float, required=True)
    parser.add_argument('--verify-samples', type=int, default=4)
    parser.add_argument('--storage-dtype', choices=('float16', 'float32'), default='float16')
    parser.add_argument('--continue-existing', action='store_true',
        help='Explicit stateless worker redistribution, preserving the original cache contract')
    parser.add_argument('--continuation-attempt', default='')
    args = parser.parse_args()
    resources = helpers.check_host()
    anchor = helpers.read(args.anchor)
    if not 0 <= args.rank < args.shards or args.chunk_size < 1 or args.max_seconds <= 0:
        raise ValueError('Invalid bounded shard request')
    if args.split == 'navtest' and not (args.output.parent/'FINAL_METHOD_LOCK.json').exists():
        raise ValueError('New Navtest extraction waits for the shared final method/endpoint lock')
    spec=importlib.util.spec_from_file_location('cache_anchor_runtime',Path(__file__).with_name('anchor_runtime.py'))
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    legacy=adapter.activate(anchor)
    import numpy as np
    import torch
    import starVLA
    from safetensors.torch import save_file
    from tools.foresight.checkpoints import load_student, scene_noise
    from starVLA.dataloader.foresight_dataset import ForesightCurrentDataset, decode_ego
    if Path(starVLA.__file__).resolve().parent != (legacy/'starVLA').resolve():
        raise RuntimeError('Pinned legacy package was shadowed')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    data = ForesightCurrentDataset(anchor['data'][args.split]['current_root'])
    if data.identity != anchor['data'][args.split]['identity']:
        raise ValueError('Current source identity changed')
    index = adapter.selected_index(anchor,args.split,args.role)
    if args.diagnostic_train:
        if args.split != 'train' or args.role is not None or anchor.get('training_role')=='A_train':
            raise ValueError('Diagnostic subset is training-only')
        index = helpers.read(anchor['diagnostic_train_index'])
    if args.limit:
        index = index[:args.limit]
    lookup = {row['token']: i for i, row in enumerate(data.index)}
    config = {'schema': 'decision_local_compact_anchor_cache_v1', 'anchor_identity': anchor['identity'],
        'checkpoint_sha256': anchor['checkpoint']['sha256'], 'source_sha': helpers.source_sha(ROOT),
        'writer_sha256': helpers.digest(__file__), 'split': args.split,
        'current_identity': data.identity['identity'], 'index': index,
        'seeds': [42, 43] if args.split == 'train' else [42], 'storage_dtype': args.storage_dtype,
        'online_feature_cast': args.storage_dtype+' then FP32; q0 always original FP32',
        'encoder_precision': 'FP32 master, FP32 inference, TF32 off, batch1, frozen eval',
        'chunk_size': args.chunk_size, 'model_forward_count_per_scene': 1,
        'ego_target_location': 'separate original label files; never cache-worker model inputs'}
    if anchor.get('training_role')=='A_train':
        config.update(data_role=args.role or args.split,action_codec_identity=anchor['action_codec_identity'],
            source_audit_identity=anchor['source_audit_identity'],runtime_source_sha=anchor['runtime_source_sha'])
    ident = helpers.identity(config)
    args.output.mkdir(parents=True, exist_ok=True)
    existing = args.output/'identity.json'
    producer={'source_sha':helpers.source_sha(ROOT),'writer_sha256':helpers.digest(__file__),
        'attempt':args.continuation_attempt,'rank':args.rank,'shards':args.shards}
    if args.continue_existing:
        if not args.continuation_attempt or not existing.exists():raise ValueError('Explicit existing-cache continuation identity required')
        origin=helpers.read(existing)
        # Only writer execution/provenance may differ. Every model, sample,
        # precision, RNG, shape and storage field must match exactly.
        for key,value in config.items():
            if key not in ('source_sha','writer_sha256') and origin.get(key)!=value:
                raise ValueError('Changed cache semantics: '+key)
        ident=origin['identity'];config={k:v for k,v in origin.items() if k!='identity'}
    else:
        if existing.exists() and helpers.read(existing)['identity'] != ident:
            raise ValueError('Refusing incompatible cache reuse')
        helpers.atomic_json(existing, {'identity': ident, **config})
    helpers.atomic_json(args.output/'index.json', index)
    for name in ('chunks', 'records', 'workers'):
        (args.output/name).mkdir(exist_ok=True)
    started = time.monotonic()
    training = helpers.read(Path(anchor['run'])/'identity.json')
    model = load_student(anchor['run'], anchor['tag'], training, precision='fp32', strip=True)
    model.requires_grad_(False).eval()
    if any(p.requires_grad for p in model.parameters()) or any(m.training for m in model.modules()):
        raise AssertionError('Anchor must be entirely frozen/eval')
    torch.cuda.synchronize()
    loaded = time.monotonic()-started
    state_before = {k: p.detach().cpu().clone() for k, p in model.named_buffers()}
    dtype = torch.float16 if args.storage_dtype == 'float16' else torch.float32
    profile, checks, completed = [], [], 0
    continuation_check=None
    if args.continue_existing:
        from safetensors.torch import load_file
        committed=sorted((args.output/'records').glob('*.json'))
        if not committed:raise ValueError('Continuation requires a committed numerical reference')
        reference=helpers.read(committed[args.rank%len(committed)])
        original_path=args.output/'chunks'/(committed[args.rank%len(committed)].stem+'.safetensors')
        if helpers.digest(original_path)!=reference['sha256']:raise ValueError('Changed committed reference')
        row=next(x for x in index if x['token']==reference['tokens'][0])
        old=load_file(str(original_path));observation=data[lookup[row['token']]]
        with torch.no_grad():
            encoded=model.encode_current([observation]);action=model.build_planner_condition(encoded)
            plans=torch.stack([model.action_model.predict_action(action,initial_noise=scene_noise(row['token'],seed,action.device)).float()[0]
                for seed in config['seeds']])
        got={'W':encoded['W'][0].to(dtype).cpu(),'H_A':encoded['action_queries'][0].to(dtype).cpu(),
             'q0_encoded':plans.cpu(),'q0':decode_ego(plans).cpu(),
             'state':torch.from_numpy(np.asarray(observation['state'],dtype=np.float32)[0].copy())}
        for key,value in got.items():torch.testing.assert_close(value,old[key][0],rtol=0,atol=0)
        continuation_check={'token':row['token'],'all_stored_fields_bit_exact':True,'reference_sha256':reference['sha256']}
        del old,got,encoded,action,plans
    chunks = (len(index)+args.chunk_size-1)//args.chunk_size
    for chunk in range(args.rank, chunks, args.shards):
        if time.monotonic()-started >= args.max_seconds:
            break
        path = args.output/'chunks'/f'{chunk:06d}.safetensors'
        receipt = args.output/'records'/f'{chunk:06d}.json'
        members = index[chunk*args.chunk_size:(chunk+1)*args.chunk_size]
        if receipt.exists():
            old = helpers.read(receipt)
            if old['identity'] != ident or old['sha256'] != helpers.digest(path):
                raise ValueError('Cached chunk changed')
            completed += len(members)
            continue
        collected = {k: [] for k in ('W', 'H_A', 'q0_encoded', 'q0', 'state')}
        input_records = []
        for row in members:
            before_read = time.monotonic()
            observation = data[lookup[row['token']]]
            read_s = time.monotonic()-before_read
            torch.cuda.synchronize(); before_encode = time.monotonic()
            with torch.no_grad():
                encoded = model.encode_current([observation])
                action = model.build_planner_condition(encoded)
                plans = []
                for seed in config['seeds']:
                    plans.append(model.action_model.predict_action(action, initial_noise=scene_noise(row['token'], seed, action.device)).float()[0])
                q_encoded = torch.stack(plans)
                q_physical = decode_ego(q_encoded)
            torch.cuda.synchronize(); elapsed = time.monotonic()-before_encode
            if len(checks) < args.verify_samples:
                with torch.no_grad():
                    original = model.predict_action([observation], initial_noise=scene_noise(row['token'], config['seeds'][0], action.device))[0]
                    poisoned = dict(observation, ego=torch.randn(8, 4), future_map='NEVER_READ', future_validity=False)
                    invariant = model.predict_action([poisoned], initial_noise=scene_noise(row['token'], config['seeds'][0], action.device))[0]
                torch.testing.assert_close(q_physical[0], original, rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(original, invariant, rtol=0, atol=0)
                errors = {}
                for key in ('W', 'action_queries'):
                    raw = encoded[key].float()
                    error = raw.to(dtype).float()-raw
                    errors[key] = {'max_absolute': float(error.abs().max()),
                                   'relative_l2': float(error.norm()/raw.norm().clamp_min(1e-12))}
                    if not torch.isfinite(raw.to(dtype)).all():
                        raise FloatingPointError('Feature storage overflow')
                checks.append({'token': row['token'], 'original_plan_max_abs': float((original-q_physical[0]).abs().max()),
                               'future_label_invariance': True, 'cache_dtype_error': errors})
            collected['W'].append(encoded['W'][0].to(dtype).cpu())
            collected['H_A'].append(encoded['action_queries'][0].to(dtype).cpu())
            collected['q0_encoded'].append(q_encoded.cpu())
            collected['q0'].append(q_physical.cpu())
            collected['state'].append(torch.from_numpy(np.asarray(observation['state'], dtype=np.float32)[0].copy()))
            raw_record = Path(data.root)/'current'/(row['token']+'.json')
            input_records.append(helpers.digest(raw_record))
            profile.append({'input_read_s': read_s, 'encode_two_plans_s': elapsed})
        tensors = {k: torch.stack(v).contiguous() for k, v in collected.items()}
        if not all(torch.isfinite(v).all() for v in tensors.values()):
            raise FloatingPointError('Nonfinite frozen representation/proposal')
        temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
        save_file(tensors, str(temporary), metadata={'identity': ident, 'chunk': str(chunk)})
        temporary.replace(path)
        helpers.atomic_json(receipt, {'identity': ident, 'sha256': helpers.digest(path),
            'bytes': path.stat().st_size, 'tokens': [r['token'] for r in members],
            'current_record_sha256': input_records, 'features': {k: list(v.shape) for k, v in tensors.items()},
            'producer':producer})
        completed += len(members)
        print(json.dumps({'rank': args.rank, 'chunk': chunk, 'scenes': completed,
                          'seconds': time.monotonic()-started, 'bytes_per_scene': path.stat().st_size/len(members)}), flush=True)
    for name, value in model.named_buffers():
        torch.testing.assert_close(value.cpu(), state_before[name], rtol=0, atol=0)
    torch.cuda.synchronize()
    seconds = time.monotonic()-started
    worker_name=(args.continuation_attempt+'_' if args.continue_existing else '')+f'{args.rank:03d}.json'
    helpers.atomic_json(args.output/'workers'/worker_name, {
        'identity': ident, 'rank': args.rank, 'shards': args.shards, 'scenes': completed,
        'seconds': seconds, 'GPU_hours': seconds/3600., 'loading_seconds': loaded,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'resource_check': resources,
        'plan_and_storage_checks': checks, 'timing': profile,'producer':producer,'continuation_reference':continuation_check,
        'all_frozen_buffers_unchanged': True, 'status': 'complete' if completed == sum(len(index[c*args.chunk_size:(c+1)*args.chunk_size]) for c in range(args.rank, chunks, args.shards)) else 'paused_at_chunk_boundary'})
    receipts = [args.output/'records'/f'{c:06d}.json' for c in range(chunks)]
    if all(p.exists() for p in receipts):
        helpers.atomic_json(args.output/'COMPLETE.json', {'identity': ident, 'scenes': len(index), 'chunks': chunks,
            'bytes': sum(helpers.read(p)['bytes'] for p in receipts), 'all_expected_chunks_present': True})


if __name__ == '__main__':
    main()
