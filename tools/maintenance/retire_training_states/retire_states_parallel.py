"""Explicit user-authorized recovery retirement, preserving inference tensors."""
import argparse
import collections
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import torch

ROOT = Path(__file__).resolve().parent


def read(path): return json.loads(Path(path).read_text())


def atomic(path, value):
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temp.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b''): h.update(block)
    return h.hexdigest()


def same_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict): return isinstance(b, dict) and a.keys() == b.keys() and all(same_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return type(a) == type(b) and len(a) == len(b) and all(same_tree(x, y) for x, y in zip(a, b))
    return a == b


def split_optimizer(value):
    if not isinstance(value, dict): raise ValueError('Unknown optimizer container')
    if 'adapters' in value:
        if not isinstance(value['adapters'], dict) or not value['adapters'] or not all(isinstance(v, torch.Tensor) for v in value['adapters'].values()):
            raise ValueError('Unknown adapter payload')
        return {k: v for k, v in value.items() if k not in {'optimizer', 'scheduler', 'scaler', 'rng', 'grad_scaler'}}
    if set(value) <= {'optimizer', 'scheduler', 'scaler', 'rng', 'grad_scaler'} and 'optimizer' in value:
        optimizer = value['optimizer']
        if not isinstance(optimizer, dict) or not set(optimizer) <= {'state', 'param_groups'}:
            raise ValueError('Optimizer may own FP32 model masters; preserve for explicit conversion')
        for state in optimizer.get('state', {}).values():
            if isinstance(state, dict) and any('master' in str(k).lower() or 'fp32' in str(k).lower() for k in state):
                raise ValueError('FP32 master weights require preservation')
        return None
    if set(value) <= {'state', 'param_groups'} and 'param_groups' in value: return None
    raise ValueError('Unknown model/master-weight ownership; retain rather than guess')


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--inventory', required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--shard-rank',type=int,default=0);parser.add_argument('--shard-world',type=int,default=1)
    parser.add_argument('--attempt',default='parallel1');args = parser.parse_args()
    if not 0 <= args.shard_rank < args.shard_world: raise ValueError('Invalid cleanup sharding')
    if Path(args.attempt).name != args.attempt: raise ValueError('Invalid attempt name')
    lock=(ROOT/f'cleanup_rank{args.shard_rank}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(1)
    inventory = read(Path(args.inventory)); protected = {tuple(x) for x in read(ROOT / 'ACTIVE_PROTECTED.json')['inodes']}
    groups = collections.defaultdict(list)
    for checkpoint in inventory['checkpoints']:
        for file in checkpoint['files']:
            groups[(file['device'], file['inode'])].append((checkpoint, file))
    (ROOT / 'inference_weights').mkdir(exist_ok=True)
    totals = dict(removed_paths=0, removed_logical_bytes=0, allocated_bytes_unlinked_last_link=0,
                  inference_payload_bytes=0, compacted_optimizer_inodes=0, skipped_protected_inodes=0,
                  skipped_unknown_inodes=0, skipped_missing_inodes=0)
    journal = (ROOT / f'{"APPLIED" if args.apply else "PLAN"}_{args.attempt}_rank{args.shard_rank}.jsonl').open('x', buffering=1)
    for inode, members in groups.items():
        placement=int.from_bytes(hashlib.sha256(str(Path(members[0][1]['path']).parent).encode()).digest()[:8],'big') % args.shard_world
        if placement != args.shard_rank: continue
        if inode in protected:
            totals['skipped_protected_inodes'] += 1; continue
        present=[m for m in members if Path(m[1]['path']).exists()]
        if not present: totals['skipped_missing_inodes'] += 1; continue
        checkpoint, file = present[0]; path = Path(file['path'])
        st = path.stat()
        if path.is_symlink() or (st.st_dev, st.st_ino, st.st_size) != (inode[0], inode[1], file['size']):
            raise ValueError('Input changed since inventory: ' + str(path))
        result = dict(inode=list(inode), original_path=str(path), aliases=[m[1]['path'] for m in members], at=time.time())
        keep = None; payload = None; phase = 'CLASSIFY'
        try:
            payload = torch.load(path, mmap=True, map_location='cpu', weights_only=False)
            if path.name == 'optimizer.pt':
                keep = split_optimizer(payload)
                if keep is not None: result['kind'] = 'ADAPTERS_PRESERVED_REMOVE_ADAM_SCHEDULER'
                else: result['kind'] = 'PURE_OPTIMIZER_SCHEDULER'
            elif path.name.startswith(('rank', 'retained_rank')):
                expected = {'state', 'rollout', 'rng', 'identity_sha256', 'identity', 'optimizer', 'scheduler', 'local_parameter_names'}
                if not isinstance(payload, dict) or not set(payload) <= expected or 'state' not in payload:
                    raise ValueError('Unknown rank payload; may include model weights')
                if payload.get('rollout') is not None:
                    keep = {k: v for k, v in payload.items() if k not in {'rng', 'optimizer', 'scheduler'}}
                    result['kind'] = 'ROLLOUT_DIAGNOSTICS_PRESERVED'
                else: result['kind'] = 'PURE_RANK_OPTIMIZER_RNG_CURSOR'
            else: raise ValueError('Unknown state filename')
            if not args.apply:
                result['apply'] = False; journal.write(json.dumps(result) + '\n'); continue
            phase = 'PRESERVE_AND_REMOVE'
            if keep is not None:
                target = ROOT / 'inference_weights' / f'{inode[0]}_{inode[1]}.pt'
                existed=target.exists()
                temporary=target if existed else target.with_suffix('.building')
                if not existed:
                    if temporary.exists():
                        # Only our own interrupted staging file; the original
                        # source is still present and classified above.
                        abandoned=temporary.with_suffix('.aborted.'+str(time.time_ns()))
                        os.rename(temporary,abandoned)
                    with temporary.open('xb') as stream:
                        torch.save(keep, stream); stream.flush(); os.fsync(stream.fileno())
                check = torch.load(temporary, mmap=True, map_location='cpu', weights_only=False)
                if not same_tree(keep, check): raise ValueError('Inference tensor roundtrip mismatch')
                del check
                if not existed: os.rename(temporary, target)
                result.update(inference_payload=str(target), inference_payload_sha256=sha(target), inference_tensor_roundtrip='EXACT')
                if not existed: totals['inference_payload_bytes'] += target.stat().st_blocks * 512
                if path.name == 'optimizer.pt': totals['compacted_optimizer_inodes'] += 1
                for _, alias in members:
                    parent = Path(alias['path']).parent
                    link = parent / ('inference_adapters.pt' if path.name == 'optimizer.pt' else 'diagnostics_' + Path(alias['path']).name)
                    if link.exists():
                        if not os.path.samefile(target,link):raise ValueError('Different existing inference archive: '+str(link))
                    else:os.link(target, link)
            # No mapped tensor references may keep the retired large file alive.
            del keep, payload; keep = payload = None
            if file['size'] > 64 * 1024**2: gc.collect()
            for cp, alias in members:
                target = Path(alias['path'])
                if not target.exists(): continue
                current = target.stat()
                if target.is_symlink() or (current.st_dev, current.st_ino, current.st_size) != (inode[0], inode[1], alias['size']):
                    raise ValueError('Changed unlink target: ' + str(target))
                parent = target.parent; receipt = parent / 'RECOVERY_REMOVED_BY_USER.json'
                receipt_lock=(parent/'.recovery-retirement.lock').open('a');fcntl.flock(receipt_lock,fcntl.LOCK_EX)
                row = read(receipt) if receipt.exists() else dict(schema='user_resume_state_retirement_v1', checkpoint=str(parent),
                    user_instruction='Ended and completed experiments do not need resumable training state',
                    resume_available=False, original_manifests_preserved=True, files=[])
                entry = dict(name=target.name, size=alias['size'], inode=list(inode), original_manifest='files.json',
                    inference_payload=result.get('inference_payload'), inference_payload_sha256=result.get('inference_payload_sha256'),
                    kind=result['kind'], at=time.time())
                previous=[e for e in row.setdefault('files',[]) if e.get('name')==target.name]
                if previous and any(e.get('inode')!=entry['inode'] or e.get('inference_payload_sha256')!=entry['inference_payload_sha256'] for e in previous):
                    raise ValueError('Retirement ancestry changed')
                row['files']=[e for e in row['files'] if e.get('name')!=target.name]+[entry]
                row['status'] = 'RETIREMENT_RECORDED_BEFORE_UNLINK'
                atomic(receipt, row)
                target.unlink()
                totals['removed_paths'] += 1; totals['removed_logical_bytes'] += current.st_size
                if current.st_nlink == 1: totals['allocated_bytes_unlinked_last_link'] += current.st_blocks * 512
                row['status'] = 'RESUME_STATES_REMOVED_INFERENCE_WEIGHTS_RETAINED'; atomic(receipt, row)
                receipt_lock.close()
            result['status'] = 'REMOVED'; journal.write(json.dumps(result) + '\n')
        except ValueError as error:
            if phase != 'CLASSIFY': raise
            result.update(status='PRESERVED_UNKNOWN', reason=str(error)); totals['skipped_unknown_inodes'] += 1
            journal.write(json.dumps(result) + '\n')
        finally:
            payload = keep = None
        if sum(totals[k] for k in ['removed_paths', 'skipped_unknown_inodes']) % 100 == 0:
            atomic(ROOT / f'PROGRESS_rank{args.shard_rank}.json', dict(at=time.time(), **totals))
    journal.flush(); os.fsync(journal.fileno()); journal.close()
    totals['net_reclaimed_allocated_bytes'] = totals['allocated_bytes_unlinked_last_link'] - totals['inference_payload_bytes']
    name=('PHASE2_COMPLETE.json' if args.shard_world==1 else f'PHASE2_COMPLETE_rank{args.shard_rank}.json') if args.apply else f'DRY_RUN_COMPLETE_rank{args.shard_rank}.json'
    atomic(ROOT / name, dict(status='COMPLETE', at=time.time(), **totals))
    print(json.dumps(totals))


if __name__ == '__main__': main()
