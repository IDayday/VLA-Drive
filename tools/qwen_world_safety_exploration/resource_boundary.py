"""Recognize only a complete owned native return at a resource deadline.

This is CPU orchestration. It does not migrate scientific identities, change
optimizer state, or retry numerical failures.
"""
from pathlib import Path
import json
import time


def cooperative_return(run, node, receipt, latest, target):
    from starVLA.rl.qwen_world_safety_exploration.contracts import (
        authorization, allowed_hosts, file_sha)
    if receipt.get('returncode') != 0 or not 0 < latest.get('updates', 0) < target:
        return None
    host = node.get('host')
    grant, _ = authorization()
    if host in allowed_hosts(admit_long_job=True):
        return None
    job = Path(node['job'])
    finite_guard = (host in grant.get('temporary_hosts', []) and
                    time.time() >= grant['expires_unix']-grant['checkpoint_guard_seconds'])
    if not ((job/'STOP_REQUESTED').exists() or receipt.get('migration_required') or finite_guard):
        return None
    run_root = Path(run['root']).resolve()
    run_id = run['id']
    if (Path(run_id).name != run_id or run_root.parent.name != 'runs'
            or not job.resolve().is_relative_to(run_root.parent.parent/'jobs'/run_id)):
        raise ValueError('Resource return job is outside its own study/run')
    cp = Path(latest['checkpoint']).resolve()
    if not cp.is_relative_to(run_root/'checkpoints'):
        raise ValueError('Resource return cannot adopt a foreign checkpoint')
    seal = json.loads((cp/'COMPLETE').read_text())
    identity = json.loads((cp/'identity.json').read_text())
    files = json.loads((cp/'files.json').read_text())
    if identity != json.loads((run_root/'identity.json').read_text()):
        raise ValueError('Resource return has a different native run identity')
    required = {'optimizer.pt', 'identity.json', 'transaction.json'} | {f'rank{r}.pt' for r in range(8)}
    if not isinstance(files, dict) or not required.issubset(files):
        raise ValueError('Resource return lacks the complete native file inventory')
    if any(not isinstance(name, str) or Path(name).is_absolute()
           or not (cp/name).resolve().is_relative_to(cp) or not (cp/name).is_file() for name in files):
        raise ValueError('Resource return file inventory escapes its checkpoint or is missing')
    u = latest['updates']
    if (seal.get('updates') != u or seal.get('world') != 8
            or seal.get('next_inner') != u % 4
            or seal.get('files_sha256') != file_sha(cp/'files.json')
            or identity.get('purpose') != 'formal' or identity.get('world') != 8
            or identity.get('recipe', {}).get('adaptation') != 'ATTN_LORA_MEMORY'
            or any(not (cp/name).is_file() for name in files)):
        raise ValueError('Incomplete native resource return; no automatic restart')
    import yaml
    if identity['recipe'] != yaml.safe_load(Path(run['config']).read_text()):
        raise ValueError('Resource return changed its scientific recipe')
    for rank in range(8):
        result = json.loads((Path(run['root'])/f'result_rank{rank}.json').read_text())
        if (not (cp/f'rank{rank}.pt').is_file() or result.get('updates') != u
                or Path(result.get('checkpoint', '')).resolve() != cp):
            raise ValueError('Missing complete eight-rank resource return')
    return dict(job=str(job), host=host, checkpoint=str(cp), updates=u,
                next_inner=u % 4, files_sha256=seal['files_sha256'],
                reason='OWNED_COMPLETE_NATIVE_RESOURCE_BOUNDARY',
                model_optimizer_rng_cursor_pending_preserved=True,
                full_tensor_hashes_deferred_to_strict_native_restore=True)
