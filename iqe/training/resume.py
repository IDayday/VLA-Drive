"""One checksum/schema-aware resolver for CLI, workflow and all trainers."""
from pathlib import Path
import torch
from ..contracts import require
from ..io import atomic_json, digest, file_hash, read_json, ContractError


def inspect_checkpoint(path, expected_dependencies=None, expected_config=None):
    path = Path(path)
    receipt = read_json(str(path) + '.COMPLETE.json')
    require(path.is_file(), 'checkpoint file missing')
    checksum = file_hash(path)
    require(checksum == receipt['checksum'], 'checkpoint checksum mismatch')
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    step = state['optimizer_step']
    require(type(step) is int and step >= 0 and step == receipt['optimizer_step'], 'optimizer_step metadata mismatch')
    if path.stem.startswith('step_'):
        require(int(path.stem[5:]) == step, 'checkpoint filename optimizer_step mismatch')
    require(state['dependencies'] == receipt['dependencies'], 'checkpoint/receipt dependency mismatch')
    require(state.get('resume_boundary') == 'optimizer_step', 'unsupported resume boundary')
    if state.get('kind') == 'iqe_query_base':
        schema = 'iqe_query_base'
        require(all(k in state for k in ('model','optimizer','scheduler','epoch','offset','rng_by_rank')), 'Base checkpoint schema')
        config = state['dependencies']['config']
        next_ids = None  # Base epoch/offset resolved by original batching code in trainer.
    else:
        schema = 'iqe_active_module_v1'
        require(state.get('schema_version') == 1 and all(k in state for k in
            ('module','optimizer','scheduler','sampler','rng_by_rank','resolved_config')), 'active-module checkpoint schema')
        config = state['resolved_config']
        next_ids = None
    if expected_dependencies is not None:
        require(state['dependencies'] == expected_dependencies, 'run/data/dependency identity mismatch')
    if expected_config is not None:
        require(config == expected_config, 'resolved training config mismatch')
    return {'path': str(path.resolve()), 'checksum': checksum, 'optimizer_step': step,
            'schema': schema, 'run_identity': digest({'dependencies': state['dependencies'], 'config': config}),
            'next_sample_ids': next_ids, 'valid': True}


def resolve_resume(folder, requested, *, expected_dependencies=None, expected_config=None, allow_fallback=False):
    if requested is None:
        return None
    requested = str(requested)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    receipts = sorted(folder.glob('step_*.pt.COMPLETE.json'))
    paths = [Path(str(p).removesuffix('.COMPLETE.json')) for p in receipts]
    # A complete filename without a receipt is not a usable checkpoint.
    paths += [p for p in sorted(folder.glob('step_*.pt')) if p not in paths]
    explicit = requested != 'auto'
    if explicit and Path(requested).resolve() not in [p.resolve() for p in paths]:
        paths.append(Path(requested))
    rows = []
    for path in paths:
        try:
            row = inspect_checkpoint(path, expected_dependencies, expected_config)
        except Exception as error:
            try:
                step = int(path.stem.split('_')[-1])
            except ValueError:
                step = -1
            row = {'path': str(path.resolve()), 'valid': False, 'optimizer_step': step, 'reason': str(error)}
        rows.append(row)
    valid = [r for r in rows if r['valid']]
    chosen = next((r for r in valid if r['path'] == str(Path(requested).resolve())), None) if explicit else max(valid, key=lambda r:r['optimizer_step'], default=None)
    result_path = folder/'result.json'
    result = read_json(result_path) if result_path.exists() else None
    report = {'schema_version': 1, 'request': requested, 'candidates': rows, 'allow_fallback': allow_fallback,
              'selected': chosen, 'exact_resume': True, 'status': 'RESOLVED' if chosen else 'REJECTED',
              'result_json_stale': bool(result and chosen and result.get('optimizer_steps', -1) < chosen['optimizer_step']),
              'next_sample_ids': None}
    error = None
    if chosen is None:
        error = 'NO_VALID_RESUME_CHECKPOINT: no compatible completed checkpoint; refusing fresh training: ' + '; '.join(r.get('reason','') for r in rows)
    elif explicit and any(r['optimizer_step'] > chosen['optimizer_step'] for r in rows if Path(r['path']).parent == folder.resolve()):
        error = 'FORK_RUN_REQUIRED: explicit older checkpoint must use a new output directory; higher-step artifacts are protected'
    elif not explicit and not allow_fallback and any(not r['valid'] and r['optimizer_step'] >= chosen['optimizer_step'] for r in rows):
        error = 'RESUME_FALLBACK_REQUIRED: newer incomplete/corrupt/incompatible checkpoint rejected; explicitly enable resume.allow_fallback'
    if error:
        report.update(status='REJECTED', error=error)
    import torch.distributed as dist
    writer = not dist.is_initialized() or dist.get_rank() == 0
    if writer:
        atomic_json(folder/'RESUME_RESOLUTION.json', report)
    if error:
        raise ContractError(error)
    if writer and explicit and Path(chosen['path']).parent != folder.resolve():
        atomic_json(folder/'RESUME_LINEAGE.json', {'parent': chosen, 'output': str(folder.resolve()), 'fork_run': True}, immutable=True)
    return chosen['path']


def record_next_batch(folder, entries, *, exact=True):
    path = Path(folder)/'RESUME_RESOLUTION.json'
    if path.exists():
        report = read_json(path)
        if report.get('next_sample_ids') is None:
            report.update(next_sample_ids=[e['scene_id'] if isinstance(e, dict) else e for e in entries],
                          exact_resume=exact, status='RESTORED')
            atomic_json(path, report)
