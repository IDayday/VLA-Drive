"""Explicit ancestry for an execution-only upgrade; no relaxed weight loading."""
import json
from pathlib import Path


def read_origin(run):
    run = Path(run)
    parent = json.loads((run/'identity.json').read_text())
    tag = (run/'checkpoints/latest').read_text().strip()
    if Path(tag).name != tag:
        raise ValueError('Unsafe parent checkpoint tag')
    complete = json.loads((run/'checkpoints'/tag/'COMPLETE.json').read_text())
    if complete['identity'] != parent['identity'] or complete['status'] != 'PAUSED':
        raise ValueError('Execution migration requires a complete paused parent checkpoint')
    origin = {'run': str(run.resolve()), 'tag': tag, 'identity': parent['identity'],
        'training_source_sha': parent['training_source_sha'], 'completed': complete['completed'],
        'file_sha256': complete['file_sha256']}
    return origin, parent, complete


def verify_scientific_contract(parent, child):
    # All meaningful model/data/optimization settings must be identical. Only
    # source identities, runtime checkpointing/prefetch and provenance differ.
    for name in ('schema', 'scope', 'config', 'dataset', 'cache_identity', 'current_and_GT_identity',
                 'DINO_identity', 'shared_geometry_identity', 'training_seed', 'world_size',
                 'initialization_assets', 'effective_global_batch', 'FM_repeat', 'microbatch',
                 'schedule_horizon', 'schedule_warmup', 'scenes_per_epoch', 'updates_per_epoch',
                 'source_population_scenes', 'engineering_training_selection', 'observation_plan',
                 'precision', 'determinism', 'external_source_versions', 'external_source_addendum'):
        if parent.get(name) != child.get(name):
            raise ValueError('Execution-only recovery changed scientific setting: '+name)
    if parent.get('execution_mode', 'reference') not in ('reference', 'io_preserving_v1', 'loss_preserving_v1'):
        raise ValueError('Unknown parent execution implementation')
    if child['execution_mode'] not in ('io_preserving_v1', 'loss_preserving_v1') or child['extra_stage_instrumentation']:
        raise ValueError('Formal recovery upgrade requires the validated common execution mode')
    if parent.get('extra_stage_instrumentation'):
        raise ValueError('Instrumented engineering checkpoints cannot initialize formal recovery')
