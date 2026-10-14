"""Version a user-requested mining change before the first expert update.

Old manifests, Base binding, split roles, features and labels remain immutable.
Only the selected expert data pointer changes, after all new GT audits finish.
"""
from datetime import datetime, timezone
from pathlib import Path
from iqe.contracts import require
from iqe.io import read_json, atomic_json, file_hash, lock

ARTIFACTS = ('EXPERT_MANIFEST.json', 'EXPERT_TRAIN_MANIFEST.json',
             'HARD_SETS_BEFORE_EXPERT.json', 'TARGET_AUDIT_QUEUE.json')


def training_root(config, kind='expert'):
    if kind != 'expert' or not (config.root/'SELECTED_EXPERT_DATA.json').exists():
        return config.root/kind
    selected = read_json(config.root/'SELECTED_EXPERT_DATA.json')
    if 'training_root' not in selected:
        return config.root/kind
    result = Path(selected['training_root'])
    require(result.resolve().is_relative_to((config.root/'expert_runs').resolve()), 'foreign expert training root')
    require(file_hash(result/'DATA_TRANSITION.json') == selected['transition_hash'], 'expert data transition changed')
    transition = read_json(result/'DATA_TRANSITION.json')
    require(transition['base_binding_hash'] == file_hash(config.root/'BASE_BINDING.json')
            and transition['partition_hash'] == file_hash(config.root/'DATA_PARTITION.json'), 'expert transition dependencies changed')
    require(transition['new_manifest_hash'] == selected['artifacts']['EXPERT_MANIFEST.json']
            and Path(transition['training_root']).resolve() == result.resolve(), 'expert transition selected identity mismatch')
    return result


def continuation_dependencies(config):
    root = training_root(config)
    return {'expert_data_transition': file_hash(root/'DATA_TRANSITION.json')} if (root/'DATA_TRANSITION.json').exists() else {}


def workflow_identity_path(config):
    root = training_root(config)
    return (root if root != config.root/'expert' else config.root)/'WORKFLOW_CODE_IDENTITY.json'


def artifact(config, name):
    require(name in ARTIFACTS, 'unsupported expert data artifact')
    pointer = config.root/'SELECTED_EXPERT_DATA.json'
    if not pointer.exists():
        return config.root/name
    selected = read_json(pointer)
    require(selected['base_binding_hash'] == file_hash(config.root/'BASE_BINDING.json') and
            selected['partition_hash'] == file_hash(config.root/'DATA_PARTITION.json'), 'expert data dependency changed')
    folder = Path(selected['directory'])
    require(folder.resolve().is_relative_to((config.root/'expert_data_revisions').resolve()), 'foreign expert data directory')
    path = folder/name
    require(file_hash(path) == selected['artifacts'][name], 'selected expert data checksum mismatch')
    return path


def selected_summary(config):
    if not (config.root/'SELECTED_EXPERT_DATA.json').exists():
        return None
    manifest = read_json(artifact(config, 'EXPERT_MANIFEST.json'))
    sets = read_json(artifact(config, 'HARD_SETS_BEFORE_EXPERT.json'))
    return {'eligible_hard': len(manifest['hard']), 'ordinary': len(manifest['ordinary']),
            'failure_train': len(sets['F_train']), 'failure_val': len(sets['F_val']),
            'hard_val': len(sets['H_val']), 'base_failure_threshold': manifest['base_failure_threshold'], 'reused': True}


def revise(config, threshold, revision_id):
    require(revision_id and '/' not in revision_id and revision_id not in {'.', '..'}, 'invalid revision ID')
    require(threshold in {.8, .85, .9}, 'only explicitly bounded requested thresholds')
    with lock(config.root/'RUN.lock'):
        folder = config.root/'expert_data_revisions'/revision_id
        request_path = folder/'REQUEST.json'
        request = {'revision_id': revision_id, 'base_failure_threshold': threshold,
                   'gt_minimum': .8, 'minimum_gt_gain': .05, 'sampling': {'hard': .8, 'ordinary': .2},
                   'base_binding_hash': file_hash(config.root/'BASE_BINDING.json'),
                   'partition_hash': file_hash(config.root/'DATA_PARTITION.json'),
                   'base_scores_sha256': file_hash(config.root/'BASE_FIT_VAL_SCORES.json'),
                   'previous_manifest_hash': file_hash(config.root/'EXPERT_MANIFEST.json'),
                   'before_any_expert_updates': True, 'dev_test_used': False}
        if request_path.exists():
            require(read_json(request_path) == request, 'mining revision input changed')
        # No post-training threshold revision can silently change this experiment.
        training = config.root/'expert'
        require(not any(training.glob('step_*.pt*')) and
                not any((training/name).exists() for name in ('result.json', 'TRAINING_BUDGET.json', 'status.json')) and
                not ((training/'steps.jsonl').exists() and (training/'steps.jsonl').stat().st_size),
                'expert already started; use a new experiment')
        atomic_json(request_path, request, immutable=True)
        atomic_json(folder/'STATUS.json', {'status': 'AUDITING_TARGETS', 'utc': datetime.now(timezone.utc).isoformat()})
        from .bank import build_threshold_hard_set
        result = build_threshold_hard_set(config, failure_threshold=threshold, output_dir=folder,
                                         verified_base_scores_hash=request['base_scores_sha256'])
        selected = {'schema': 'iqe_pretraining_expert_data_revision_v1', 'directory': str(folder),
                    'revision_id': revision_id, 'request_hash': file_hash(request_path),
                    'base_binding_hash': request['base_binding_hash'], 'partition_hash': request['partition_hash'],
                    'artifacts': {name: file_hash(folder/name) for name in ARTIFACTS}, 'result': result}
        atomic_json(folder/'COMPLETE.json', selected, immutable=True)
        atomic_json(config.root/'SELECTED_EXPERT_DATA.json', selected)
        atomic_json(folder/'STATUS.json', {'status': 'COMPLETE', 'result': result, 'utc': datetime.now(timezone.utc).isoformat()})
        return result
