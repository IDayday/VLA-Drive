"""Explicit, state-preserving expert data change at an optimizer boundary.

The old experiment artifacts and checkpoint are never rewritten. Only the
manifest-dependent sampling population and validation bookkeeping change.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import torch

from iqe.contracts import require
from iqe.io import atomic_bytes, atomic_json, atomic_torch, digest, file_hash, lock, read_json
from .expert_data_revision import ARTIFACTS, artifact, training_root
from .sampling import ExpertSampler


PRESERVED_FIELDS = ('module', 'optimizer', 'scheduler', 'scaler', 'optimizer_step',
                    'consumed_sample_ids', 'rng_by_rank', 'world_size', 'global_batch',
                    'accumulation', 'resolved_config', 'environment', 'resume_boundary')


def state_hash(value):
    """Content hash for nested optimizer/RNG state, independent of torch archives."""
    h = hashlib.sha256()
    def visit(x):
        h.update(type(x).__name__.encode()+b':')
        if isinstance(x, torch.Tensor):
            y = x.detach().cpu().contiguous()
            h.update(str((y.dtype, tuple(y.shape))).encode())
            h.update(y.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(str((x.dtype, x.shape)).encode()); h.update(x.tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=repr): visit(k); visit(x[k])
        elif isinstance(x, (tuple, list)):
            for y in x: visit(y)
        else:
            h.update(repr(x).encode())
        h.update(b';')
    visit(value)
    return h.hexdigest()


def positive_gain_artifacts(previous, previous_sets, paused_step):
    """Reselect from previously verified real labels; no model/scoring calls."""
    from .bank import failure_and_target_audit
    require(previous['base_failure_threshold'] == .85 and previous['gt_minimum'] == .8
            and previous['target_gain_minimum'] == .05, 'unsupported parent mining rule')
    require(previous['sampling'] == {'hard': .8, 'ordinary': .2}, 'parent mixture changed')
    manifest, sets = deepcopy(previous), deepcopy(previous_sets)
    audits = manifest['gt_audit']
    for role in ('train', 'val'):
        selected = []
        for sid in sets['F_'+role]:
            old = audits[sid]
            updated = failure_and_target_audit(old['Base'], old['GT'],
                                               failure_threshold=.85, minimum_gt_gain=0.)
            require(updated['failure'], 'previous failure population changed')
            require(old.get('input_consistency') == 'verified', 'unverified target input')
            updated.update({k: old[k] for k in ('Base', 'GT', 'input_consistency', 'target_hash')})
            audits[sid] = updated
            if updated['eligible']: selected.append(sid)
        require(set(previous_sets['H_'+role]) <= set(selected), 'positive gain unexpectedly removes old hard samples')
        sets['H_'+role] = selected
    manifest.update(hard=sets['H_train'], hard_val=sets['H_val'], target_gain_minimum=0.,
                    target_gain_strict=True, revised_after_optimizer_step=paused_step)
    require(manifest['hard'] and manifest['ordinary'], 'empty H/N population')
    manifest['hash'] = digest({k:v for k,v in manifest.items() if k != 'hash'})
    sets['rules'].update(target_gain=0., target_gain_strict=True)
    sets.update(defined_before_expert_training=False, defined_after_optimizer_step=paused_step,
                change='user_requested_strict_positive_GT_gain')
    return {'EXPERT_MANIFEST.json': manifest, 'EXPERT_TRAIN_MANIFEST.json': manifest,
            'HARD_SETS_BEFORE_EXPERT.json': sets,
            'TARGET_AUDIT_QUEUE.json': {sid:a for sid,a in audits.items() if a['reasons']}}


def converted_state(source, old_manifest, new_manifest, dependencies, transition_hash):
    """No optimizer construction: retain moments, RNG and already-consumed IDs."""
    old = ExpertSampler(old_manifest['hard'], old_manifest['ordinary'], old_manifest['seed'],
                        old_manifest['hash'], old_manifest['sampling']['hard'])
    old.load_state_dict(source['sampler'])
    require(source['sampler']['consumed'] == len(source['consumed_sample_ids'])
            == source['optimizer_step']*source['global_batch'], 'consumed boundary mismatch')
    new = ExpertSampler(new_manifest['hard'], new_manifest['ordinary'], new_manifest['seed'],
                        new_manifest['hash'], new_manifest['sampling']['hard'])
    new.load_state_dict({**source['sampler'], 'identity': new.identity})
    result = deepcopy(source)
    result['sampler'] = new.state_dict()
    result['dependencies'] = dependencies
    result['trainer_state'] = {'best': None, 'bad_cycles': 0, 'best_snapshot': None,
        'termination': {'termination_reason': 'RUNNING_CHECKPOINT', 'terminal': False,
                        'stopped_at_step': source['optimizer_step'],
                        'early_stop_patience': source['resolved_config']['early_stop_patience'],
                        'best_snapshot': None}}
    result['data_transition'] = {'hash': transition_hash, 'distribution_changed': True,
                                'optimizer_state_preserved': True, 'parent_step': source['optimizer_step']}
    for field in PRESERVED_FIELDS:
        require(state_hash(result[field]) == state_hash(source[field]), 'state changed: '+field)
    require(result['sampler']['rng'] == source['sampler']['rng'], 'sampler RNG changed')
    return result


def revise_positive_gain(config, revision_id, parent_checkpoint):
    """Commit new manifest + derived checkpoint, then atomically select them."""
    require(revision_id and Path(revision_id).name == revision_id and revision_id not in {'.', '..'}, 'invalid revision ID')
    parent_checkpoint = Path(parent_checkpoint).resolve()
    folder = config.root/'expert_data_revisions'/revision_id
    destination = config.root/'expert_runs'/revision_id
    with lock(config.root/'RUN.lock'):
        require(read_json(config.root/'runtime/HOLD_HEAD_TRAINING.json')['active'], 'training hold required during data transition')
        if (folder/'COMPLETE.json').exists():
            selected = read_json(folder/'COMPLETE.json')
            transition = read_json(destination/'DATA_TRANSITION.json')
            require(transition['parent_checkpoint']['path'] == str(parent_checkpoint), 'different transition parent')
            require(file_hash(parent_checkpoint) == transition['parent_checkpoint']['checksum'], 'parent checkpoint changed')
            for name, checksum in selected['artifacts'].items():
                require(file_hash(folder/name) == checksum, 'transition artifact corrupt')
            require(file_hash(destination/'DATA_TRANSITION.json') == selected['transition_hash'], 'transition changed')
            from iqe.training.resume import inspect_checkpoint
            inspected = inspect_checkpoint(destination/parent_checkpoint.name)
            require(inspected['checksum'] == read_json(destination/'STATE_PRESERVATION_AUDIT.json')['derived']['checksum'],
                    'derived transition checkpoint changed')
            current = read_json(config.root/'SELECTED_EXPERT_DATA.json')
            require(current in (selected, read_json(folder/'PREVIOUS_SELECTION.json')),
                    'another revision has been selected; refusing pointer rollback')
            atomic_json(config.root/'SELECTED_EXPERT_DATA.json', selected)
            return selected
        previous_path = folder/'PREVIOUS_SELECTION.json'
        if previous_path.exists():
            previous = read_json(previous_path)
            require(read_json(config.root/'SELECTED_EXPERT_DATA.json') == previous, 'selected expert data changed during transaction')
        else:
            previous = read_json(config.root/'SELECTED_EXPERT_DATA.json')
            atomic_json(previous_path, previous, immutable=True)
        previous_root = training_root(config)
        require(parent_checkpoint.parent == previous_root.resolve(), 'parent outside selected training lineage')
        from iqe.training.resume import inspect_checkpoint
        from .training import training_config
        from .budget import training_dependencies
        old_manifest_path = artifact(config, 'EXPERT_MANIFEST.json')
        old_manifest = read_json(old_manifest_path)
        old_sets = read_json(artifact(config, 'HARD_SETS_BEFORE_EXPERT.json'))
        old_deps = {'experiment_id': config.experiment_id, 'base': file_hash(config.root/'BASE_BINDING.json'),
                    'manifest': file_hash(old_manifest_path), 'kind': 'own_expert_1', **training_dependencies(config)}
        inspected = inspect_checkpoint(parent_checkpoint, old_deps, training_config(config, 'expert'))
        source = torch.load(parent_checkpoint, map_location='cpu', weights_only=False)
        require(source['trainer_state']['termination']['termination_reason'] == 'OPERATOR_PAUSED', 'cooperatively paused parent required')
        require(read_json(previous_root/'result.json')['checkpoint'] == str(parent_checkpoint), 'parent is not the latest paused result')
        require(not list((config.root/'scorer').glob('step_*.pt*')) and not (config.root/'PAIR_BANK.json').exists(),
                'Scorer/bank already bound; new experiment required')
        artifacts = positive_gain_artifacts(old_manifest, old_sets, inspected['optimizer_step'])
        for name, value in artifacts.items(): atomic_json(folder/name, value, immutable=True)
        new_manifest = artifacts['EXPERT_MANIFEST.json']
        transition = {'schema': 'iqe_state_preserving_expert_data_transition_v1', 'revision_id': revision_id,
            'user_rule': 'GT strictly higher than fixed Base output; remove only 0.05 gain margin',
            'base_binding_hash': old_deps['base'], 'partition_hash': file_hash(config.root/'DATA_PARTITION.json'),
            'parent_checkpoint': inspected, 'parent_receipt_hash': file_hash(str(parent_checkpoint)+'.COMPLETE.json'),
            'parent_training_root': str(previous_root), 'parent_steps_hash': file_hash(previous_root/'steps.jsonl'),
            'parent_manifest': str(old_manifest_path), 'parent_manifest_hash': old_deps['manifest'],
            'new_manifest': str(folder/'EXPERT_MANIFEST.json'), 'new_manifest_hash': file_hash(folder/'EXPERT_MANIFEST.json'),
            'training_root': str(destination), 'continued_optimizer_step': source['optimizer_step'],
            'consumed_samples_preserved': len(source['consumed_sample_ids']),
            'state_hashes_preserved': {k:state_hash(source[k]) for k in PRESERVED_FIELDS},
            'sampler_rng_hash': state_hash(source['sampler']['rng']), 'old_sampler_identity': source['sampler']['identity'],
            'validation_bookkeeping_reset': True, 'previous_best_snapshot': source['trainer_state'].get('best_snapshot'),
            'validation_reset_reason': 'H_val changed; new-data validation starts at next scheduled validation',
            'same_data_exact_continuation': False, 'optimizer_or_scheduler_reset': False,
            'new_VLM_calls': 0, 'new_official_scoring_calls': 0}
        atomic_json(destination/'DATA_TRANSITION.json', transition, immutable=True)
        transition_hash = file_hash(destination/'DATA_TRANSITION.json')
        dependencies = {**old_deps, 'manifest': transition['new_manifest_hash'], 'expert_data_transition': transition_hash}
        state = converted_state(source, old_manifest, new_manifest, dependencies, transition_hash)
        checkpoint = destination/parent_checkpoint.name
        if checkpoint.exists():
            existing = torch.load(checkpoint, map_location='cpu', weights_only=False)
            require(state_hash(existing) == state_hash(state), 'conflicting derived checkpoint')
        else:
            atomic_torch(checkpoint, state, immutable=True)
        atomic_json(str(checkpoint)+'.COMPLETE.json', {'checksum': file_hash(checkpoint),
            'optimizer_step': source['optimizer_step'], 'world_size': source['world_size'], 'dependencies': dependencies}, immutable=True)
        inspection = inspect_checkpoint(checkpoint, dependencies, training_config(config, 'expert'))
        persisted = torch.load(checkpoint, map_location='cpu', weights_only=False)
        require(state_hash(persisted) == state_hash(state), 'checkpoint serialization changed transition')
        require(file_hash(parent_checkpoint) == inspected['checksum'], 'parent modified')
        audit = {'status': 'PASS', 'parent': inspected, 'derived': inspection,
                 'preserved_fields': list(PRESERVED_FIELDS), 'preserved_content_hashes': transition['state_hashes_preserved'],
                 'sampler_rng_and_consumed_preserved': True, 'new_sampler_identity': state['sampler']['identity'],
                 'validation_reset_only': True, 'optimizer_reset': False, 'same_data_exact_continuation': False}
        atomic_json(destination/'STATE_PRESERVATION_AUDIT.json', audit, immutable=True)
        # Preserve the consumed history verbatim. Subsequent appends start at
        # parent_step+1; audits replay each segment under its own manifest.
        atomic_bytes(destination/'steps.jsonl', (previous_root/'steps.jsonl').read_bytes(), immutable=True)
        sets = artifacts['HARD_SETS_BEFORE_EXPERT.json']
        selected = {'schema': 'iqe_state_preserving_expert_data_revision_v2', 'revision_id': revision_id,
            'directory': str(folder), 'training_root': str(destination), 'transition_hash': transition_hash,
            'base_binding_hash': old_deps['base'], 'partition_hash': transition['partition_hash'],
            'parent_selection_hash': file_hash(previous_path), 'artifacts': {n:file_hash(folder/n) for n in ARTIFACTS},
            'result': {'eligible_hard': len(new_manifest['hard']), 'ordinary': len(new_manifest['ordinary']),
                'failure_train': len(sets['F_train']), 'failure_val': len(sets['F_val']), 'hard_val': len(sets['H_val'])}}
        atomic_json(folder/'COMPLETE.json', selected, immutable=True)
        atomic_json(config.root/'SELECTED_EXPERT_DATA.json', selected)
        return selected


def replay_expert_history(config, manifest, rows, replay):
    """Verify old and new sample IDs without pretending the population stayed fixed."""
    root = training_root(config)
    new = ExpertSampler(manifest['hard'], manifest['ordinary'], manifest['seed'], manifest['hash'],
                        manifest['sampling']['hard'])
    if not (root/'DATA_TRANSITION.json').exists():
        return replay(new, rows)
    transition = read_json(root/'DATA_TRANSITION.json')
    old_path = Path(transition['parent_manifest'])
    require(file_hash(old_path) == transition['parent_manifest_hash'], 'parent expert manifest changed')
    old_manifest = read_json(old_path)
    old = ExpertSampler(old_manifest['hard'], old_manifest['ordinary'], old_manifest['seed'],
                        old_manifest['hash'], old_manifest['sampling']['hard'])
    boundary = transition['continued_optimizer_step']
    counts = replay(old, rows[:boundary])
    require(state_hash(old.state_dict()['rng']) == transition['sampler_rng_hash'], 'parent sampler replay changed')
    require(old.consumed == transition['consumed_samples_preserved'], 'parent exposure mismatch')
    new.load_state_dict({**old.state_dict(), 'identity': new.identity})
    for bucket, additions in replay(new, rows[boundary:]).items():
        counts.setdefault(bucket, additions.__class__()).update(additions)
    return counts
