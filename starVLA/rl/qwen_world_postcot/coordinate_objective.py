"""Explicit coordinate-only IL study; default World loss remains untouched.

Weights act on the normalized clean-x residual, not the actor input or sampler.
Physical diagnostics use the existing affine codec; angular training semantics
are deliberately unchanged in this first single-factor comparison.
"""
import json
import ast
import math
from pathlib import Path

import torch

from .contracts import file_sha, fingerprint

SLOPES = (33.37, 21., 1.765)


def review_loading_adapter(reference, current):
    """Only historical explicit-RL-parent selection additions may differ.

    All IL inspection/loading functions and imports must retain the same AST.
    The old file's full hash is separately bound to the actual parent identity.
    """
    def core(path):
        tree=ast.parse(Path(path).read_text())
        tree.body=[node for node in tree.body if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))
                   or node.name not in ('validate_rl_parent_selection','_validate_explicit_40k_parent')]
        return ast.dump(tree,include_attributes=False)
    if core(reference)!=core(current):
        raise ValueError('IL checkpoint loader has an unreviewed computation change')
    return dict(reference=str(Path(reference).resolve()),reference_sha256=file_sha(reference),
        current_sha256=file_sha(current),all_non_RL_selection_AST_equal=True)


def tolerance_weights(tolerances=(.5, .25, math.radians(2))):
    if len(tolerances) != 3 or any(not math.isfinite(x) or x <= 0 for x in tolerances):
        raise ValueError('Three finite positive physical tolerances required')
    raw = [s / t for s, t in zip(SLOPES, tolerances)]
    return [x / (sum(raw) / 3) for x in raw]


class CoordinateObjective:
    """Parameter-free objective; never adds deployment state or parameters."""
    def __init__(self, weights, scale=1.):
        if len(weights) != 3 or any(not math.isfinite(x) or x <= 0 for x in weights):
            raise ValueError('Invalid positive coordinate weights')
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('Invalid fixed calibration scale')
        self.weights, self.scale = tuple(weights), float(scale)

    def __call__(self, prediction, target, valid=None):
        from starVLA.model.modules.qwen_world_postcot.flow import clean_x_l1
        _, stats = clean_x_l1(prediction, target, valid, return_statistics=True)
        mask = torch.ones_like(target, dtype=torch.bool) if valid is None else valid.detach().bool()
        if mask.shape == target.shape[:2]:
            mask = mask[:, :, None].expand_as(target)
        if mask.shape != target.shape:
            raise ValueError('Misaligned coordinate validity')
        error = torch.where(mask, prediction.float() - target.float(), torch.zeros_like(prediction).float()).abs()
        weights = error.new_tensor(self.weights)
        stats['numerator'] = (error * weights).sum() * self.scale
        diagnostics = dict(normalized_sum=error.detach().sum(0), valid_count=mask.sum(0).detach().float())
        stats['coordinate_diagnostics'] = diagnostics
        return stats['numerator'] / stats['denominator'].clamp_min(1), stats


def read_registration(path, arm):
    row = json.loads(Path(path).read_text()); body = dict(row)
    if body.pop('sha256', None) != fingerprint(body):
        raise ValueError('Changed coordinate study registration')
    if row.get('schema')=='world_IL_loss_designs_v1':
        from .loss_designs import read_loss_registration
        return read_loss_registration(row,arm)
    if (row['schema'] != 'world_coordinate_loss_study_v1' or arm not in ('CONTROL', 'PHYSICAL')
            or row['parent_updates'] != 63000 or row['added_updates'] != 2000
            or row['seed'] != 42 or row['evaluation_seed'] != 2101
            or row['horizon_weights'] != [1.] * 8 or not row['freeze_parent_candidate_bank']):
        raise ValueError('Coordinate-only study contract changed')
    if row['arms']['CONTROL'] != dict(weights=[1., 1., 1.], scale=1.):
        raise ValueError('Control is not original mean L1')
    expected = tolerance_weights(row['physical_tolerances'])
    if row['arms']['PHYSICAL']['weights'] != expected:
        raise ValueError('Tolerance/weight mapping changed')
    review=row['loading_adapter_review']
    current=Path(__file__).resolve().parent/'loading.py'
    if review_loading_adapter(review['reference'],current)!=review:
        raise ValueError('Reviewed loading adapter source identity changed')
    return row


def restore_fork(parent, engine, identity, builder, registration, output):
    """Explicit scientific fork, with native Adam/scheduler/RNG preserved.

    Same-run resume still goes through checkpoint.restore's exact identity test.
    This path is permitted only at the registered immutable parent boundary.
    """
    from . import checkpoint as cp
    from .loading import inspect_checkpoint
    info = cp._rank_zero_call(lambda: inspect_checkpoint(parent))
    old = info['identity']
    if (str(Path(parent).resolve()) != registration['parent_checkpoint']
            or info['model_sha256'] != registration['parent_model_sha256']
            or fingerprint(old) != registration['parent_identity_sha256']
            or info['seal']['updates'] != registration['parent_updates']):
        raise ValueError('Study parent identity changed')
    # Every scientific/data/numerical/optimizer field is retained. The only
    # allowed new metadata is this named experiment and reviewed source edits.
    projected = {k: v for k, v in identity.items() if k != 'coordinate_experiment'}
    projected['source_files'] = old['source_files']
    projected['qualification'] = old['qualification']
    if projected != old:
        raise ValueError('Unregistered parent fork differences: ' + repr(cp.identity_diff(old, projected)))
    allowed = {'tools/qwen_world_postcot/train_il.py',
               'starVLA/model/modules/qwen_world_postcot/model.py',
               'starVLA/model/modules/qwen_world_postcot/flow.py',
               'starVLA/rl/qwen_world_postcot/contracts.py'}
    loading='starVLA/rl/qwen_world_postcot/loading.py'
    review=registration['loading_adapter_review']
    if (review['reference_sha256']!=old['source_files'][loading]
            or review['current_sha256']!=identity['source_files'][loading]):
        raise ValueError('Loading adapter audit does not bind parent and current source')
    allowed.add(loading)
    changed = {k for k in old['source_files'] if identity['source_files'].get(k) != old['source_files'][k]}
    if not changed <= allowed or set(old['source_files']) - set(identity['source_files']):
        raise ValueError('Unexpected imported source changed: ' + repr(changed))
    receipt = dict(schema='explicit_coordinate_objective_scientific_fork_v1',
        parent_checkpoint=str(parent), parent_identity_sha256=fingerprint(old),
        current_identity_sha256=fingerprint(identity), source_differences=sorted(changed),
        optimizer_reset=False, scheduler_reset=False, data_cursor_reset=False,
        actor_or_sampling_architecture_changed=False, registration_sha256=registration['sha256'])
    loaded, _ = engine.load_checkpoint(str(parent), tag='native', load_optimizer_states=True,
        load_lr_scheduler_states=True, load_module_strict=True, load_module_only=False)
    if loaded is None:
        raise RuntimeError('Native parent model/Adam/scheduler failed to load')
    client = torch.load(Path(parent) / f'rank{cp.rank():02d}_state.pt', map_location='cpu', weights_only=False)
    if client['identity_sha256'] != fingerprint(old):
        raise ValueError('Parent rank state identity changed')
    if any(client['state'][k] != info['seal'][k] for k in ('updates', 'cursor', 'phase')):
        raise ValueError('Parent rank boundary differs from sealed boundary')
    if engine.global_steps != client['state']['updates']:
        raise ValueError('Restored native optimizer and logical update differ')
    builder.load_state_dict(client['builder']); cp.restore_rng(client['rng'])
    if builder.bank.bank_hash != registration['candidate_bank_hash']:
        raise ValueError('Parent auxiliary bank changed')
    cp._rank_zero_call(lambda: cp.atomic_json(Path(output) / 'FORK_RECEIPT.json', receipt))
    return client['state']
