"""Independent preregistered forward producer alongside UNCHANGED strict.

One new eight-rank, no-update native probe; never convert historical maxima to
elementwise acceptance. Original validator writes --output/strict and retains
its assertion/exit failure. Independent compact statistics go to
--output/FORWARD_COMPARISON.json. No image/raw tensor/gradient bank is saved.
Launch only from a separately frozen source on parent-admitted GPUs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import functools
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
REGISTRATION = ROOT/'reports/structured_world/execution_20261011/INDEPENDENT_ACCEPTANCE_PREREGISTRATION_V1.json'
REGISTRATION_SHA = '8bc0cc1dec82c88cc73e8cde4e5e9eac1d4d88633339c6fcd6670a1d984a8007'
LEGACY = ROOT/'tools/structured_world/validate_engineering_execution.py'
LEGACY_SHA = '31dc8caf31df0daa1e96c074569d1d9cff39ca80ca522f94ae9331534e8d2ffd'
PAIRS = {'A_A': ('A', 'AA'), 'A_B': ('A', 'B'), 'AA_B': ('AA', 'B')}
CORE = {'H_A', 'B0', 'Bt', 'q0', 'q_final', 'loss'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registration(path=REGISTRATION, protected_original_root=None):
    require(digest(path) == REGISTRATION_SHA, 'Preregistration SHA256 mismatch; no threshold override')
    policy = json.loads(Path(path).read_text())
    bounds = policy['tiers']['E2']['forward_by_mode']['engineering_timevec']['module_tolerances']
    require(bounds['atol'] == 1e-6 and bounds['rtol'] == 1e-4 and bounds['relative_L2_max'] == 1e-4
            and bounds['physical_q_final_max_position_error_m'] == 1e-4, 'Unknown prospective forward bounds')
    # Preserve all historical protected checks. Current capture-only trainer
    # changes are handled separately by a prospective producer-source binding.
    if protected_original_root is not None:
        require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=protected_original_root, text=True).strip()
                == policy['frozen_comparator_commit'], 'Original root is not the preregistered comparator commit')
    for record in policy['protected_files']:
        if protected_original_root is None:
            raw = subprocess.check_output(['git', 'show', policy['frozen_comparator_commit']+':'+record['path']], cwd=ROOT)
            actual = hashlib.sha256(raw).hexdigest()
        else:
            actual = digest(Path(protected_original_root)/record['path'])
        require(actual == record['sha256'], 'Historical protected strict source changed: '+record['path'])
    return policy, bounds


CAPTURE_REASON = 'pilot-native-observation/precision-metadata, noalgorithm'
BINDING_FILES = ('tools/structured_world/validate_execution_numerics.py',
                 'tools/structured_world/validate_engineering_execution.py',
                 'tools/structured_world/train_vla.py',
                 'tools/structured_world/compare_execution_checkpoints.py',
                 'tools/structured_world/validate_execution_equivalence.py',
                 'starVLA/model/framework/vla_structured_fgtr.py',
                 'starVLA/model/framework/DDPForesight.py',
                 'starVLA/model/modules/structured_world/execution.py',
                 'starVLA/model/modules/structured_world/fgtr.py',
                 'starVLA/model/modules/structured_world/future.py',
                 'starVLA/model/modules/structured_world/geo_bev.py',
                 'starVLA/model/modules/structured_world/queries.py',
                 'starVLA/model/modules/structured_world/losses.py',
                 'starVLA/model/modules/structured_world/qwen_execution.py',
                 'starVLA/model/modules/action_model/GR00T_ActionHeader.py',
                 'starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit.py',
                 'starVLA/model/modules/vlm/qwen3_vl/modeling_qwen3_vl.py',
                 'starVLA/dataloader/structured_world/prepared_current.py',
                 'starVLA/dataloader/foresight_dataset.py')


def source_binding(path, expected_hash, policy):
    require(expected_hash is not None and digest(path) == expected_hash, 'Producer-source binding SHA256 mismatch')
    binding = json.loads(Path(path).read_text())
    require(binding['schema'] == 'execution_forward_producer_source_binding_v1'
            and binding['state'] == 'frozen_before_candidate', 'Explicit frozen producer-source binding required')
    require(binding['registration_sha256'] == REGISTRATION_SHA, 'Binding refers to a different preregistration')
    require(binding['candidate_mode'] == 'engineering_timevec' and binding['roles'] ==
            {'A': 'io_preserving_v1', 'AA': 'io_preserving_v1', 'B': 'engineering_timevec'},
            'Freeze the actual same-input role/mode assignment before candidate execution')
    frozen = datetime.fromisoformat(binding['frozen_UTC'].replace('Z', '+00:00'))
    require(frozen.tzinfo is not None and frozen <= datetime.now(timezone.utc), 'Binding must precede candidate execution')
    require(binding['source_git_sha'] == subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'Frozen producer source commit changed')
    require(not subprocess.check_output(['git', 'diff', '--name-only', 'HEAD', '--'], cwd=ROOT, text=True).strip(),
            'Producer must run from frozen committed source, without tracked edits')
    files = binding['files_sha256']
    require(set(BINDING_FILES) <= files.keys(), 'Binding omits required producer/algorithm/control source')
    for name, expected in files.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe bound source path')
        require(digest(ROOT/name) == expected, 'Bound producer source changed: '+name)
    require(files['tools/structured_world/validate_engineering_execution.py'] == LEGACY_SHA, 'Legacy strict validator changed')
    changes = {row['path']: row for row in binding['protected_source_changes']}
    require(len(changes) == len(binding['protected_source_changes']), 'Duplicate protected source exception')
    actual_changes = []
    for old in policy['protected_files']:
        current = digest(ROOT/old['path'])
        if current == old['sha256']:
            require(old['path'] not in changes, 'Unused protected source exception')
            continue
        require(old['path'] == 'tools/structured_world/train_vla.py' and old['path'] in changes,
                'Unapproved current protected source change: '+old['path'])
        row = changes[old['path']]
        require(row['historical_sha256'] == old['sha256'] and row['producer_sha256'] == current
                and row['reason'] == CAPTURE_REASON and row['algorithm_changed'] is False,
                'Protected trainer exception must bind the exact reviewed capture-only change')
        actual_changes.append(row)
    require(set(changes) == {row['path'] for row in actual_changes}, 'Unexpected protected source exception')
    return {'path': str(Path(path).resolve()), 'sha256': expected_hash, 'frozen_UTC': binding['frozen_UTC'],
            'source_git_sha': binding['source_git_sha'], 'files_sha256': files,
            'historical_protected_files': policy['protected_files'], 'protected_source_changes': actual_changes,
            'capture_only_reason_is_operator_reviewed_declaration': True}


def bitwise(a, b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        return (torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape
                and torch.equal(a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                                b.detach().cpu().contiguous().reshape(-1).view(torch.uint8)))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(bitwise(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and type(a) is type(b):
        return len(a) == len(b) and all(bitwise(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        import struct
        return struct.pack('!d', a) == struct.pack('!d', b)
    return type(a) is type(b) and a == b


def typed_tensor_hash(value):
    require(torch.is_tensor(value) and not value.requires_grad, 'Query hash requires an actual detached tensor')
    tensor = value.detach().cpu().contiguous()
    descriptor = {'dtype': str(tensor.dtype), 'shape': list(tensor.shape), 'elements': tensor.numel(), 'order': 'logical row-major'}
    encoded = json.dumps(descriptor, sort_keys=True, separators=(',', ':')).encode()
    result = hashlib.sha256(len(encoded).to_bytes(8, 'little')+encoded)
    result.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()).cast('B'))
    return {**descriptor, 'sha256': result.hexdigest(), 'detached': True}


def observe_queries(original, collector):
    @functools.wraps(original)
    def wrapped(q0, *args, **kwargs):
        value = original(q0, *args, **kwargs)
        collector.query(q0, value)
        return value
    return wrapped


def observe_counts(original, collector):
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        collector.before_counts()
        value = original(*args, **kwargs)
        collector.counts(value)
        return value
    return wrapped


def tensor_stats(reference, candidate, bounds):
    require(reference.shape == candidate.shape and reference.dtype == candidate.dtype, 'Forward tensor shape/dtype changed')
    a, b = reference.detach().cpu(), candidate.detach().cpu()
    require(a.numel() > 0 and a.is_floating_point(), 'Nonempty floating forward tensor required')
    finite = torch.isfinite(a) & torch.isfinite(b)
    bad = int((~finite).sum())
    close = torch.isclose(b, a, atol=bounds['atol'], rtol=bounds['rtol'], equal_nan=False)
    row = {'elements': a.numel(), 'nonfinite_elements': bad, 'atol': bounds['atol'], 'rtol': bounds['rtol'],
           'torch_allclose_passed': bool(torch.allclose(b, a, atol=bounds['atol'], rtol=bounds['rtol'], equal_nan=False)),
           'elementwise_violating_elements': int((~close).sum()), 'bitwise_equal': bitwise(a, b)}
    row['violating_elements'] = row['elementwise_violating_elements']
    if bad:
        row.update(reference_norm=None, candidate_norm=None, difference_norm=None, relative_L2=None,
                   max_absolute=None, zero_reference_norm=False, advisory_passed=False)
        return row
    da, db = a.double(), b.double()
    an, bn, dn = (float(torch.linalg.vector_norm(v.reshape(-1))) for v in (da, db, db-da))
    zero = an == 0.
    if zero:
        # Exact zero policy, even when an individual error fits the absolute
        # elementwise allowance. Never clamp a zero denominator with epsilon.
        row['violating_elements'] = int(torch.count_nonzero(b))
    relative = dn/an if not zero else 0. if dn == 0. else None
    row.update(reference_norm=an, candidate_norm=bn, difference_norm=dn, relative_L2=relative,
               max_absolute=float((db-da).abs().max()), zero_reference_norm=zero,
               advisory_passed=row['violating_elements'] == 0 and relative is not None and relative <= bounds['relative_L2_max'])
    return row


def physical_stats(reference, candidate):
    require(reference.shape == candidate.shape and reference.ndim == 3 and reference.shape[-1] == 3,
            'q_final must already be physical [scene,time,x/y/yaw], not normalized four-channel actions')
    a, b = reference.double(), candidate.double()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return {'finite': False, 'max_position_error_m': None, 'max_heading_error_rad': None}
    delta = b-a
    angle = torch.atan2(torch.sin(delta[..., 2]), torch.cos(delta[..., 2]))
    return {'finite': True, 'max_position_error_m': float(torch.linalg.vector_norm(delta[..., :2], dim=-1).max()),
            'max_heading_error_rad': float(angle.abs().max())}


def compact(value):
    # Only explicitly whitelisted query/count/scalar metadata reaches here.
    if torch.is_tensor(value):
        require(value.numel() <= 4096, 'Large tensor metadata is forbidden')
        value = value.detach().cpu().tolist()
    if isinstance(value, dict): return {str(k): compact(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [compact(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    require(value is None or type(value) in (str, bool, int, float), 'Unregistered metadata type')
    return value


class Collector:
    """Two transient CPU forward banks; representatives only for gradients."""
    def __init__(self, bounds, before_candidate=None):
        self.bounds, self.banks, self.rows, self.roles = bounds, {}, {}, {}
        self.result = None
        self.before_candidate = before_candidate
        self.global_counts, self.queries = {}, {}
        self.count_starts = {}

    def before_counts(self):
        role = ('A', 'AA', 'B')[len(self.roles)]
        if role == 'B' and self.before_candidate is not None:
            self.before_candidate()
        self.count_starts[role] = datetime.now(timezone.utc).isoformat()

    def counts(self, counts):
        role = ('A', 'AA', 'B')[len(self.roles)]
        require(role not in self.global_counts, 'Duplicate optimizer_batch_counts call')
        self.global_counts[role] = compact(counts)

    def query(self, q0, returned):
        require(self.role not in self.queries, 'Expected exactly one same-input query sampler call')
        self.queries[self.role] = {'call_count': 1, 'input_q0': typed_tensor_hash(q0),
                                   'poses': typed_tensor_hash(returned['poses']),
                                   'time_index': typed_tensor_hash(returned['time_index'])}

    def begin(self):
        role = ('A', 'AA', 'B')[len(self.roles)]
        require(role in self.count_starts, 'Missing observed global-count start')
        self.roles[role] = {'started_UTC': self.count_starts[role],
                            'forward_started_UTC': datetime.now(timezone.utc).isoformat(),
                            'started_UTC_definition': 'before native optimizer_batch_counts, after native RNG restore'}
        self.role = role

    def forward(self, result):
        values = {name: value for name, value in result['diagnostics'].items()}
        values['loss'] = result['loss']
        values.update({'task_loss/'+name: value for name, value in result['losses'].items()})
        values.update({'raw_loss/'+name: value for name, value in result['metrics']['raw'].items()})
        require(CORE <= values.keys(), 'Missing registered complete forward diagnostics')
        self.values = {name: value.detach().cpu().clone() for name, value in values.items()}
        self.metadata = {name: compact(result['metrics'][name]) for name in
                         ('effective_counts', 'query_strata', 'proposal_out_of_range', 'refine_effective_weight')}
        require(self.role in self.global_counts and self.role in self.queries, 'Missing actual query/global denominator observation')
        self.metadata['optimizer_batch_counts'] = self.global_counts[self.role]
        self.metadata['sample_queries_typed_hashes'] = self.queries[self.role]
        self.roles[self.role]['scalar_values'] = {name: compact(value) for name, value in self.values.items() if value.numel() == 1}
        self.roles[self.role]['query_and_count_metadata'] = self.metadata
        self.result = result

    def after_backward(self, model, rng):
        require(self.result is not None, 'Missing observed forward before RNG capture')
        gradients = {name: value.detach().cpu().clone() for name, value in self.result['metrics']['activation_gradient_samples'].items()}
        require(set(gradients) == {'H_A', 'B0', 'Bt'}, 'Missing registered representative activation gradients')
        buffers = {name: value.detach().cpu().clone() for name, value in model.named_buffers()}
        inventory = [(name, list(value.shape), str(value.dtype), value.requires_grad,
                      None if value.grad is None else [list(value.grad.shape), str(value.grad.dtype)])
                     for name, value in model.named_parameters()]
        bank = {'values': self.values, 'metadata': self.metadata, 'rng': rng, 'buffers': buffers,
                'gradients': gradients, 'inventory': inventory}
        role = self.role
        self.roles[role]['tensor_inventory'] = {name: {'shape': list(value.shape), 'dtype': str(value.dtype), 'elements': value.numel()}
                                               for name, value in self.values.items()}
        self.roles[role]['representative_gradient_elements'] = {name: value.numel() for name, value in gradients.items()}
        for pair, (left, right) in PAIRS.items():
            if right != role: continue
            old = self.banks[left]
            require(old['values'].keys() == bank['values'].keys(), 'Forward module inventory changed')
            bn = lambda values: {name: value for name, value in values.items()
                                 if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
            self.rows[pair] = {
                'modules': {name: tensor_stats(old['values'][name], value, self.bounds) for name, value in self.values.items()},
                'representative_gradients': {name: tensor_stats(old['gradients'][name], value, self.bounds) for name, value in gradients.items()},
                'physical_q_final': physical_stats(old['values']['q_final'], self.values['q_final']),
                'exact': {'q0': bitwise(old['values']['q0'], self.values['q0']) and bool(torch.isfinite(self.values['q0']).all()),
                          'RNG': bitwise(old['rng'], rng), 'BN': bitwise(bn(old['buffers']), bn(buffers)),
                          'BN_finite': all(bool(torch.isfinite(v).all()) for values in (old['buffers'], buffers) for v in bn(values).values()),
                          'buffer_inventory': [(n, list(v.shape), str(v.dtype)) for n,v in old['buffers'].items()] ==
                                              [(n, list(v.shape), str(v.dtype)) for n,v in buffers.items()],
                          'all_buffers': bitwise(old['buffers'], buffers),
                          'gradient_inventory': old['inventory'] == inventory,
                          'query_and_counts': bitwise(old['metadata'], self.metadata)}}
        if role != 'B': self.banks[role] = bank
        self.result = self.values = self.metadata = None
        if role == 'B': self.banks.clear()

    def report(self, rank, tokens):
        require(set(self.rows) == set(PAIRS), 'Incomplete A/AA/B collection')
        return {'rank': rank, 'tokens': tokens, 'roles': self.roles, 'comparisons': self.rows}


def aggregate_stat(rows, bounds):
    out = {name: sum(row[name] for row in rows) for name in
           ('elements', 'nonfinite_elements', 'elementwise_violating_elements', 'violating_elements')}
    out.update(atol=bounds['atol'], rtol=bounds['rtol'], torch_allclose_passed=all(row['torch_allclose_passed'] for row in rows),
               bitwise_equal=all(row['bitwise_equal'] for row in rows),
               all_rank_bounds_passed=all(row['advisory_passed'] for row in rows))
    if out['nonfinite_elements']:
        out.update(reference_norm=None, candidate_norm=None, difference_norm=None, relative_L2=None,
                   max_absolute=None, zero_reference_norm=False, advisory_passed=False)
        return out
    an, bn, dn = (math.sqrt(math.fsum(row[name]**2 for row in rows)) for name in
                  ('reference_norm', 'candidate_norm', 'difference_norm'))
    relative = dn/an if an else 0. if dn == 0. else None
    out.update(reference_norm=an, candidate_norm=bn, difference_norm=dn, relative_L2=relative,
               max_absolute=max(row['max_absolute'] for row in rows), zero_reference_norm=an == 0.,
               advisory_passed=out['all_rank_bounds_passed'] and out['violating_elements'] == 0
                               and relative is not None and relative <= bounds['relative_L2_max'])
    return out


def combine(records, bounds, strict_ref, *, expected_ranks=8):
    require(len(records) == expected_ranks and {r['rank'] for r in records} == set(range(expected_ranks)), 'Missing/duplicate rank coverage')
    names = set(records[0]['comparisons']['A_B']['modules'])
    require(CORE <= names, 'Missing complete core module inventory')
    require(all(set(r['comparisons']) == set(PAIRS) and all(set(r['comparisons'][p]['modules']) == names for p in PAIRS) for r in records),
            'Incomplete registered module/pair coverage')
    modules = {name: {pair: aggregate_stat([r['comparisons'][pair]['modules'][name] for r in records], bounds)
                     for pair in ('A_B', 'AA_B')} for name in sorted(names)}
    # Existing independent reviewer checks every module. Include full local
    # tensors as explicit modules too, so a small-reference rank cannot hide
    # behind a pooled denominator. Freeze these names in the case inventory.
    for record in sorted(records, key=lambda row: row['rank']):
        for name in sorted(names):
            modules[f"rank{record['rank']}/{name}"] = {pair: record['comparisons'][pair]['modules'][name]
                                                       for pair in ('A_B', 'AA_B')}
    physical = {'unit': 'm', 'heading_unit': 'rad', 'heading_rule': 'abs(atan2(sin(delta_yaw),cos(delta_yaw))); diagnostic, no new bound',
                'full_tensor_rank_coverage': True, 'conversion_source_sha256': digest(ROOT/'starVLA/dataloader/foresight_dataset.py')}
    for pair in PAIRS:
        rows = [r['comparisons'][pair]['physical_q_final'] for r in records]
        physical[pair] = {'finite': all(row['finite'] for row in rows),
                          **{name: max(row[name] for row in rows) if all(row['finite'] for row in rows) else None
                             for name in ('max_position_error_m', 'max_heading_error_rad')}}
    exacts = {pair: {name: all(r['comparisons'][pair]['exact'][name] for r in records)
                    for name in records[0]['comparisons'][pair]['exact']} for pair in PAIRS}
    numeric = all(row['advisory_passed'] for module in modules.values() for row in module.values())
    numeric = numeric and all(physical[p]['finite'] and physical[p]['max_position_error_m'] <= bounds['physical_q_final_max_position_error_m']
                              for p in ('A_B', 'AA_B'))
    numeric = numeric and all(all(row.values()) for row in exacts.values())
    return {'schema': 'execution_acceptance_forward_stats_v1', 'registration_sha256': REGISTRATION_SHA,
            'passed': False, 'formal_authorized': False, 'trainer_consumable': False,
            'advisory_forward_passed': numeric, 'full_tensor_rank_coverage': True, 'ranks': expected_ranks,
            'scene_count': sum(len(row['tokens']) for row in records),
            'logical_forward_module_inventory': sorted(names),
            'forward_module_inventory': sorted(modules),
            'modules': modules, 'physical_q_final': physical, 'forward_exact_by_pair': exacts,
            'forward_exact': {**{name: all(row[name] for row in exacts.values()) for name in exacts['A_B']},
                              'BN_float': all(row['BN'] for row in exacts.values()), 'BN_counter': all(row['BN'] for row in exacts.values())},
            'original_strict': strict_ref, 'per_rank': records,
            'representative_gradients_are_diagnostic_only': True,
            'limits': ['Numerical producer only; prospective bindings/complete E1/AA gradient budgets/dev/recovery reviewed separately',
                       'No retroactive classification from historical maximum-error summaries',
                       'Forward banks are transient CPU tensors; no raw arrays, images or gradient banks are persisted',
                       'Original full gradient strict guard, thresholds and failure exit remain unchanged']}


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'cache', 'dino-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--model-state-file', type=Path)
    parser.add_argument('--microbatch', type=int, choices=(2, 4), default=4)
    parser.add_argument('--registration', type=Path, default=REGISTRATION)
    parser.add_argument('--source-binding', type=Path, required=True)
    parser.add_argument('--source-binding-sha256', required=True)
    parser.add_argument('--protected-original-root', type=Path)
    parser.add_argument('--optimized-mode', choices=('engineering_timevec',), default='engineering_timevec')
    args = parser.parse_args()
    policy, bounds = registration(args.registration, args.protected_original_root)
    binding_proof = source_binding(args.source_binding, args.source_binding_sha256, policy)
    require(digest(LEGACY) == LEGACY_SHA, 'Legacy strict source changed; re-audit required')
    require(not args.output.exists(), 'Output must be a NEW directory; old strict artifacts cannot be overwritten')
    # Set this before importing the original GPU validator; no process is
    # launched here. --help and CPU unit tests do not import the model.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    legacy = importlib.import_module('tools.structured_world.validate_engineering_execution')
    collector = Collector(bounds, before_candidate=lambda: source_binding(args.source_binding, args.source_binding_sha256, policy))
    framework = importlib.import_module('starVLA.model.framework.vla_structured_fgtr')
    original_class, original_rng, original_dist = legacy.VLAStructuredFGTR, legacy.capture_rng, legacy.dist
    original_queries, original_counts = framework.sample_queries, legacy.optimizer_batch_counts
    model_holder = {}
    class ObservedModel(original_class):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw); model_holder['model'] = self
        def forward(self, *a, **kw):
            collector.begin()
            result = super().forward(*a, **kw)
            collector.forward(result)
            return result
    def captured_rng(generators):
        value = original_rng(generators)
        collector.after_backward(model_holder['model'], value)
        return value
    class DistributedObserver:
        def __getattr__(self, name): return getattr(original_dist, name)
        def all_gather_object(self, output, strict_record):
            original_dist.all_gather_object(output, strict_record)
            compact_records = [None]*len(output)
            original_dist.all_gather_object(compact_records, collector.report(strict_record['rank'], strict_record['tokens']))
            if strict_record['rank'] == 0:
                path = args.output/'strict'/'COMPLETE.json'
                # The original main writes COMPLETE after this hook. Bind it
                # later, without changing any field or swallowing its failure.
                model_holder['independent'] = combine(compact_records, bounds, {'path': str(path.resolve()), 'pending': True})
    legacy.VLAStructuredFGTR, legacy.capture_rng, legacy.dist = ObservedModel, captured_rng, DistributedObserver()
    legacy.optimizer_batch_counts, framework.sample_queries = observe_counts(original_counts, collector), observe_queries(original_queries, collector)
    old_argv = sys.argv
    forwarded = ['--run', str(args.run), '--tag', args.tag, '--cache', str(args.cache), '--dino-root', str(args.dino_root),
                 '--output', str(args.output/'strict'), '--microbatch', str(args.microbatch), '--optimized-mode', args.optimized_mode,
                 '--attention-diagnostic', 'native', '--precision-diagnostic', 'native']
    if args.model_state_file: forwarded += ['--model-state-file', str(args.model_state_file)]
    try:
        sys.argv = [str(LEGACY)]+forwarded
        legacy.main()  # Original strict failure is intentionally NOT caught.
    finally:
        sys.argv = old_argv
        legacy.VLAStructuredFGTR, legacy.capture_rng, legacy.dist = original_class, original_rng, original_dist
        legacy.optimizer_batch_counts, framework.sample_queries = original_counts, original_queries
        if 'independent' in model_holder:
            report = model_holder['independent']
            path = args.output/'strict'/'COMPLETE.json'
            require(path.is_file(), 'Original strict completion receipt missing')
            original = json.loads(path.read_text())
            report['original_strict'] = {'path': str(path.resolve()), 'sha256': digest(path), 'passed': original['passed'],
                                          'tool_sha256': LEGACY_SHA, 'unchanged_original_json': True}
            report['producer_sha256'] = digest(__file__)
            report['producer_source_binding'] = binding_proof
            report['candidate_mode'] = args.optimized_mode
            report['checkpoint'] = {'run': str(args.run.resolve()), 'tag': args.tag}
            write_new(args.output/'FORWARD_COMPARISON.json', report)


if __name__ == '__main__':
    main()
