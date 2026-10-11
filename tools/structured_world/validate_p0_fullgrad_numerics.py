"""Supplement cold 6d16 P0 observation with directly measured A/AA/B full grads.

No model/operator/optimizer update changes. Original be5 legacy main, thresholds,
strict JSON and failure exit are retained. CPU FP32 banks are transient; only
inventory and chunk-float64 statistics are saved. Freeze/register this separate
producer before a future probe; earlier 819d/44c receipts are not relabelled.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RELATIVE = 'tools/structured_world/validate_p0_fullgrad_numerics.py'
FORWARD_COMMIT = '6d16abf207877340afd36ecca58e6f1904d37d02'
FORWARD_SHA = '9c03229e16e4c055724e569c0d76b3138dfad6a99c1e10a08139a2de4ba99919'
FORWARD_BINDING_SHA = '44c78ea7aa563a4d6b863081ffd8db7af92d0fdb549c957c334a66a4fd608799'
COMPARATOR_SHA = '9dede8df91b9def8623c608e9581cb5246647c4870c8e52cdca4f6a8c76cb703'
PAIRS = {'A_A': ('A', 'AA'), 'A_B': ('A', 'B'), 'AA_B': ('AA', 'B')}
CONTRACT = {'capture': 'after each original role backward at native capture_rng',
            'bank_dtype': 'torch.float32', 'reduction_dtype': 'torch.float64',
            'pairs': PAIRS, 'None_preserved': True, 'optimizer_update': False,
            'original_gradient_rtol': 1e-4, 'original_gradient_atol': 1e-6,
            'formal_authorized': False}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+x)))


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')


def source_module(path, sha, name):
    require(digest(path) == sha, 'Pinned source hash mismatch: '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def upstream(root):
    module = source_module(Path(root)/'tools/structured_world/validate_p0_execution_numerics.py',
                           FORWARD_SHA, '_fullgrad_cold_p0')
    module.frozen_root(root, FORWARD_COMMIT)
    return module


def comparator(root):
    return source_module(Path(root)/'tools/structured_world/compare_execution_checkpoints.py',
                         COMPARATOR_SHA, '_fullgrad_native_comparator')


def checked_forward(p0, model_root, path, sha):
    require(sha == FORWARD_BINDING_SHA, 'Original 44c source binding must be preserved')
    shared = p0.shared_collector(model_root)
    return p0.check_binding(path, sha, model_root, shared), shared


def register(args):
    p0 = upstream(args.forward_source_root)
    original, _ = checked_forward(p0, args.model_source_root, args.forward_source_binding, args.forward_source_binding_sha256)
    commit = p0.frozen_root(ROOT)
    committed = subprocess.check_output(['git', 'show', commit+':'+RELATIVE], cwd=ROOT)
    require(hashlib.sha256(committed).hexdigest() == digest(__file__), 'New producer must be in frozen commit')
    require(args.chunk_elements > 0 and args.memory_reserve_bytes >= 0, 'Invalid CPU chunk/reserve')
    native = load(original['original_acceptance_bindings']['path'])
    # The prior native source/criteria/origins stay intact. New producer binding
    # authorizes only prospective supplementary observation, not retroactive V1.
    value = {'schema': 'execution_p0_fullgrad_source_binding_v1', 'state': 'frozen_before_probe',
             'frozen_UTC': datetime.now(timezone.utc).isoformat(), 'contract': CONTRACT,
             'registration_sha256': p0.REGISTRATION_SHA, 'candidate_mode': 'engineering_fast',
             'roles': p0.ROLES, 'preset': p0.PRESET, 'model': original['model'],
             'producer': {'root': str(ROOT), 'git_sha': commit, 'files_sha256': {RELATIVE: digest(__file__)}},
             'forward_producer': {'root': str(Path(args.forward_source_root).resolve()), 'git_sha': FORWARD_COMMIT,
                                  'tool_sha256': FORWARD_SHA},
             'forward_source_binding': {'path': str(args.forward_source_binding.resolve()), 'sha256': FORWARD_BINDING_SHA},
             'original_acceptance_bindings': original['original_acceptance_bindings'],
             'cases': native['cases'], 'chunk_elements': args.chunk_elements, 'memory_reserve_bytes': args.memory_reserve_bytes,
             'comparator_sha256': COMPARATOR_SHA, 'legacy_sha256': p0.LEGACY_SHA,
             'verification_relationship': 'Separate future fullgrad supplement; original native819 and forward44c bindings unchanged',
             'native_binding_or_prior_reports_modified': False, 'formal_authorized': False}
    write_new(args.output, value)
    return {'path': str(args.output.resolve()), 'sha256': digest(args.output), 'frozen_UTC': value['frozen_UTC']}


def check_binding(args, p0):
    require(digest(args.source_binding) == args.source_binding_sha256, 'New fullgrad binding SHA mismatch')
    b = load(args.source_binding)
    require(b['schema'] == 'execution_p0_fullgrad_source_binding_v1' and b['state'] == 'frozen_before_probe'
            and b['contract'] == json.loads(json.dumps(CONTRACT)) and b['preset'] == p0.PRESET
            and b['roles'] == p0.ROLES and b['registration_sha256'] == p0.REGISTRATION_SHA
            and b['candidate_mode'] == 'engineering_fast', 'Supplement contract/preset changed')
    frozen = datetime.fromisoformat(b['frozen_UTC'].replace('Z', '+00:00'))
    require(frozen.tzinfo is not None and frozen <= datetime.now(timezone.utc), 'Fresh binding must precede probe')
    require(Path(b['producer']['root']).resolve() == ROOT and Path(b['model']['root']).resolve() == args.model_source_root.resolve()
            and Path(b['forward_producer']['root']).resolve() == args.forward_source_root.resolve(), 'Explicit immutable roots differ')
    p0.frozen_root(ROOT, b['producer']['git_sha'])
    require(b['producer']['files_sha256'] == {RELATIVE: digest(__file__)} and b['comparator_sha256'] == COMPARATOR_SHA
            and b['legacy_sha256'] == p0.LEGACY_SHA, 'Producer/legacy/comparator changed')
    require(b['forward_producer']['git_sha'] == FORWARD_COMMIT and b['forward_producer']['tool_sha256'] == FORWARD_SHA,
            'Original 6d16 producer changed')
    ref = b['forward_source_binding']
    original, shared = checked_forward(p0, args.model_source_root, ref['path'], ref['sha256'])
    native_ref = original['original_acceptance_bindings']
    require(b['original_acceptance_bindings'] == native_ref and b['model'] == original['model']
            and b['cases'] == load(native_ref['path'])['cases'], 'Historical bindings/origins changed')
    require(type(b['chunk_elements']) is int and b['chunk_elements'] > 0 and
            type(b['memory_reserve_bytes']) is int and b['memory_reserve_bytes'] >= 0, 'Invalid chunk/reserve binding')
    return b, shared


def groups_for(run, case):
    path = Path(run)/'optimizer_groups.json'
    require(digest(path) == case['optimizer_groups_sha256'], 'Original optimizer groups changed')
    groups = load(path)
    require(isinstance(groups, list) and groups, 'Actual native optimizer groups required')
    names = [n for g in groups for n in g['names']]
    require(len(names) == len(set(names)) and all(type(n) is str for n in names), 'Duplicate/invalid optimizer group names')
    require(all(type(g['elements']) is int and g['elements'] > 0 for g in groups), 'Invalid native group elements')
    return groups


def host_available_bytes():
    values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(values['MemAvailable'].split()[0])*1024


def memory_estimate(elements, world=8, chunk_elements=1 << 20, reserve_bytes=4 << 30, largest_parameter_elements=0):
    # Three new banks plus original legacy A bank, across all ranks; FP64
    # comparison scratch is bounded and never a full FP64 model clone.
    return {'FP32_bank_bytes_per_rank': elements*4, 'new_three_banks_bytes_all_ranks': world*elements*12,
            'legacy_A_bank_bytes_all_ranks': world*elements*4, 'CPU_chunk_scratch_allowance_bytes_all_ranks': world*chunk_elements*64,
            'legacy_largest_tensor_scratch_allowance_bytes_all_ranks': world*largest_parameter_elements*40,
            'largest_parameter_elements': largest_parameter_elements,
            'reserve_bytes': reserve_bytes, 'minimum_host_available_bytes_before_A':
            world*(elements*16+chunk_elements*64+largest_parameter_elements*40)+reserve_bytes}


class FullGradientBanks:
    """Actual CPU snapshots of all named parameters, including None inventory."""
    def __init__(self, cmp, groups, chunk_elements=1 << 20, memory_check=None):
        names = [n for group in groups for n in group['names']]
        require(type(chunk_elements) is int and chunk_elements > 0, 'Positive CPU chunk required')
        require(len(names) == len(set(names)), 'Duplicate optimizer group parameter')
        self.cmp, self.groups, self.chunk = cmp, groups, chunk_elements
        self.memory_check, self.banks, self.inventories = memory_check, {}, {}
        self.statistics = None

    def capture(self, role, model):
        import torch
        require(len(self.inventories) < 3 and role == ('A', 'AA', 'B')[len(self.inventories)], 'Gradient roles must be captured once in order')
        parameters = dict(model.named_parameters())
        group_names = [n for g in self.groups for n in g['names']]
        require(set(group_names) == {n for n, p in parameters.items() if p.requires_grad}, 'Native optimizer/trainable inventory differs')
        for group in self.groups:
            require(sum(parameters[n].numel() for n in group['names']) == group['elements'], 'Optimizer group element count differs')
        if role == 'A' and self.memory_check:
            self.memory_check(sum(parameters[n].numel() for n in group_names), max(parameters[n].numel() for n in group_names))
        bank, inventory = {}, []
        for name, parameter in parameters.items():
            gradient = parameter.grad
            require(parameter.dtype == torch.float32, 'be5 FP32 parameter required: '+name)
            if gradient is not None:
                require(gradient.dtype == torch.float32 and gradient.layout == torch.strided
                        and gradient.shape == parameter.shape, 'Original dense FP32 gradient required: '+name)
                # dtype/device transfer only after detach; copy=True produces an
                # independent CPU bank without altering model grads or GPU math.
                bank[name] = gradient.detach().to(device='cpu', dtype=torch.float32, copy=True).contiguous()
            else:
                bank[name] = None
            inventory.append([name, list(parameter.shape), str(parameter.dtype), parameter.requires_grad,
                              None if gradient is None else [list(gradient.shape), str(gradient.dtype)]])
        if self.inventories:
            require(inventory == self.inventories['A'], 'Parameter/None-gradient inventory changed across roles')
        self.inventories[role], self.banks[role] = inventory, bank
        if role == 'B':
            self._compare(parameters, group_names)
            self.banks.clear()  # Only compact statistics survive capture.

    def _compare(self, parameters, group_names):
        comparison = self.cmp.NumericalComparison(self.chunk, has_origin=False)
        group_of = {n: i for i, group in enumerate(self.groups) for n in group['names']}
        # None is retained in inventory, never converted to an invented zero.
        for name in group_names:
            values = [self.banks[r][name] for r in ('A', 'AA', 'B')]
            if values[0] is not None:
                comparison.tensor('gradient', name, values, optimizer_group=group_of[name])
        tensors = {}
        for name, row in comparison.rows.get('gradient', {}).items():
            tensors[name] = {'component': name.split('.')[0], 'optimizer_group': str(row.optimizer_group),
                'pairs': {pair: {'elements': s.n, 'reference_square': s.reference2, 'candidate_square': s.candidate2,
                                'difference_square': s.error2, 'dot': s.dot, 'max_absolute': s.maximum,
                                'nonfinite_elements': s.nonfinite, 'bit_different_elements': s.bit_differences,
                                'fixed_numeric_violations': s.violations}
                          for pair, s in row.pairs.items()}}
        expected = [[n, list(parameters[n].shape), str(parameters[n].dtype)] for n in group_names]
        self.statistics = {'inventory': self.inventories, 'expected_trainable_inventory': expected,
            'gradient_inventory_sha256': metadata_hash(expected), 'trainable_parameter_count': len(group_names),
            'trainable_elements': sum(parameters[n].numel() for n in group_names),
            'all_trainable_gradients_present': all(self.banks['A'][n] is not None for n in group_names),
            'captured_roles': list(self.inventories), 'tensors': tensors, 'optimizer_groups': self.groups,
            'measurement': 'Three independent post-backward CPU FP32 banks; all three pairs directly chunk-reduced in CPU FP64'}

    def report(self):
        require(self.statistics is not None and not self.banks, 'Incomplete three-role full gradient capture')
        return self.statistics


def summarize(rows):
    raw = {k: math.fsum(s[k] for _, s in rows) for k in ('reference_square', 'candidate_square', 'difference_square', 'dot')}
    n, nonfinite = sum(s['elements'] for _, s in rows), sum(s['nonfinite_elements'] for _, s in rows)
    a, b, d = (math.sqrt(raw[k]) for k in ('reference_square', 'candidate_square', 'difference_square'))
    maxima = sorted(s['max_absolute'] for _, s in rows)
    def quantile(q):
        if not maxima: return None
        pos = (len(maxima)-1)*q; lo, hi = math.floor(pos), math.ceil(pos)
        return maxima[lo]+(maxima[hi]-maxima[lo])*(pos-lo)
    result = {'elements': n, 'nonfinite_elements': nonfinite, 'logical_tensor_count': len(rows),
        'reference_norm': a, 'candidate_norm': b, 'difference_norm': d, 'dot': raw['dot'],
        'relative_L2': d/a if a else 0. if not d else None,
        'cosine': raw['dot']/(a*b) if a and b else None, 'zero_reference_norm': a == 0.,
        'max_absolute': max(maxima, default=0.), 'bit_different_elements': sum(s['bit_different_elements'] for _, s in rows),
        'fixed_numeric_violations': sum(s['fixed_numeric_violations'] for _, s in rows),
        'logical_tensor_max_error_quantiles': {k: quantile(q) for k, q in (('p50', .5), ('p90', .9), ('p99', .99))},
        'worst_tensors': [{'name': name, 'max_absolute': s['max_absolute']} for name, s in
                          sorted(rows, key=lambda row: row[1]['max_absolute'], reverse=True)[:8]]}
    # Finite-subset norms from the mature reducer must never look like complete
    # norms when NaN/Inf is present. Keep counts and explicitly invalidate them.
    if nonfinite:
        for key in ('reference_norm', 'candidate_norm', 'difference_norm', 'dot', 'relative_L2', 'cosine', 'max_absolute'):
            result[key] = None
    result['bitwise_passed'] = nonfinite == 0 and result['bit_different_elements'] == 0
    result['fixed_numeric_passed'] = nonfinite == 0 and result['fixed_numeric_violations'] == 0
    return result


def combine_gradients(records, expected_hash, expected_ranks=8):
    require(len(records) == expected_ranks and {r['rank'] for r in records} == set(range(expected_ranks)), 'Missing/duplicate full gradient ranks')
    first = records[0]['fullgrad']
    for record in records:
        value = record['fullgrad']
        require(value['gradient_inventory_sha256'] == metadata_hash(value['expected_trainable_inventory']) == expected_hash and
                value['expected_trainable_inventory'] == first['expected_trainable_inventory'] and
                value['optimizer_groups'] == first['optimizer_groups'] and value['captured_roles'] == ['A', 'AA', 'B'],
                'Gradient/group/role inventory differs from prospective binding')
        require(value['trainable_elements'] == sum(math.prod(row[1]) for row in value['expected_trainable_inventory'])
                and value['trainable_parameter_count'] == len(value['expected_trainable_inventory']), 'Gradient inventory element/count mismatch')
        require(all(value['inventory'][r] == value['inventory']['A'] for r in ('AA', 'B')), 'Role None inventory differs')
        for tensor in value['tensors'].values():
            require(set(tensor['pairs']) == set(PAIRS), 'Actual AA_B full gradient pair missing')
    coverage = all(r['fullgrad']['all_trainable_gradients_present'] and
                   set(r['fullgrad']['tensors']) == {row[0] for row in first['expected_trainable_inventory']} and
                   sum(t['pairs']['A_A']['elements'] for t in r['fullgrad']['tensors'].values()) == first['trainable_elements']
                   for r in records)
    comparisons = {}
    for pair in PAIRS:
        rows, components, groups = [], {}, {}
        for record in records:
            for name, tensor in record['fullgrad']['tensors'].items():
                row = (f"rank{record['rank']}/{name}", tensor['pairs'][pair]); rows.append(row)
                components.setdefault(tensor['component'], []).append(row)
                groups.setdefault(tensor['optimizer_group'], []).append(row)
        comparisons[pair] = {'gradient': {'all': summarize(rows),
            'by_component': {n: summarize(v) for n, v in components.items()},
            'by_optimizer_group': {n: summarize(v) for n, v in groups.items()}}}
    return {'schema': 'execution_acceptance_gradient_stats_v1', 'registration_sha256': None,
            'gradient_inventory_sha256': expected_hash, 'full_parameter_rank_coverage': coverage,
            'full_parameter_gradient_statistics_captured': coverage, 'ranks': expected_ranks,
            'comparisons': comparisons, 'per_rank': records, 'AA_B_directly_measured': True,
            'passed': False, 'formal_authorized': False, 'trainer_consumable': False,
            'numeric_thresholds_unchanged': True, 'limits': ['Statistics only; actual forward/E1/native/dev/recovery and prospective receipt envelopes require separate review',
                'Original strict failure/exit preserved; no automatic grant', 'No persisted raw tensor or gradient banks']}


def run_observed(args, p0, shared, cmp, legacy, framework, binding, recheck, groups, case):
    old_class, old_combine = p0.collector_class, p0.combine
    holder = {}
    def memory_check(elements, largest):
        estimate = memory_estimate(elements, 8, binding['chunk_elements'], binding['memory_reserve_bytes'], largest)
        available = host_available_bytes()
        require(available >= estimate['minimum_host_available_bytes_before_A'], 'Insufficient actual host RAM for full gradients and original legacy bank')
        holder['memory'] = {**estimate, 'actual_MemAvailable_before_A': available}
    base = old_class(shared)
    class FullCollector(base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.full = FullGradientBanks(cmp, groups, binding['chunk_elements'], memory_check)
        def after_backward(self, model, rng):
            self.full.capture(self.role, model)
            return super().after_backward(model, rng)
        def report(self, rank, tokens):
            row = super().report(rank, tokens)
            row['_fullgrad'] = self.full.report(); row['_memory'] = holder['memory']
            return row
    def observed_combine(unused_shared, records, strict_ref, expected_ranks=8):
        compact = [{'rank': r['rank'], 'tokens': r['tokens'], 'fullgrad': r['_fullgrad'], 'memory': r['_memory']} for r in records]
        holder['gradient_report'] = combine_gradients(compact, binding['cases'][case]['gradient_inventory_sha256'], expected_ranks)
        # Leave the original forward report shape and strict path intact.
        return old_combine(shared, [{k: v for k, v in r.items() if k not in ('_fullgrad', '_memory')} for r in records], strict_ref, expected_ranks)
    p0.collector_class, p0.combine = lambda _: FullCollector, observed_combine
    try:
        return p0.run_observed(args, shared, legacy, framework, binding, recheck)
    finally:
        p0.collector_class, p0.combine = old_class, old_combine
        if 'gradient_report' in holder:
            strict_path, forward_path = args.output/'strict/COMPLETE.json', args.output/'P0_FORWARD_COMPARISON.json'
            require(strict_path.is_file() and forward_path.is_file(), 'Original strict/forward completion missing')
            report = holder['gradient_report']
            report.update(registration_sha256=p0.REGISTRATION_SHA, candidate_mode='engineering_fast',
                producer_sha256=digest(__file__), producer_source_binding={'path': str(args.source_binding.resolve()), 'sha256': args.source_binding_sha256},
                checkpoint={'run': str(args.run.resolve()), 'tag': args.tag, 'case': case},
                model_source=binding['model'], original_acceptance_bindings=binding['original_acceptance_bindings'],
                original_forward_source_binding=binding['forward_source_binding'], contract=CONTRACT,
                original_strict={'path': str(strict_path.resolve()), 'sha256': digest(strict_path), 'passed': load(strict_path)['passed'],
                                 'tool_sha256': p0.LEGACY_SHA, 'unchanged_original_json': True},
                forward_observation={'path': str(forward_path.resolve()), 'sha256': digest(forward_path)},
                verification_relationship=binding['verification_relationship'])
            write_new(args.output/'P0_FULL_GRADIENT_COMPARISON.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    reg = sub.add_parser('register')
    for name in ('model-source-root', 'forward-source-root', 'forward-source-binding', 'output'):
        reg.add_argument('--'+name, type=Path, required=True)
    reg.add_argument('--forward-source-binding-sha256', default=FORWARD_BINDING_SHA)
    reg.add_argument('--chunk-elements', type=int, default=1 << 20)
    reg.add_argument('--memory-reserve-bytes', type=int, default=4 << 30)
    probe = sub.add_parser('probe')
    for name in ('model-source-root', 'forward-source-root', 'source-binding', 'run', 'cache', 'dino-root', 'output'):
        probe.add_argument('--'+name, type=Path, required=True)
    probe.add_argument('--source-binding-sha256', required=True); probe.add_argument('--tag', required=True)
    probe.add_argument('--model-state-file', type=Path); probe.add_argument('--microbatch', type=int, choices=(4,), default=4)
    probe.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.command == 'register':
        print(json.dumps(register(args))); return
    p0 = upstream(args.forward_source_root)
    binding, shared = check_binding(args, p0)
    require(not args.output.exists(), 'NEW output required; no original report overwrite')
    case = p0.selected_origin(binding, args.run, args.tag)
    groups = groups_for(args.run, binding['cases'][case])
    if args.preflight:
        print(json.dumps({'preflight_passed': True, 'case': case, 'model_source_sha': p0.MODEL_SHA,
              'forward_source_sha': FORWARD_COMMIT, 'GPU_jobs_started': 0,
              'source_binding_sha256': args.source_binding_sha256,
              'memory_estimate': memory_estimate(sum(g['elements'] for g in groups), 8, binding['chunk_elements'], binding['memory_reserve_bytes']),
              'preflight_estimate_excludes_legacy_largest_tensor_scratch_until_actual_model_inventory': True,
              'actual_host_MemAvailable_bytes': host_available_bytes()})); return
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    model_root = args.model_source_root.resolve(); sys.path.insert(0, str(model_root))
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if path and (name == 'starVLA' or name.startswith('starVLA.') or name == 'tools' or name.startswith('tools.')):
            require(Path(path).resolve().is_relative_to(model_root), 'Foreign model/trainer module already imported: '+name)
    legacy = importlib.import_module('tools.structured_world.validate_engineering_execution')
    framework = importlib.import_module('starVLA.model.framework.vla_structured_fgtr')
    require(Path(legacy.__file__).resolve() == model_root/'tools/structured_world/validate_engineering_execution.py'
            and digest(legacy.__file__) == p0.LEGACY_SHA and
            Path(framework.__file__).resolve() == model_root/'starVLA/model/framework/vla_structured_fgtr.py', 'Actual imported be5 source differs')
    recheck = lambda: check_binding(args, p0)
    run_observed(args, p0, shared, comparator(model_root), legacy, framework, binding, recheck, groups, case)


if __name__ == '__main__':
    main()
