"""Compare native ZeRO-2 continuation checkpoints on CPU, without consolidation.

Example:
  CUDA_VISIBLE_DEVICES='' python -B tools/structured_world/compare_execution_checkpoints.py \
    --reference A/checkpoints/TAG --reference-replay AA/checkpoints/TAG \
    --candidate B/checkpoints/TAG --output comparison.json

Origin is resolved from identity.json engineering_continue_origin unless supplied
with --origin. Native bitwise failures and a separate fixed numerical gate are
both retained. A/A context never changes a threshold or grants acceptance.
"""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time

import numpy as np
import torch

ATOL, RTOL = 1e-6, 1e-4
PAIRS = {'A_A': (0, 1), 'A_B': (0, 2), 'AA_B': (1, 2)}
LABELS = ('A', 'AA', 'B')
CLIENT_FIELDS = ('completed', 'epoch', 'offset', 'exposure', 'scheduler_horizon',
                 'shared_geometry_identity', 'global_steps', 'global_samples',
                 'skipped_steps', 'dp_world_size', 'mp_world_size', 'lr_scheduler',
                 'data_sampler', 'random_ltd', 'ds_config', 'ds_version')
SCIENCE_FIELDS = ('config', 'cache_identity', 'current_and_GT_identity', 'DINO_identity',
                  'shared_geometry_identity', 'training_seed', 'world_size',
                  'effective_global_batch', 'FM_repeat', 'microbatch', 'dataset',
                  'scenes_per_epoch', 'updates_per_epoch', 'schedule_horizon',
                  'schedule_warmup', 'source_population_scenes', 'engineering_training_selection',
                  'precision', 'determinism', 'initialization_assets')
ORIGIN_SCIENCE_FIELDS = ('config', 'cache_identity', 'current_and_GT_identity', 'DINO_identity',
                         'shared_geometry_identity', 'training_seed', 'world_size',
                         'effective_global_batch', 'FM_repeat', 'microbatch', 'dataset',
                         'scenes_per_epoch', 'updates_per_epoch', 'schedule_horizon',
                         'schedule_warmup', 'precision', 'determinism')


class InvalidCheckpoint(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidCheckpoint(message)


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            sha.update(chunk)
    return sha.hexdigest()


def load_native(path):
    # mmap keeps tensor storages file-backed; chunked reductions bound scratch
    # memory. OS page residency is not a fixed RSS guarantee.
    return torch.load(path, map_location='cpu', mmap=True, weights_only=False)


def exact(left, right, path='', errors=None):
    errors = errors if errors is not None else []
    if torch.is_tensor(left) or torch.is_tensor(right):
        equal = (torch.is_tensor(left) and torch.is_tensor(right) and left.dtype == right.dtype
                 and left.shape == right.shape
                 and torch.equal(left.detach().contiguous().reshape(-1).view(torch.uint8),
                                 right.detach().contiguous().reshape(-1).view(torch.uint8)))
    elif isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        equal = (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
                 and left.dtype == right.dtype and left.shape == right.shape
                 and left.tobytes() == right.tobytes())
    elif isinstance(left, dict) and isinstance(right, dict):
        equal = set(left) == set(right)
        if equal:
            for key in left:
                exact(left[key], right[key], path+'/'+str(key), errors)
            return errors
    elif isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        equal = len(left) == len(right)
        if equal:
            for i, (a, b) in enumerate(zip(left, right)):
                exact(a, b, path+'/'+str(i), errors)
            return errors
    elif isinstance(left, float) and isinstance(right, float):
        equal = struct.pack('!d', left) == struct.pack('!d', right)
    else:
        equal = type(left) is type(right) and left == right
    if not equal:
        errors.append(path)
    return errors


def inventory(values):
    return [(key, list(value.shape), str(value.dtype)) for key, value in values.items()]


def fragment(value):
    if isinstance(value, dict):
        return int(value['start']), int(value['numel'])
    return int(value.start), int(value.numel)


class Bank:
    def __init__(self, folder, *, verify_hashes, progress=None):
        self.folder = Path(folder).resolve()
        self.complete = json.loads((self.folder/'COMPLETE.json').read_text())
        self.run = self.folder.parent.parent
        self.identity = json.loads((self.run/'identity.json').read_text())
        self.groups = json.loads((self.run/'optimizer_groups.json').read_text())
        self.ranks = self.complete['ranks']
        require(isinstance(self.ranks, int) and self.ranks > 0, 'Invalid rank count')
        expected = {'mp_rank_00_model_states.pt'}
        expected |= {f'rng_rank{r}.pt' for r in range(self.ranks)}
        expected |= {f'zero_pp_rank_{r}_mp_rank_00_optim_states.pt' for r in range(self.ranks)}
        hashes = self.complete.get('file_sha256', {})
        require(set(hashes) == expected, 'COMPLETE must bind every native model/optimizer/RNG file')
        require(self.identity['identity'] == self.complete['identity'], 'Run/COMPLETE identity mismatch')
        self.bytes_hashed = 0
        for name in sorted(expected):
            require(Path(name).name == name and (self.folder/name).is_file(), 'Missing native payload: '+name)
            require(isinstance(hashes[name], str) and len(hashes[name]) == 64, 'Malformed payload hash')
            if verify_hashes:
                if progress:
                    progress('hash', self.folder, name)
                require(digest(self.folder/name) == hashes[name], 'Native payload hash changed: '+name)
                self.bytes_hashed += (self.folder/name).stat().st_size
        self.integrity_verified = verify_hashes

    def model(self):
        model = load_native(self.folder/'mp_rank_00_model_states.pt')
        require(model['identity'] == self.identity['identity'], 'Model/client identity mismatch')
        require(model['completed'] == self.complete['completed'], 'Model/COMPLETE completed mismatch')
        require(model['dp_world_size'] == self.ranks, 'Model/native rank count mismatch')
        require(isinstance(model['module'], dict), 'Native module state required')
        return model

    def optimizer(self, rank):
        state = load_native(self.folder/f'zero_pp_rank_{rank}_mp_rank_00_optim_states.pt')['optimizer_state_dict']
        require(int(state['zero_stage']) == 2, 'Only the registered native ZeRO-2 format is supported')
        require(isinstance(state['base_optimizer_state'], dict), 'Native non-elastic Adam state required')
        return state

    def provenance(self):
        return {'path': str(self.folder), 'identity': self.identity['identity'],
                'COMPLETE_sha256': digest(self.folder/'COMPLETE.json'),
                'identity_sha256': digest(self.run/'identity.json'),
                'optimizer_groups_sha256': digest(self.run/'optimizer_groups.json'),
                'completed': self.complete['completed'], 'ranks': self.ranks,
                'execution_mode': self.identity.get('execution_mode'),
                'source': self.identity.get('training_source_sha'),
                'payload_hashes_verified': self.integrity_verified, 'bytes_hashed': self.bytes_hashed}


def model_layout(model):
    required = set(CLIENT_FIELDS) | {'param_shapes', 'frozen_param_shapes', 'shared_params', 'buffer_names'}
    require(required <= set(model), 'Missing native metadata fields: '+str(sorted(required-set(model))))
    for name, value in model['module'].items():
        require(torch.is_tensor(value), 'Nontensor native model value: '+name)
        if value.is_floating_point():
            require(value.dtype == torch.float32, 'Non-FP32 model state: '+name)
    shapes = model['param_shapes']
    require(isinstance(shapes, list) and shapes, 'Missing native trainable shape groups')
    names = [name for group in shapes for name in group]
    require(len(names) == len(set(names)), 'Duplicate trainable parameter name')
    for group in shapes:
        for name, shape in group.items():
            require(name in model['module'] and tuple(model['module'][name].shape) == tuple(shape),
                    'Trainable shape/model mismatch: '+name)
    return {'module_order_shape_dtype': inventory(model['module']),
            'param_shapes': [(list(group), [list(shape) for shape in group.values()]) for group in shapes],
            'frozen_param_shapes': [(name, list(shape)) for name, shape in model['frozen_param_shapes'].items()],
            'shared_params': model['shared_params'], 'buffer_names': model['buffer_names'],
            'frozen_fragment_order_shape_dtype': inventory(model.get('frozen_param_fragments', {}))}


def optimizer_layout(state, model):
    masters = state['single_partition_of_fp32_groups']
    mapping = state['param_slice_mappings']
    base = state['base_optimizer_state']
    require(len(masters) == len(mapping) == len(base['param_groups']) == len(model['param_shapes']),
            'Native optimizer group count changed')
    addresses, moment_inventory, scalar_state = [], [], {}
    for g, (master, slices, param_group) in enumerate(zip(masters, mapping, base['param_groups'])):
        require(master.dtype == torch.float32 and master.ndim == 1, 'Invalid FP32 master partition')
        require(len(param_group['params']) == 1, 'Expected one flattened parameter per native optimizer group')
        require(all(name in model['param_shapes'][g] for name in slices), 'Unknown optimizer slice name')
        entries = [(name, *fragment(address)) for name, address in slices.items()]
        covered = 0
        for name, first, length in sorted(entries, key=lambda row: row[1]):
            require(first == covered and length > 0, 'Overlapping/gapped native slice mapping')
            covered += length
        require(covered == master.numel(), 'Slice mapping must cover every saved lean master value')
        addresses.append(entries)
        pid = param_group['params'][0]
        adam = base['state'].get(pid, {})
        moment_inventory.append([])
        scalar_state[g] = {}
        for key, value in adam.items():
            if key == 'step' or not torch.is_tensor(value):
                scalar_state[g][key] = value
            else:
                require(value.dtype == torch.float32 and value.ndim == 1,
                        'Unsupported/non-FP32 native Adam tensor: '+str(key))
                # Native non-elastic base state may retain end alignment padding;
                # master partitions and hp slice mappings exclude that padding.
                require(value.numel() in (master.numel(), master.numel()+state['group_paddings'][g]),
                        'Adam tensor length inconsistent with saved alignment padding')
                moment_inventory[g].append((key, list(value.shape), str(value.dtype)))
    require(set(base['state']) <= {p for g in base['param_groups'] for p in g['params']},
            'Optimizer state contains an unregistered flattened parameter')
    return {'masters': [(list(t.shape), str(t.dtype)) for t in masters], 'slice_mappings': addresses,
            'moment_inventory': moment_inventory, 'scalar_state': scalar_state,
            'param_groups': base['param_groups'], 'zero_stage': int(state['zero_stage']),
            'group_paddings': state['group_paddings'], 'partition_count': state['partition_count'],
            'dynamic_loss_scale': state.get('dynamic_loss_scale'), 'overflow': state.get('overflow'),
            'clip_grad': state.get('clip_grad'), 'ds_version': state['ds_version'],
            'loss_scaler_state': vars(state['loss_scaler']) if state.get('loss_scaler') is not None else None}


class PairStatistics:
    def __init__(self):
        self.n = self.bit_differences = self.violations = self.nonfinite = 0
        self.maximum = self.error2 = self.reference2 = self.candidate2 = self.dot = 0.

    def add(self, reference, candidate):
        self.n += reference.numel()
        self.bit_differences += int((reference.contiguous().view(torch.int32)
                                    != candidate.contiguous().view(torch.int32)).sum())
        finite = torch.isfinite(reference) & torch.isfinite(candidate)
        self.nonfinite += int((~finite).sum())
        a, b = reference[finite].double(), candidate[finite].double()
        error = b-a
        self.violations += int((error.abs() > ATOL+RTOL*a.abs()).sum())
        self.violations += int((~finite).sum())
        if error.numel():
            self.maximum = max(self.maximum, float(error.abs().max()))
            self.error2 += float(error.square().sum())
            self.reference2 += float(a.square().sum())
            self.candidate2 += float(b.square().sum())
            self.dot += float((a*b).sum())

    def merge(self, other):
        for key in ('n', 'bit_differences', 'violations', 'nonfinite', 'error2', 'reference2', 'candidate2', 'dot'):
            setattr(self, key, getattr(self, key)+getattr(other, key))
        self.maximum = max(self.maximum, other.maximum)

    def report(self):
        return {'elements': self.n, 'max_absolute': self.maximum,
                'reference_norm': math.sqrt(self.reference2), 'candidate_norm': math.sqrt(self.candidate2),
                'difference_norm': math.sqrt(self.error2),
                'relative_L2': math.sqrt(self.error2/self.reference2) if self.reference2 else (0. if not self.error2 else None),
                'cosine': self.dot/math.sqrt(self.reference2*self.candidate2) if self.reference2 and self.candidate2 else None,
                'zero_reference_norm': self.reference2 == 0, 'nonfinite_elements': self.nonfinite,
                'bit_different_elements': self.bit_differences, 'bitwise_passed': self.bit_differences == 0 and self.nonfinite == 0,
                'fixed_numeric_violations': self.violations, 'fixed_numeric_passed': self.violations == 0}


class TensorStatistics:
    def __init__(self, name, optimizer_group):
        self.name, self.optimizer_group = name, optimizer_group
        self.pairs = {key: PairStatistics() for key in PAIRS}
        self.updates2 = [0., 0., 0.]
        self.update_dot = self.origin2 = 0.
        self.origin_missing_moments = False

    def add(self, values, origin):
        for pair, (i, j) in PAIRS.items():
            self.pairs[pair].add(values[i], values[j])
        if origin is not None:
            require(bool(torch.isfinite(origin).all()), 'Nonfinite origin tensor: '+self.name)
            finite = torch.isfinite(origin)
            for value in values:
                finite &= torch.isfinite(value)
            o = origin[finite].double()
            updates = [value[finite].double()-o for value in values]
            self.origin2 += float(o.square().sum())
            for i, update in enumerate(updates):
                self.updates2[i] += float(update.square().sum())
            self.update_dot += float((updates[0]*updates[2]).sum())


def quantiles(values):
    if not values:
        return None
    return dict(zip(('p50', 'p90', 'p99'), np.quantile(values, [.5, .9, .99]).tolist()))


def summarize(rows, pair, *, has_origin):
    combined = PairStatistics()
    for row in rows:
        combined.merge(row.pairs[pair])
    result = combined.report()
    result['logical_tensor_count'] = len(rows)
    result['logical_tensor_max_error_quantiles'] = quantiles([row.pairs[pair].maximum for row in rows])
    result['worst_tensors'] = [{'name': row.name, 'max_absolute': row.pairs[pair].maximum}
                              for row in sorted(rows, key=lambda row: row.pairs[pair].maximum, reverse=True)[:8]]
    if has_origin:
        update2 = [sum(row.updates2[i] for row in rows) for i in range(3)]
        origin2 = sum(row.origin2 for row in rows)
        error2 = sum(row.pairs['A_B'].error2 for row in rows)
        result['origin_relative_updates'] = {
            'origin_norm': math.sqrt(origin2),
            'update_norms': dict(zip(LABELS, [math.sqrt(v) for v in update2])),
            'update_to_origin_norms': dict(zip(LABELS, [math.sqrt(v/origin2) if origin2 else None for v in update2])),
            'candidate_minus_reference_over_reference_update_norm': math.sqrt(error2/update2[0]) if update2[0] else (0. if not error2 else None),
            'candidate_to_reference_update_norm_ratio': math.sqrt(update2[2]/update2[0]) if update2[0] else None,
            'candidate_reference_update_cosine': sum(row.update_dot for row in rows)/math.sqrt(update2[0]*update2[2]) if update2[0] and update2[2] else None,
            'zero_reference_update_norm': update2[0] == 0,
            'missing_origin_moments_initialized_as_zero': any(row.origin_missing_moments for row in rows)}
    return result


class NumericalComparison:
    def __init__(self, chunk_elements, has_origin):
        self.chunk, self.has_origin = chunk_elements, has_origin
        self.rows = defaultdict(dict)

    def tensor(self, category, name, values, origin=None, optimizer_group=None, missing_origin_moment=False):
        require(all(v.dtype == torch.float32 and v.shape == values[0].shape for v in values),
                'Numerical tensor dtype/shape mismatch: '+name)
        if origin is not None:
            require(origin.dtype == torch.float32 and origin.shape == values[0].shape, 'Origin tensor mismatch: '+name)
        row = self.rows[category].setdefault(name, TensorStatistics(name, optimizer_group))
        row.origin_missing_moments |= missing_origin_moment
        flat = [value.detach().reshape(-1) for value in values]
        old = origin.detach().reshape(-1) if origin is not None else None
        for first in range(0, flat[0].numel(), self.chunk):
            parts = [value[first:first+self.chunk] for value in flat]
            original = old[first:first+self.chunk] if old is not None else (torch.zeros_like(parts[0]) if missing_origin_moment else None)
            row.add(parts, original)

    def report(self):
        result = {pair: {} for pair in PAIRS}
        for category, named in self.rows.items():
            rows = list(named.values())
            components, groups = defaultdict(list), defaultdict(list)
            for row in rows:
                components[row.name.split('.')[0]].append(row)
                if row.optimizer_group is not None:
                    groups[str(row.optimizer_group)].append(row)
            for pair in PAIRS:
                result[pair][category] = {'all': summarize(rows, pair, has_origin=self.has_origin),
                    'by_component': {k: summarize(v, pair, has_origin=self.has_origin) for k, v in components.items()},
                    'by_optimizer_group': {k: summarize(v, pair, has_origin=self.has_origin) for k, v in groups.items()}}
        return result


def origin_path(banks, explicit):
    ancestry = [bank.identity.get('engineering_continue_origin') for bank in banks]
    require(all(isinstance(value, dict) for value in ancestry), 'All continuation runs must bind their native origin')
    require(not exact(ancestry[0], ancestry[1]) and not exact(ancestry[0], ancestry[2]), 'Continuation origins differ')
    value = ancestry[0]
    require(Path(value['tag']).name == value['tag'], 'Unsafe origin tag')
    path = Path(explicit).resolve() if explicit else Path(value['run'])/'checkpoints'/value['tag']
    require(digest(path/'COMPLETE.json') == value['complete_sha256'], 'Origin COMPLETE binding changed')
    return path, value


def compare_checkpoints(reference, reference_replay, candidate, *, origin=None,
                        verify_hashes=True, chunk_elements=1 << 18, progress=None):
    started = time.monotonic()
    report = {'schema': 'native_execution_checkpoint_comparison_v1', 'device': 'cpu',
              'fixed_numeric_gate': {'atol': ATOL, 'rtol': RTOL, 'configurable': False},
              'original_strict_basis': 'native recovery boundary: E1 exact and all FP32 values bitwise',
              'quantile_scope': 'exact empirical quantiles of logical parameter/buffer max error, not approximate element quantiles',
              'comparison_performed': False, 'valid_native_inputs': False}
    try:
        require(chunk_elements > 0, 'Positive chunk size required')
        banks = [Bank(p, verify_hashes=verify_hashes, progress=progress) for p in (reference, reference_replay, candidate)]
        old_path, ancestry = origin_path(banks, origin)
        old = Bank(old_path, verify_hashes=verify_hashes, progress=progress)
        require(old.identity['identity'] == ancestry['identity'] and old.complete['completed'] == ancestry['completed'],
                'Origin ancestry identity/completed mismatch')
        require(all(bank.ranks == old.ranks for bank in banks), 'Native world size changed')
        report['provenance'] = dict(zip((*LABELS, 'origin'), [b.provenance() for b in (*banks, old)]))
        e1 = {pair: [] for pair in PAIRS}
        report['E1_failure_paths'] = e1
        def check(values, path):
            for pair, (i, j) in PAIRS.items():
                exact(values[i], values[j], path, e1[pair])
        check([bank.groups for bank in banks], 'optimizer_groups.json')
        for bank in (*banks, old):
            require(set(ORIGIN_SCIENCE_FIELDS) <= set(bank.identity), 'Incomplete native scientific contract')
            require(not exact({key: bank.identity[key] for key in ORIGIN_SCIENCE_FIELDS},
                              {key: old.identity[key] for key in ORIGIN_SCIENCE_FIELDS}),
                    'Continuation scientific/data/schedule contract changed from origin')
            require(bank.complete['completed'] >= old.complete['completed'], 'Continuation precedes its native origin')
        check([{key: bank.identity.get(key) for key in SCIENCE_FIELDS} for bank in banks], 'scientific_contract')
        models = [bank.model() for bank in banks]
        original = old.model()
        layouts = [model_layout(model) for model in models]
        origin_layout = model_layout(original)
        check(layouts, 'model_inventory_and_group_order')
        require(all(not exact(layouts[0], layout) for layout in layouts[1:]), 'Incompatible model inventory')
        require(not exact(layouts[0], origin_layout), 'Origin model inventory changed')
        require(not exact(banks[0].groups, old.groups), 'Origin optimizer group registration changed')
        require(len(banks[0].groups) == len(models[0]['param_shapes']), 'Optimizer group registration count changed')
        for groups, shapes in zip(banks[0].groups, models[0]['param_shapes']):
            require(groups['names'] == list(shapes) and groups['elements'] == sum(math.prod(shape) for shape in shapes.values()),
                    'Optimizer group registration differs from native parameter order/elements')
        check([{key: model[key] for key in CLIENT_FIELDS} for model in models], 'client_step_scheduler_data')
        check([bank.complete['completed'] for bank in banks], 'COMPLETE/completed')
        report['client_counters'] = dict(zip(LABELS, [{key: model[key] for key in CLIENT_FIELDS if not isinstance(model[key], (dict, list, torch.Tensor))} for model in models]))
        comparison = NumericalComparison(chunk_elements, True)
        group_by_name = {name: g for g, shapes in enumerate(models[0]['param_shapes']) for name in shapes}
        buffer_names = set(models[0]['buffer_names'])
        aliases = set(models[0]['shared_params'])
        for name, tensor in models[0]['module'].items():
            values = [model['module'][name] for model in models]
            if not tensor.is_floating_point():
                check(values, 'discrete_model_buffer/'+name)
                continue
            category = ('BN_float' if name.endswith(('running_mean', 'running_var')) else
                        'other_float_buffer' if name in buffer_names else
                        'model_alias' if name in aliases else 'model_FP32')
            comparison.tensor(category, name, values, original['module'][name], group_by_name.get(name))
        for name in models[0].get('frozen_param_fragments', {}):
            comparison.tensor('frozen_fragment', name, [m['frozen_param_fragments'][name] for m in models],
                              original['frozen_param_fragments'][name])
        del models, original
        gc.collect()
        coverage = defaultdict(int)
        report['rank_checks'] = []
        for rank in range(banks[0].ranks):
            if progress:
                progress('optimizer_compare', banks[0].folder, str(rank))
            rng = [load_native(bank.folder/f'rng_rank{rank}.pt') for bank in banks]
            require(all({'python', 'numpy', 'torch', 'cuda', 'generators'} <= set(v) for v in rng), 'Incomplete native RNG state')
            check(rng, f'RNG/rank{rank}')
            states = [bank.optimizer(rank) for bank in banks]
            initial = old.optimizer(rank)
            # Model shape metadata is small; avoid retaining full mapped model
            # storages while reading the optimizer partitions.
            shape_model = {'param_shapes': [{name: torch.Size(shape) for name, shape in zip(names, shapes)}
                                            for names, shapes in layouts[0]['param_shapes']]}
            rank_layouts = [optimizer_layout(state, shape_model) for state in states]
            start_layout = optimizer_layout(initial, shape_model)
            check(rank_layouts, f'optimizer/rank{rank}/E1')
            for key in ('masters', 'slice_mappings', 'group_paddings', 'partition_count', 'zero_stage'):
                require(all(not exact(rank_layouts[0][key], layout[key]) for layout in rank_layouts[1:]), 'Incompatible native optimizer layout: '+key)
                require(not exact(rank_layouts[0][key], start_layout[key]), 'Origin partition routing changed: '+key)
            report['rank_checks'].append({'rank': rank, 'groups': len(rank_layouts[0]['masters']),
                                         'padding_excluded_elements': sum(rank_layouts[0]['group_paddings'])})
            for g, mapping in enumerate(states[0]['param_slice_mappings']):
                pids = [state['base_optimizer_state']['param_groups'][g]['params'][0] for state in (*states, initial)]
                adam = [state['base_optimizer_state']['state'].get(pid, {}) for state, pid in zip((*states, initial), pids)]
                moments = [key for key, value in adam[0].items() if key != 'step' and torch.is_tensor(value)]
                require(all(set(moments) == {key for key, value in a.items() if key != 'step' and torch.is_tensor(value)} for a in adam[1:3]),
                        'Incompatible Adam tensor inventory')
                for name, address in mapping.items():
                    first, length = fragment(address)
                    cut = slice(first, first+length)
                    coverage[name] += length
                    comparison.tensor('master_FP32', name, [state['single_partition_of_fp32_groups'][g][cut] for state in states],
                                      initial['single_partition_of_fp32_groups'][g][cut], g)
                    for key in moments:
                        value = adam[3].get(key)
                        comparison.tensor('Adam/'+key, name, [a[key][cut] for a in adam[:3]],
                                          value[cut] if value is not None else None, g,
                                          missing_origin_moment=value is None)
            del states, initial, rng, adam
            gc.collect()
        expected = {name: math.prod(shape) for names, shapes in layouts[0]['param_shapes'] for name, shape in zip(names, shapes)}
        require(dict(coverage) == expected, 'All-rank master slices do not cover every logical trainable parameter exactly once')
        report['coverage'] = {'trainable_parameters': len(expected), 'actual_parameter_elements': sum(expected.values()),
                              'all_rank_slice_membership_complete': True,
                              'alignment_padding': 'Excluded using native hp slice mappings; no parameter dropped'}
        report['comparisons'] = comparison.report()
        report['E1'] = {pair: {'passed': not errors, 'failure_count': len(errors), 'failure_paths': errors}
                        for pair, errors in e1.items()}
        report['original_strict'] = {}
        report['fixed_numeric_results'] = {}
        for pair in PAIRS:
            stats = [category['all'] for category in report['comparisons'][pair].values()]
            bitwise = all(stat['bitwise_passed'] for stat in stats)
            numeric = all(stat['fixed_numeric_passed'] for stat in stats)
            report['original_strict'][pair] = {'E1_exact': not e1[pair], 'FP32_bitwise': bitwise, 'passed': not e1[pair] and bitwise}
            report['fixed_numeric_results'][pair] = {'E1_exact': not e1[pair], 'passed': not e1[pair] and numeric,
                'violating_elements': sum(stat['fixed_numeric_violations'] for stat in stats)}
        report['A_A_context'] = {'not_an_acceptance_rule': True, 'thresholds_changed': False,
            'A_A_strict_passed': report['original_strict']['A_A']['passed'],
            'A_A_fixed_numeric_passed': report['fixed_numeric_results']['A_A']['passed'],
            'A_B_strict_failure_preserved': not report['original_strict']['A_B']['passed'],
            'note': 'Use all A_A/A_B/AA_B group norms and update-relative errors for context; no strict failure is relabeled as a pass.'}
        report['comparison_performed'] = report['valid_native_inputs'] = True
        report['passed'] = all(value['passed'] for value in report['original_strict'].values())
        report['integrity_verified'] = verify_hashes
        report['qualification_passed'] = report['passed'] and verify_hashes
    except (InvalidCheckpoint, KeyError, ValueError, TypeError, OSError, RuntimeError) as error:
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
        report['passed'] = report['qualification_passed'] = False
    report['wall_seconds'] = time.monotonic()-started
    report['resource_scope'] = 'CPU only; mmap file-backed storages; bounded numeric scratch, no consolidated FP32 export or GPU use'
    return report


def metadata_only(paths):
    from torch._subclasses.fake_tensor import FakeTensorMode
    results = []
    for path in paths:
        bank = Bank(path, verify_hashes=False)
        with FakeTensorMode():
            model = bank.model()
            layout = model_layout(model)
            rank = bank.optimizer(0)
            opt = optimizer_layout(rank, model)
        results.append({'provenance': bank.provenance(), 'module_entries': len(layout['module_order_shape_dtype']),
                        'native_trainable_groups': len(layout['param_shapes']), 'buffer_count': len(layout['buffer_names']),
                        'model_keys_sample': layout['module_order_shape_dtype'][:4],
                        'optimizer_rank0_master_shapes': opt['masters'],
                        'optimizer_rank0_moment_inventory': opt['moment_inventory'],
                        'client': {k: model[k] for k in ('completed', 'epoch', 'offset', 'exposure', 'global_steps', 'scheduler_horizon')}})
    return {'schema': 'native_execution_checkpoint_metadata_v1', 'metadata_only': True,
            'comparison_performed': False, 'qualification_passed': False, 'payload_hashes_verified': False,
            'tensor_payloads': 'FakeTensorMode; shapes/metadata only, no large payload reductions', 'banks': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--reference-replay', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--origin', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--metadata-only', action='store_true')
    parser.add_argument('--skip-payload-hashes', action='store_true', help='Explicit audit shortcut; result cannot claim integrity-qualified acceptance')
    parser.add_argument('--chunk-elements', type=int, default=1 << 18)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    require(args.threads > 0, 'Positive CPU thread count required')
    torch.set_num_threads(args.threads)
    def progress(stage, path, detail):
        print(json.dumps({'stage': stage, 'checkpoint': str(path), 'item': detail}), file=sys.stderr, flush=True)
    if args.metadata_only:
        report = metadata_only([args.reference, args.reference_replay, args.candidate])
        exit_code = 0
    else:
        report = compare_checkpoints(args.reference, args.reference_replay, args.candidate,
            origin=args.origin, verify_hashes=not args.skip_payload_hashes, chunk_elements=args.chunk_elements,
            progress=progress)
        exit_code = 0 if report['qualification_passed'] else (2 if report['comparison_performed'] else 1)
    report['tool_sha256'] = digest(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name+'.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    temporary.replace(args.output)
    print(json.dumps({'report': str(args.output), 'comparison_performed': report['comparison_performed'],
                      'qualification_passed': report['qualification_passed'], 'exit_code': exit_code}))
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
