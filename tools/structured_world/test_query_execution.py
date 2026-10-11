"""CPU value/gradient/RNG checks for opt-in query execution reuse.

No worker dispatch, GPU probing, asset mutation or training occurs here.
Run with --output NEW_JSON to retain a reproducible per-case evidence record.
"""
import argparse
import copy
import hashlib
import json
import os
import pickle
from pathlib import Path
import sys
import subprocess
import time
import types
import tempfile
import unittest
from unittest.mock import patch

if __name__ == '__main__': os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from starVLA.model.modules.structured_world.grid import GridSpec
from starVLA.model.modules.structured_world.queries import (
    QueryExecutionCache, ForwardFieldFiniteCache, _cached_body_template, body_points, body_relations,
    bind_query_label_identities, boundary_candidates, prepare_query_candidates, read_time_field, sample_queries,
)
from starVLA.model.modules.structured_world.losses import compute_structured_losses
from starVLA.model.modules.structured_world.semantics import StructuredSemantics


GRID = GridSpec(-6., 6., -5., 5., .5, .5)
STATS = dict(current_positive_weight=2., future_positive_weight=3.,
             arrival_positive_weight=4., release_positive_weight=2.)
BODY = dict(length=2., width=1., rear_axle_to_center=.3)
REFERENCE_COMMIT = '267701f3b2c6cf36a97ff3a5e6bfb535e5aab232'
FORMAL_REFERENCE_COMMIT = '0c54f1c49e7ed2a76b22826b654d6c72625f5b6d'


def reference_modules(commit=REFERENCE_COMMIT):
    """Read the task's original reference directly from git without checkout."""
    name = '_registered_query_reference_'+commit[:8]
    package = types.ModuleType(name)
    package.__path__ = []
    sys.modules[name] = package
    modules, hashes = {}, {}
    for stem in ('grid', 'queries', 'losses'):
        path = 'starVLA/model/modules/structured_world/'+stem+'.py'
        source = subprocess.run(['git', 'show', commit+':'+path], cwd=ROOT,
                                check=True, capture_output=True).stdout
        module = types.ModuleType(name+'.'+stem)
        sys.modules[module.__name__] = module
        exec(compile(source, commit+':'+path, 'exec'), module.__dict__)
        modules[stem], hashes[path] = module, hashlib.sha256(source).hexdigest()
    return modules, hashes


def labels(steps=8, batch=3):
    generator = torch.Generator().manual_seed(93)
    shape = (batch, steps+1, GRID.height, GRID.width)
    valid = torch.rand(shape, generator=generator) > .2
    valid[-1] = False
    occupancy = torch.randint(2, shape, generator=generator).byte()
    occupancy[~valid] = 255
    return dict(road_distance=torch.randn(batch, GRID.height, GRID.width, generator=generator),
                road_valid=valid[:, 0].clone(), occupancy=occupancy, occupancy_valid=valid,
                label_sha256=[hashlib.sha256(str(row).encode()).hexdigest() for row in range(batch)],
                label_schema=['structured_scene_cache_v1']*batch)


def close(a, b, *, exact=False):
    torch.testing.assert_close(a, b, rtol=0 if exact else 2e-6,
                               atol=0 if exact else 2e-7, equal_nan=True)


class QueryExecutionTests(unittest.TestCase):
    def test_sampling_RNG_values_records_and_detachment(self):
        for steps in (6, 8):
            for mode in ('uniform', 'local'):
                with self.subTest(steps=steps, mode=mode):
                    target = labels(steps)
                    proposal = torch.zeros(3, steps, 3, requires_grad=True)
                    cache = QueryExecutionCache()
                    for repeat in range(2):  # Includes an actual index cache hit.
                        snapshots = []
                        for fast in (False, True):
                            generator = torch.Generator().manual_seed(123+repeat)
                            global_rng = torch.get_rng_state().clone()
                            query = sample_queries(proposal, target, mode=mode, generator=generator,
                                count=32, grid=GRID, execution_fast=fast, execution_cache=cache)
                            self.assertTrue(torch.equal(global_rng, torch.get_rng_state()))
                            self.assertFalse(query['poses'].requires_grad)
                            snapshots.append((query, generator.get_state()))
                        close(snapshots[0][0]['poses'], snapshots[1][0]['poses'], exact=True)
                        close(snapshots[0][0]['time_index'], snapshots[1][0]['time_index'], exact=True)
                        self.assertEqual(snapshots[0][0]['strata_records'], snapshots[1][0]['strata_records'])
                        self.assertTrue(torch.equal(snapshots[0][1], snapshots[1][1]))
                    self.assertEqual(cache.hits, 3)

    def test_invalid_scene_consumes_no_RNG(self):
        target = labels(batch=1)
        target['occupancy_valid'].zero_()
        for fast in (False, True):
            generator = torch.Generator().manual_seed(37)
            before = generator.get_state().clone()
            query = sample_queries(torch.zeros(1, 8, 3), target, mode='local', generator=generator,
                                   count=16, grid=GRID, execution_fast=fast)
            self.assertTrue(torch.equal(before, generator.get_state()))
            self.assertEqual(query['strata_records'], [dict(global_valid_cells=0, fallback=16, invalid_scene=True)])

    def test_fast_strata_diagnostics_are_detached_int64_scalars(self):
        target = labels()
        for mode in ('uniform', 'local'):
            query = sample_queries(torch.zeros(3, 8, 3, requires_grad=True), target, mode=mode,
                                   generator=torch.Generator().manual_seed(42), count=16, grid=GRID, execution_fast=True)
            for record in query['strata_records']:
                for key in ('fallback', 'neighbor_time_queries'):
                    if key not in record: continue
                    value = record[key]
                    self.assertIsInstance(value, torch.Tensor)
                    self.assertEqual(value.shape, ())
                    self.assertEqual(value.dtype, torch.int64)
                    self.assertFalse(value.requires_grad)

    def test_empty_boundary_and_outside_proposal_RNG(self):
        target = labels(batch=1)
        target['occupancy_valid'].fill_(True)
        target['occupancy'].zero_()
        target['road_distance'].fill_(10.)
        proposal = torch.full((1, 8, 3), 1000.)
        snapshots = []
        for fast in (False, True):
            generator = torch.Generator().manual_seed(42)
            snapshots.append((sample_queries(proposal, target, mode='local', generator=generator,
                count=16, grid=GRID, execution_fast=fast), generator.get_state()))
        self.assertEqual(snapshots[0][0]['strata_records'][0]['fallback'], 12)
        close(snapshots[0][0]['poses'], snapshots[1][0]['poses'], exact=True)
        self.assertEqual(snapshots[0][0]['strata_records'], snapshots[1][0]['strata_records'])
        self.assertTrue(torch.equal(snapshots[0][1], snapshots[1][1]))

    def test_candidate_order_hash_schema_grid_and_mutation_invalidation(self):
        target, cache = labels(batch=1), QueryExecutionCache()
        # Give this scene labels: labels() deliberately empties its last row.
        target['occupancy_valid'].fill_(True)
        target['occupancy'].zero_()
        def read(): return cache.candidates(target, 0, GRID, local=True)
        candidates, boundary = read()
        close(candidates, torch.nonzero(target['occupancy_valid'][0, 1:]), exact=True)
        close(boundary, torch.nonzero(boundary_candidates(target['road_distance'][0],
            target['occupancy'][0], target['occupancy_valid'][0])), exact=True)
        again, _ = read()
        self.assertIs(candidates, again)
        target['occupancy_valid'][0, 1, 0, 0] = False
        changed, _ = read()
        self.assertIsNot(changed, again)
        self.assertEqual(len(changed), len(again)-1)
        for name, value in [('label_sha256', ['f'*64]), ('label_schema', ['v2'])]:
            previous, _ = read()
            target[name] = value
            self.assertIsNot(previous, read()[0])
        previous, _ = read()
        other_grid = GridSpec(-5., 7., -5., 5., .5, .5)
        self.assertIsNot(previous, cache.candidates(target, 0, other_grid, local=True)[0])
        previous, _ = read()
        target['road_distance'].fill_(100.)
        _, boundary = read()
        close(boundary, torch.nonzero(boundary_candidates(target['road_distance'][0],
            target['occupancy'][0], target['occupancy_valid'][0])), exact=True)
        target['occupancy'] = target['occupancy'].clone()
        self.assertIsNot(previous, read()[0])

    def test_cache_is_bounded_and_missing_identity_does_not_cache(self):
        cache, target = QueryExecutionCache(max_samples=1), labels()
        for row in range(3): cache.candidates(target, row, GRID, local=True)
        self.assertEqual(len(cache._samples), 1)
        cache.clear()
        self.assertEqual(len(cache._samples), 0)
        target.pop('label_sha256')
        for _ in range(2): cache.candidates(target, 0, GRID, local=True)
        self.assertEqual(cache.hits, 0)
        self.assertEqual(len(cache._samples), 0)
        target['label_sha256'] = ['forged']*3
        with self.assertRaises(ValueError): cache.candidates(target, 0, GRID, local=True)

    def test_CPU_preparation_pickle_transfer_slice_and_all_RNG(self):
        target = labels()
        before = torch.get_rng_state().clone()
        prepared = pickle.loads(pickle.dumps(prepare_query_candidates(target, grid=GRID)))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        subset = {key: value[1:3] for key, value in target.items()}
        subset['query_candidates'] = [item.to('cpu', non_blocking=True) for item in prepared[1:3]]
        for mode in ('uniform', 'local'):
            snapshots = []
            for fast in (False, True):
                generator = torch.Generator().manual_seed(428)
                if fast:
                    with patch('starVLA.model.modules.structured_world.queries.boundary_candidates', side_effect=AssertionError('morphology repeated')), \
                         patch('torch.nonzero', side_effect=AssertionError('indices recomputed')):
                        query = sample_queries(torch.zeros(2, 8, 3), subset, mode=mode, generator=generator,
                                               count=16, grid=GRID, execution_fast=True)
                else:
                    query = sample_queries(torch.zeros(2, 8, 3), subset, mode=mode, generator=generator, count=16, grid=GRID)
                snapshots.append((query, generator.get_state()))
            for key in ('poses', 'time_index'): close(snapshots[0][0][key], snapshots[1][0][key], exact=True)
            self.assertEqual(snapshots[0][0]['strata_records'], snapshots[1][0]['strata_records'])
            self.assertTrue(torch.equal(snapshots[0][1], snapshots[1][1]))
        pinned = []
        def pin_stub(value):
            pinned.append(value)
            return value
        with patch.object(torch.Tensor, 'pin_memory', new=pin_stub):
            prepared[0].pin_memory()
            self.assertEqual(len(pinned), 2)

    def test_prepared_indices_reject_wrong_hash_schema_grid_shape_and_rows(self):
        target = labels()
        target['query_candidates'] = prepare_query_candidates(target, grid=GRID)
        def sample(t, grid=GRID):
            return sample_queries(torch.zeros(3, 8, 3), t, mode='local', generator=torch.Generator().manual_seed(42),
                                  count=16, grid=grid, execution_fast=True)
        for key, changed in [('label_sha256', ['e'*64]*3), ('label_schema', ['wrong']*3),
                             ('query_candidates', target['query_candidates'][::-1]),
                             ('occupancy', target['occupancy'][:, :-1])]:
            with self.subTest(key=key):
                with self.assertRaises(ValueError): sample(dict(target, **{key: changed}))
        with self.assertRaises(ValueError): sample(target, GridSpec(-5., 7., -5., 5., .5, .5))
        with self.assertRaises(ValueError): sample(dict(target, query_candidates=target['query_candidates'][:1]))
        missing = dict(target); missing.pop('label_sha256')
        with self.assertRaises(ValueError): prepare_query_candidates(missing, grid=GRID)

    def test_original_record_binding_is_read_only_and_checks_population(self):
        target = labels()
        tokens = ['scene'+str(row) for row in range(3)]
        with tempfile.TemporaryDirectory(prefix='query-identity-fixture-') as temp:
            root = Path(temp); (root/'records').mkdir()
            identity = dict(identity='original-cache', grid=GRID.__dict__, schema='structured_scene_cache_v1')
            (root/'identity.json').write_text(json.dumps(identity))
            for row, token in enumerate(tokens):
                (root/'records'/(token+'.json')).write_text(json.dumps(dict(token=token,
                    cache_identity='original-cache', label_sha256=target['label_sha256'][row])))
            before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            bound = bind_query_label_identities(target, tokens, cache_root=root,
                                                expected_cache_identity='original-cache', grid=GRID)
            for key in ('occupancy', 'occupancy_valid', 'road_distance'): self.assertIs(bound[key], target[key])
            self.assertEqual(bound['label_sha256'], target['label_sha256'])
            prepare_query_candidates(bound, grid=GRID)
            after = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            with self.assertRaises(ValueError):
                bind_query_label_identities(target, tokens, cache_root=root, expected_cache_identity='foreign', grid=GRID)
            with self.assertRaises(ValueError):
                bind_query_label_identities(target, tokens, cache_root=root, expected_cache_identity='original-cache',
                                            grid=GridSpec(-5., 7., -5., 5., .5, .5))
            with self.assertRaises(ValueError):
                bind_query_label_identities(target, tokens[:1], cache_root=root, expected_cache_identity='original-cache', grid=GRID)

    def test_forward_finite_cache_shares_row_views_and_rechecks_mutations(self):
        field = torch.ones(3, 8, 1, GRID.height, GRID.width)
        cache = ForwardFieldFiniteCache()
        for row in range(3):
            read_time_field(field[row:row+1], torch.zeros(1, 8, 2), torch.arange(8)[None], GRID,
                            execution_fast=True, finite_cache=cache)
        self.assertEqual(cache.checks, 1)
        field[0, 2].fill_(float('nan'))
        self.assertFalse(cache.is_finite(field))
        self.assertEqual(cache.checks, 2)
        reference = read_time_field(field[:1], torch.zeros(1, 1, 2), torch.zeros(1, 1, dtype=torch.long), GRID)
        fast = read_time_field(field[:1], torch.zeros(1, 1, 2), torch.zeros(1, 1, dtype=torch.long), GRID,
                               execution_fast=True, finite_cache=cache)
        close(reference[0], fast[0], exact=True)
        self.assertEqual(cache.checks, 2)
        self.assertTrue(cache.is_finite(torch.ones(1, dtype=torch.uint8)))
        self.assertEqual(cache.checks, 2)

    def test_loss_checks_prediction_fields_once_and_skips_verified_byte_labels(self):
        target = labels()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(181)
            model = StructuredSemantics('event', channels=2)
        predicted = model(torch.zeros(3, 2, GRID.height, GRID.width), torch.zeros(3, 8, 2, GRID.height, GRID.width))
        shapes, native = [], torch.isfinite
        def observed(tensor):
            if tensor.ndim >= 4: shapes.append(tuple(tensor.shape))
            return native(tensor)
        with patch('torch.isfinite', side_effect=observed):
            compute_structured_losses(predicted, target, dict(poses=torch.zeros(3, 16, 3), time_index=torch.arange(8).repeat(3, 2)),
                                      [BODY]*3, semantic_mode='event', statistics=STATS, grid=GRID, execution_fast=True)
        self.assertEqual(len(shapes), 2)  # All rows of logits and pt, each once.

    def test_read_time_field_values_gradients_NaN_XY_and_OOR(self):
        generator = torch.Generator().manual_seed(17)
        for steps in (6, 8):
            for mode in ('bilinear', 'nearest'):
                with self.subTest(steps=steps, mode=mode):
                    field = torch.randn(3, steps, 2, GRID.height, GRID.width, generator=generator, dtype=torch.float64)
                    xy = torch.randn(3, 37, 2, generator=generator, dtype=torch.float64)*3
                    xy[0, :7] = torch.tensor([[-6., -5.], [6., 0.], [0., 5.], [-6.1, 0.],
                                              [float('nan'), 0.], [0., float('inf')], [1000., 0.]])
                    times = torch.randint(-1, steps+1, (3, 37), generator=generator)
                    weight = torch.randn(3, 37, 2, generator=generator, dtype=torch.float64)
                    snapshots = []
                    for fast in (False, True):
                        f, x = field.clone().requires_grad_(), xy.clone().requires_grad_()
                        value, valid = read_time_field(f, x, times, GRID, mode=mode, execution_fast=fast)
                        gradients = torch.autograd.grad((value*weight).sum(), (f, x))
                        snapshots.append((value, valid, gradients))
                    close(snapshots[0][0], snapshots[1][0], exact=True)
                    close(snapshots[0][1], snapshots[1][1], exact=True)
                    for a, b in zip(snapshots[0][2], snapshots[1][2]): close(a, b)

    def test_nonfinite_unselected_time_preserves_reference_contamination(self):
        for bad in (float('nan'), float('inf'), -float('inf')):
            for mode in ('nearest', 'bilinear'):
                with self.subTest(bad=str(bad), mode=mode):
                    field = torch.ones(1, 3, 1, GRID.height, GRID.width, dtype=torch.float64)
                    field[:, 2].fill_(bad)
                    snapshots = []
                    for fast in (False, True):
                        f = field.clone().requires_grad_()
                        xy = torch.zeros(1, 2, 2, dtype=torch.float64, requires_grad=True)
                        value, valid = read_time_field(f, xy, torch.zeros(1, 2, dtype=torch.long), GRID,
                                                      mode=mode, execution_fast=fast)
                        snapshots.append((value, valid, torch.autograd.grad(value.sum(), (f, xy))))
                    self.assertTrue(torch.isnan(snapshots[0][0]).all())
                    close(snapshots[0][0], snapshots[1][0], exact=True)
                    close(snapshots[0][1], snapshots[1][1], exact=True)
                    for a, b in zip(snapshots[0][2], snapshots[1][2]): close(a, b, exact=True)

    def test_missing_times_and_empty_queries_keep_zero_gradients(self):
        for count in (0, 4):
            snapshots = []
            for fast in (False, True):
                field = torch.ones(2, 3, 1, GRID.height, GRID.width, requires_grad=True)
                xy = torch.zeros(2, count, 2, requires_grad=True)
                times = torch.full((2, count), 3, dtype=torch.long)
                value, valid = read_time_field(field, xy, times, GRID, execution_fast=fast)
                gradients = torch.autograd.grad(value.sum(), (field, xy))
                self.assertFalse(valid.any())
                self.assertTrue(all(not torch.count_nonzero(g) for g in gradients))
                snapshots.append((value, gradients))
            close(snapshots[0][0], snapshots[1][0], exact=True)
            for a, b in zip(snapshots[0][1], snapshots[1][1]): close(a, b, exact=True)

    def test_grouped_kernel_is_2D_and_does_not_read_all_K_at_each_time(self):
        field = torch.arange(2*8*GRID.height*GRID.width).float().reshape(2, 8, 1, GRID.height, GRID.width)
        xy, times = torch.zeros(2, 64, 2), torch.arange(8).repeat(2, 8)
        shapes = []
        native = F.grid_sample
        def observed(fields, coordinates, **kwargs):
            shapes.append((tuple(fields.shape), tuple(coordinates.shape), kwargs))
            return native(fields, coordinates, **kwargs)
        with patch('torch.nn.functional.grid_sample', side_effect=observed):
            fast = read_time_field(field, xy, times, GRID, execution_fast=True)
        reference = read_time_field(field, xy, times, GRID)
        close(fast[0], reference[0], exact=True)
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes[0][0], (16, 1, GRID.height, GRID.width))
        self.assertEqual(shapes[0][1], (16, 8, 1, 2))
        self.assertFalse(shapes[0][2]['align_corners'])

    def test_noncontiguous_fields_and_prepared_coordinates(self):
        grid = GridSpec(-4., 4., -4., 4., .5, .5)
        generator = torch.Generator().manual_seed(27)
        field = torch.randn(2, 4, 3, 16, 16, generator=generator).transpose(-1, -2)
        xy = torch.randn(2, 19, 2, generator=generator, requires_grad=True)
        times = torch.randint(4, (2, 19), generator=generator)
        self.assertFalse(field.is_contiguous())
        reference = read_time_field(field, xy, times, grid)
        fast = read_time_field(field, xy, times, grid, execution_fast=True)
        close(reference[0], fast[0], exact=True)
        a = grid.sample(field[:, 0], xy)
        b = grid.sample(field[:, 0], xy, _prepared=grid.prepare_sample(xy))
        close(a[0], b[0], exact=True)
        close(torch.autograd.grad(a[0].sum(), xy)[0], torch.autograd.grad(b[0].sum(), xy)[0], exact=True)

    def test_probability_and_logit_interpolation_remain_distinct(self):
        grid = GridSpec(0., 2., 0., 1., 1., 1.)
        logits = torch.tensor([[[[[0., 4.]]]]], requires_grad=True)
        xy, time = torch.tensor([[[1., .5]]]), torch.zeros(1, 1, dtype=torch.long)
        outputs = []
        for fast in (False, True):
            probability = read_time_field(logits.sigmoid(), xy, time, grid, execution_fast=fast)[0]
            logit = read_time_field(logits, xy, time, grid, execution_fast=fast)[0]
            self.assertGreater(float((probability-logit.sigmoid()).abs()), .1)
            outputs.append((probability, logit))
        for a, b in zip(*outputs): close(a, b, exact=True)

    def test_body_template_preserves_values_pose_gradients_and_default_dtype(self):
        _cached_body_template.cache_clear()
        for pose_dtype in (torch.float32, torch.float64):
            for default_dtype in (torch.float32, torch.float64):
                with self.subTest(pose_dtype=str(pose_dtype), default_dtype=str(default_dtype)):
                    old = torch.get_default_dtype()
                    try:
                        torch.set_default_dtype(default_dtype)
                        snapshots = []
                        for fast in (False, True):
                            poses = torch.tensor([[0., 0., .7], [1., 2., -.3]], dtype=pose_dtype, requires_grad=True)
                            points = body_points(poses, BODY, lateral_offset_m=.1, execution_fast=fast)
                            snapshots.append((points, torch.autograd.grad(points.square().sum(), poses)[0]))
                        for a, b in zip(*snapshots): close(a, b, exact=True)
                    finally: torch.set_default_dtype(old)
        before = _cached_body_template.cache_info().hits
        body_points(torch.zeros(2, 3), BODY, lateral_offset_m=.1, execution_fast=True)
        self.assertGreater(_cached_body_template.cache_info().hits, before)

    def test_body_min_max_tie_gradients_split_across_every_point(self):
        poses = torch.zeros(1, 1, 3)
        time = torch.zeros(1, 1, dtype=torch.long)
        body = dict(length=2., width=2., rear_axle_to_center=0.)
        points = body_points(poses[0], body)
        # Zero is an exact plateau under bilinear interpolation. Nonzero
        # constants can differ by one float32 ULP and thus are not exact ties.
        expected_road = torch.zeros(1, 1, GRID.height, GRID.width, requires_grad=True)
        expected_occ = torch.zeros_like(expected_road, requires_grad=True)
        # A constant field ties at every footprint point: amin/amax distribute
        # gradients uniformly, whereas an indexed min/max concentrates them.
        expected = []
        for field in (expected_road, expected_occ):
            values = GRID.sample(field, points.reshape(1, -1, 2))[0]
            expected.append(torch.autograd.grad(values.mean(), field)[0])
        for fast in (False, True):
            road = expected_road.detach().clone().requires_grad_()
            occupancy = expected_occ.detach().clone()[:, None].requires_grad_()
            relation = body_relations(road, occupancy, poses, time, [body], grid=GRID, execution_fast=fast)
            grads = torch.autograd.grad(relation['road_margin_m'].sum()+relation['occupancy_overlap_proxy'].sum(), (road, occupancy))
            close(grads[0], expected[0])
            close(grads[1][:, 0], expected[1])

    def test_fractional_NaN_and_int32_time_indices_follow_equality_selection(self):
        field = torch.arange(4*GRID.height*GRID.width).float().reshape(1, 4, 1, GRID.height, GRID.width)
        for time in (torch.tensor([[0., 1., .5, float('nan'), float('inf'), -1., 4.]]),
                     torch.tensor([[0, 1, 2, 3, -1, 4, 2]], dtype=torch.int32)):
            xy = torch.zeros(1, 7, 2)
            reference = read_time_field(field, xy, time, GRID)
            fast = read_time_field(field, xy, time, GRID, execution_fast=True)
            close(reference[0], fast[0], exact=True)
            close(reference[1], fast[1], exact=True)

    def test_default_path_matches_original_git_reference(self):
        modules, _ = reference_modules()
        original_grid = modules['grid'].GridSpec(**GRID.__dict__)
        generator = torch.Generator().manual_seed(642)
        target = labels()
        proposal = torch.randn(3, 8, 3, generator=generator)
        for mode in ('uniform', 'local'):
            snapshots = []
            for queries, grid in ((modules['queries'], original_grid),
                                  (sys.modules[sample_queries.__module__], GRID)):
                rng = torch.Generator().manual_seed(234)
                q = queries.sample_queries(proposal, target, mode=mode, generator=rng, count=16, grid=grid)
                snapshots.append((q, rng.get_state()))
            for key in ('poses', 'time_index'): close(snapshots[0][0][key], snapshots[1][0][key], exact=True)
            self.assertEqual(snapshots[0][0]['strata_records'], snapshots[1][0]['strata_records'])
            self.assertTrue(torch.equal(snapshots[0][1], snapshots[1][1]))
        query = snapshots[0][0]
        for semantic in ('none', 'full', 'event'):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(162)
                model = StructuredSemantics(semantic, channels=2)
            current = torch.randn(3, 2, GRID.height, GRID.width, generator=generator)
            future = torch.randn(3, 8, 2, GRID.height, GRID.width, generator=generator)
            prediction = model(current, future)
            snapshots = []
            for compute, grid in ((modules['losses'].compute_structured_losses, original_grid),
                                  (compute_structured_losses, GRID)):
                poses = query['poses'].clone().requires_grad_()
                raw, counts = compute(prediction, target, dict(query, poses=poses), [BODY]*3,
                                      semantic_mode=semantic, statistics=STATS, grid=grid)
                gradient = torch.autograd.grad(sum(raw.values()), poses)[0]
                snapshots.append((raw, counts, gradient))
            for index in (0, 1):
                self.assertEqual(snapshots[0][index].keys(), snapshots[1][index].keys())
                for key in snapshots[0][index]: close(snapshots[0][index][key], snapshots[1][index][key], exact=True)
            close(snapshots[0][2], snapshots[1][2], exact=True)

    def test_loss_gradients_all_groups_horizons_and_empty_masks(self):
        for steps in (6, 8):
            for mode, distribution in [('none', 'uniform'), ('full', 'uniform'), ('event', 'uniform'), ('event', 'local')]:
                for empty in (False, True):
                    with self.subTest(steps=steps, mode=mode, distribution=distribution, empty=empty):
                        generator = torch.Generator().manual_seed(157)
                        with torch.random.fork_rng(devices=[]):
                            torch.manual_seed(42)
                            initial = StructuredSemantics(mode, channels=3)
                        target = labels(steps)
                        if empty:
                            target['road_valid'].zero_(); target['occupancy_valid'].zero_()
                        proposal = torch.zeros(3, steps, 3)
                        query = sample_queries(proposal, target, mode=distribution, generator=generator, count=16, grid=GRID)
                        # Include out-of-range poses while retaining valid queries.
                        query['poses'][0, 0, 0] = 100.
                        B0 = torch.randn(3, 3, GRID.height, GRID.width, generator=generator)
                        Bt = torch.randn(3, steps, 3, GRID.height, GRID.width, generator=generator)
                        snapshots = []
                        for fast in (False, True):
                            model = copy.deepcopy(initial)
                            current, future = B0.clone().requires_grad_(), Bt.clone().requires_grad_()
                            q = dict(query, poses=query['poses'].clone().requires_grad_())
                            rng = torch.get_rng_state().clone()
                            predicted = model(current, future)
                            raw, counts = compute_structured_losses(predicted, target, q, [BODY]*3,
                                semantic_mode=mode, statistics=STATS, global_scenes=5, grid=GRID, execution_fast=fast)
                            (sum(raw.values())+predicted['future_logits'].sum()*0.).backward()
                            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                            grads = {name: None if p.grad is None else p.grad.clone() for name, p in model.named_parameters()}
                            snapshots.append((raw, counts, grads, current.grad, future.grad, q['poses'].grad))
                        self.assertEqual(snapshots[0][0].keys(), snapshots[1][0].keys())
                        for a, b in zip(snapshots[0][:3], snapshots[1][:3]):
                            self.assertEqual(a.keys(), b.keys())
                            for key in a:
                                if a[key] is None: self.assertIsNone(b[key])
                                else: close(a[key], b[key])
                        for a, b in zip(snapshots[0][3:], snapshots[1][3:]): close(a, b)
                        if mode == 'none':
                            for name, grad in snapshots[1][2].items():
                                if name.startswith('future.'):
                                    self.assertIsNotNone(grad)
                                    self.assertFalse(torch.count_nonzero(grad))

    def test_G0_zero_gradient_AdamW_decay_and_optimizer_state_are_retained(self):
        original = reference_modules(FORMAL_REFERENCE_COMMIT)[0]
        original_grid = original['grid'].GridSpec(**GRID.__dict__)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(54)
            initial = StructuredSemantics('none', channels=2)
        target = labels(steps=6, batch=1)
        query = dict(poses=torch.zeros(1, 4, 3), time_index=torch.zeros(1, 4, dtype=torch.long))
        snapshots = []
        for compute, grid, fast in ((original['losses'].compute_structured_losses, original_grid, False),
                                    (compute_structured_losses, GRID, True)):
            model = copy.deepcopy(initial)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.2)
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                prediction = model(torch.ones(1, 2, GRID.height, GRID.width), torch.ones(1, 6, 2, GRID.height, GRID.width))
                raw, _ = compute(prediction, target, query, [BODY], semantic_mode='none',
                    statistics=STATS, grid=grid, **({'execution_fast': True} if fast else {}))
                (sum(raw.values())+prediction['future_logits'].sum()*0.).backward()
                self.assertTrue(all(p.grad is not None for p in model.future.parameters()))
                self.assertTrue(all(not torch.count_nonzero(p.grad) for p in model.future.parameters()))
                optimizer.step()
            snapshots.append((model.state_dict(), optimizer.state_dict()['state']))
        for a, b in zip(snapshots[0], snapshots[1]):
            for key in a:
                if isinstance(a[key], dict):
                    for field in a[key]: close(a[key][field], b[key][field], exact=True)
                else: close(a[key], b[key], exact=True)
        for name, parameter in initial.named_parameters():
            if name.startswith('future.'):
                close(snapshots[1][0][name], parameter.detach()*(1-.01*.2)**3)

    def test_loss_reuse_frozen0c54_all_prediction_gradients(self):
        original = reference_modules(FORMAL_REFERENCE_COMMIT)[0]
        original_grid = original['grid'].GridSpec(**GRID.__dict__)
        for steps in (6, 8):
            for group, mode in (('G0', 'none'), ('G1', 'full'), ('G3', 'event')):
                for kind in ('random', 'tie', 'OOR', 'all_invalid', 'prediction_NaN', 'floating_valid_NaN'):
                    with self.subTest(steps=steps, group=group, kind=kind):
                        gen = torch.Generator().manual_seed(815)
                        target = labels(steps)
                        target['occupancy_valid'][:2].fill_(True)
                        target['road_valid'][:2].fill_(True)
                        target['occupancy'][:2] = torch.randint(2, target['occupancy'][:2].shape, generator=gen).byte()
                        target['occupancy_valid'][1, :, 0, 0] = False
                        target['road_valid'][1, 0, 0] = False
                        poses = torch.rand(3, 12, 3, generator=gen)*2-1
                        times = torch.randint(steps, (3, 12), generator=gen)
                        source = {'road_distance_m': torch.randn(3, 1, GRID.height, GRID.width, generator=gen),
                                  'current_logits': torch.randn(3, 1, GRID.height, GRID.width, generator=gen),
                                  'future_logits': torch.randn(3, steps, 2, GRID.height, GRID.width, generator=gen),
                                  'pt': torch.rand(3, steps, 1, GRID.height, GRID.width, generator=gen)}
                        if kind == 'tie':
                            target['road_distance'].fill_(1.)
                            target['occupancy'][:2].fill_(1)
                            for value in source.values(): value.zero_()
                        elif kind == 'OOR':
                            poses[0, :4, :2] = torch.tensor([[-100., 0.], [0., 100.], [6., 0.], [float('nan'), 0.]])
                            times[1, :4] = torch.tensor([-1, steps, steps+1, 0])
                        elif kind == 'all_invalid':
                            target['road_valid'].zero_(); target['occupancy_valid'].zero_()
                        elif kind == 'prediction_NaN':
                            times.zero_()
                            source['future_logits'][:, -1].fill_(float('nan'))
                            source['pt'][:, -1].fill_(float('nan'))
                        elif kind == 'floating_valid_NaN':
                            times.zero_()
                            target['occupancy_valid'] = target['occupancy_valid'].float()
                            target['occupancy_valid'][:, -1].fill_(float('nan'))
                        snapshots = []
                        for compute, grid, fast in ((original['losses'].compute_structured_losses, original_grid, None),
                                                   (compute_structured_losses, GRID, False),
                                                   (compute_structured_losses, GRID, True)):
                            prediction = {key: value.clone().requires_grad_() for key, value in source.items()}
                            query = {'poses': poses.clone().requires_grad_(), 'time_index': times.clone()}
                            rng = torch.get_rng_state().clone()
                            raw, counts = compute(prediction, target, query, [BODY]*3, semantic_mode=mode,
                                statistics=STATS, global_scenes=5, grid=grid,
                                **({} if fast is None else {'execution_fast': fast}))
                            # Same G0 zero term as the VLA wrapper; retain its
                            # non-None zero gradient and AdamW participation.
                            total = sum(raw.values())+(prediction['future_logits'].sum()*0. if mode == 'none' else 0.)
                            leaves = list(prediction.values())+[query['poses']]
                            gradients = torch.autograd.grad(total, leaves, allow_unused=True)
                            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                            snapshots.append((raw, counts, gradients))
                        for at in (1, 2):
                            for first, candidate in zip(snapshots[0][:2], snapshots[at][:2]):
                                self.assertEqual(first.keys(), candidate.keys())
                                for key in first: close(first[key], candidate[key], exact=at == 1)
                            for first, candidate in zip(snapshots[0][2], snapshots[at][2]):
                                if first is None: self.assertIsNone(candidate)
                                else: close(first, candidate, exact=at == 1)
                        if mode == 'none':
                            self.assertIsNotNone(snapshots[2][2][2])
                            self.assertFalse(torch.count_nonzero(snapshots[2][2][2]))

    def test_loss_reuse_removes_only_discarded_valid_body_reads(self):
        loss_module = sys.modules[compute_structured_losses.__module__]
        query_module = sys.modules[read_time_field.__module__]
        query = {'poses': torch.zeros(3, 8, 3), 'time_index': torch.arange(8)[None].expand(3, -1)}
        target = labels()
        prediction = {'road_distance_m': torch.zeros(3, 1, GRID.height, GRID.width),
                      'current_logits': torch.zeros(3, 1, GRID.height, GRID.width),
                      'future_logits': torch.zeros(3, 8, 2, GRID.height, GRID.width),
                      'pt': torch.zeros(3, 8, 1, GRID.height, GRID.width)}
        for semantic in ('none', 'full', 'event'):
            calls = []
            for fast in (False, True):
                with patch.object(loss_module, 'body_relations', wraps=body_relations) as relations, \
                     patch.object(loss_module, 'read_time_field', wraps=read_time_field) as direct_reads, \
                     patch.object(query_module, 'read_time_field', wraps=read_time_field) as body_reads:
                    compute_structured_losses(prediction, target, query, [BODY]*3,
                        semantic_mode=semantic, statistics=STATS, grid=GRID, execution_fast=fast)
                    calls.append((relations.call_count, direct_reads.call_count+body_reads.call_count))
            self.assertEqual(calls[0][0], 3)
            self.assertEqual(calls[1][0], 2)
            self.assertEqual(calls[0][1]-calls[1][1], len(target['occupancy']))


class EvidenceResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.records.append(dict(case=test.id(), passed=True))

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        self.records.append(dict(case=subtest.id(), passed=err is None,
                                 error=None if err is None else str(err[1])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--loss-reuse-only', action='store_true', help='Bounded loss reuse / frozen0c54 gradient / G0 AdamW checks only')
    args = parser.parse_args()
    if args.output and args.output.exists(): raise ValueError('Evidence output must be fresh')
    torch.set_num_threads(2)
    start = time.monotonic()
    suite = (unittest.TestSuite(QueryExecutionTests(name) for name in (
        'test_loss_reuse_frozen0c54_all_prediction_gradients',
        'test_loss_reuse_removes_only_discarded_valid_body_reads',
        'test_G0_zero_gradient_AdamW_decay_and_optimizer_state_are_retained')) if args.loss_reuse_only
        else unittest.defaultTestLoader.loadTestsFromTestCase(QueryExecutionTests))
    result = unittest.TextTestRunner(verbosity=2, resultclass=EvidenceResult).run(suite)
    if args.output:
        files = ['starVLA/model/modules/structured_world/'+name+'.py' for name in ('queries', 'losses', 'grid')]
        files.append(str(Path(__file__).relative_to(ROOT)))
        report = dict(schema='query_execution_cpu_equivalence_v1', root=str(ROOT), python=sys.executable,
                      torch_version=torch.__version__, device='cpu', gpu_validation=False,
                      tests_run=result.testsRun, all_passed=result.wasSuccessful(), records=result.records,
                      elapsed_seconds=time.monotonic()-start,
                      source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},
                      failures=[dict(case=test.id(), error=error) for test, error in result.failures+result.errors],
                      reference_commit=REFERENCE_COMMIT, reference_source_sha256=reference_modules()[1],
                      formal_reference_commit=FORMAL_REFERENCE_COMMIT,
                      formal_reference_source_sha256=reference_modules(FORMAL_REFERENCE_COMMIT)[1],
                      loss_reuse_only=args.loss_reuse_only,
                      limits='Synthetic CPU queries/losses/optimizer; parent owns full-model GPU validation.')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as handle: json.dump(report, handle, indent=2)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__': sys.exit(main())
