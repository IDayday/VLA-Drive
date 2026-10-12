"""CPU-only safety/denominator tests for explicit 8/16-rank profile admission."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.structured_world.world_scaling_profile import validate_args, validate_ranks, contract
from tools.foresight.student_state import validate_rank_batches


def arguments(**overrides):
    args = SimpleNamespace(world_scaling_profile=True, scope='profile',
        use_formal_schedule=True, benchmark_no_end_save=True, debug_updates=220,
        resume=False, resume_origin_run=None, pilot_origin_run=None, pilot_origin_tag=None,
        capture_updates=None, capture_storage_budget=None, debug_scenes=0, stop_after=0,
        geometry=Path('/shared/geometry.pt'), geometry_identity='registered',
        execution_mode='io_preserving_v1', micro_batch=4)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class WorldScalingTests(unittest.TestCase):
    def test_fresh_eight_and_sixteen_rank_admission(self):
        for world in (8, 16):
            for mode in ('io_preserving_v1', 'engineering_host_fast'):
                args = arguments(micro_batch=32 // world, execution_mode=mode)
                validate_args(args)
                for rank in range(world):
                    validate_ranks(args, rank=rank, world=world, local=rank % 8, local_world=8)
                identity = contract(args, world)
                self.assertEqual(identity['microbatch'] * identity['world_size'], 32)
                self.assertFalse(identity['native_checkpoint_restored'])
                self.assertFalse(identity['formal_authorized'])

    def test_never_expands_formal_or_resume_scope(self):
        changes = dict(scope='formal', use_formal_schedule=False,
            benchmark_no_end_save=False, debug_updates=221, resume=True,
            resume_origin_run=Path('/parent'), pilot_origin_run=Path('/parent'),
            pilot_origin_tag='update_25000', capture_updates='32',
            capture_storage_budget=Path('/budget'), debug_scenes=64, stop_after=32,
            geometry=None, geometry_identity=None, execution_mode='precision_fast')
        for key, value in changes.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_args(arguments(**{key: value}))
        for steps in (-1, 0, 221, 100000):
            with self.assertRaises(ValueError): validate_args(arguments(debug_updates=steps))

    def test_fixed_batch_nodes_and_rank_mapping(self):
        valid = dict(rank=8, world=16, local=0, local_world=8)
        args = arguments(micro_batch=2)
        for field, value in (('world', 4), ('world', 32), ('rank', 16),
                             ('rank', -1), ('local', 1), ('local_world', 4)):
            invalid = {**valid, field: value}
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_ranks(args, **invalid)
        for micro in (1, 4, 8):
            with self.assertRaises(ValueError):
                validate_ranks(arguments(micro_batch=micro), **valid)

    def test_original_guard_is_not_relaxed(self):
        args = arguments(world_scaling_profile=False)
        for micro in (1, 2, 4):
            args.micro_batch = micro
            validate_ranks(args, rank=0, world=8, local=0, local_world=8)
            with self.assertRaises(ValueError):
                validate_ranks(args, rank=0, world=16, local=0, local_world=8)
        self.assertIsNone(contract(args, 8))

    def test_true_tail_population_and_matching_backward_count(self):
        # The exact existing denominator helper: never pad/drop the NAVSIM tail.
        for world in (8, 16):
            validate_rank_batches(101592, 32, world, 32 // world)
            tail = [list(range(rank, 24, world)) for rank in range(world)]
            self.assertEqual(sorted(i for rows in tail for i in rows), list(range(24)))
            self.assertTrue(all(0 < len(rows) <= 32 // world for rows in tail))
            # Averaging world*local_numerator/global_count equals scene mean.
            values = list(range(1, 25))
            distributed = sum(world * sum(values[i] for i in rows) / 24 for rows in tail) / world
            self.assertAlmostEqual(distributed, sum(values) / 24)
        with self.assertRaises(ValueError): validate_rank_batches(101592, 32, 16, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
