"""Small-source boundary fixtures; no Torch/model/index import or real-data claim."""
import ast
from copy import deepcopy
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]

def function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), namespace)
    return namespace[name]

class Boundary(unittest.TestCase):
    def setUp(self):
        mod = types.ModuleType('tools.foresight.student_state')
        mod.validate_rank_batches = function('tools/foresight/student_state.py', 'validate_rank_batches', {})
        self.imports = patch.dict(sys.modules, {'tools.foresight.student_state': mod})
        self.imports.start(); self.addCleanup(self.imports.stop)
        self.consume = function('tools/st_world_policy/teacher_provider.py', 'full_profile_consumption',
                                {'CONSUMPTION_SCHEMA': 'st_full_index_profile_consumption_v1'})
        self.cfg = {'execution': {'scope': 'PROFILE', 'world_size': 8, 'micro_batch': 4,
                    'global_batch': 32, 'stop_after': 240, 'sample_limit': 101592, 'max_updates': 20000},
                    'runtime': {'profile_mode': 'timing'}, 'optimizer': {'warmup': 1000}}

    def test_fixed_timing_exposures_and_no_approval(self):
        got = self.consume(self.cfg)
        self.assertEqual(got['cumulative_scene_exposures'], 7680)
        self.assertEqual(got['per_rank_scene_exposures'], [960] * 8)
        self.assertEqual(got['gradient_accumulation_steps'], 1)
        self.assertFalse(got['formal_global32_throughput_qualified'])
        self.assertEqual(got['resume_budget'], 'timing_requires_fresh_complete_windows_no_resume')

    def test_original_diagnostic_unchanged(self):
        for world in (1, 2):
            cfg = deepcopy(self.cfg); cfg['execution'].update(world_size=world, micro_batch=1, global_batch=2, stop_after=64)
            cfg['runtime']['profile_mode'] = 'none'; got = self.consume(cfg)
            self.assertEqual(got['cumulative_scene_exposures'], 128)
            self.assertEqual(got['per_rank_scene_exposures'], [128 // world] * world)
            self.assertEqual(got['resume_budget'], 'same_run_cumulative_not_per_invocation')

    def test_reject_nearby_topology_scope_budget(self):
        for field, value in [('world_size', 4), ('micro_batch', 2), ('global_batch', 64),
                             ('stop_after', 239), ('stop_after', 241), ('sample_limit', 101591),
                             ('max_updates', 240), ('world_size', True), ('scope', 'FORMAL')]:
            with self.subTest(field=field, value=value):
                cfg = deepcopy(self.cfg); cfg['execution'][field] = value
                with self.assertRaises(ValueError): self.consume(cfg)
        for mode in ('none', 'trace'):
            cfg = deepcopy(self.cfg); cfg['runtime']['profile_mode'] = mode
            with self.assertRaises(ValueError): self.consume(cfg)
        cfg = deepcopy(self.cfg); cfg['optimizer']['warmup'] = 20
        with self.assertRaises(ValueError): self.consume(cfg)
        cfg = deepcopy(self.cfg); cfg['continuation'] = {}
        with self.assertRaises(ValueError): self.consume(cfg)

    def test_legacy_tail_empty_rank_is_still_rejected(self):
        cfg = deepcopy(self.cfg); cfg['execution'].update(world_size=2,micro_batch=1,global_batch=2,stop_after=64,sample_limit=257)
        cfg['runtime']['profile_mode'] = 'none'
        with self.assertRaisesRegex(ValueError, 'Empty rank tail'): self.consume(cfg)

    def test_timing_and_math_sources_unchanged(self):
        import subprocess
        names = ['tools/st_world_policy/train_world.py', 'tools/st_world_policy/profile_pipeline.py',
                 'tools/st_world_policy/fixed_batch_teacher.py', 'starVLA/dataloader/st_world_indexed_provider.py']
        for path in names:
            old = subprocess.check_output(['git', 'show', 'a8b0036f:'+path], cwd=ROOT)
            self.assertEqual((ROOT/path).read_bytes(), old)

if __name__ == '__main__': unittest.main(verbosity=2)
