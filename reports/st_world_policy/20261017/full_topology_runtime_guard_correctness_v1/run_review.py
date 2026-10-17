"""Narrow AST-executed guard tests, without importing the training runtime."""
import os
import resource
import signal
import time
from pathlib import Path
import sys
import ast
import copy
import dataclasses
import hashlib
import importlib.util
import json
import subprocess
import types
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
os.sched_setaffinity(0,{2,3})
resource.setrlimit(resource.RLIMIT_AS,(1<<30,1<<30))
signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Bounded review deadline')))
signal.alarm(90)
started=time.monotonic()
source=(ROOT/'tools/st_world_policy/teacher_provider.py').read_text()
parent=subprocess.check_output(['git','show','a8b0036f:tools/st_world_policy/teacher_provider.py'],cwd=ROOT).decode()
spec=importlib.util.spec_from_file_location('author_boundary_tests',ROOT/'tests/st_world/test_full_profile_timing_boundary.py')
author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)

class Independent(unittest.TestCase):
    def test_only_guard_and_help_changed(self):
        before={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(parent).body if isinstance(n,ast.FunctionDef)}
        after={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
        self.assertEqual(before.keys(),after.keys())
        self.assertEqual({k for k in before if before[k]!=after[k]}, {'full_profile_consumption','main'})
        self.assertEqual(source.replace('Predeclared cumulative diagnostic or fixed world8 timing240 consumption JSON',
                         'Predeclared cumulative global2 diagnostic consumption JSON').split('def main():')[1],
                         parent.split('def main():')[1])

    def test_original_trainer_actually_rejects_timing_resume(self):
        tree=ast.parse((ROOT/'tools/st_world_policy/train_world.py').read_text())
        outer=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
                   and ast.unparse(n.test)=="profile_mode != 'none'")
        inner=outer.body[0]
        self.assertIsInstance(inner,ast.If)
        self.assertEqual(ast.unparse(inner.test),'args.resume')
        tiny=ast.Module(body=[copy.deepcopy(inner)],type_ignores=[])
        for flag in (True,False):
            ns={'args':types.SimpleNamespace(resume=flag)}
            if flag:
                with self.assertRaisesRegex(ValueError,'complete separately registered measurement window'):
                    exec(compile(tiny,'<unchanged-trainer-resume-guard>','exec'),ns)
            else:exec(compile(tiny,'<unchanged-trainer-resume-guard>','exec'),ns)

    def test_two_original_timing_windows_and_exposure_accounting(self):
        tree=ast.parse((ROOT/'tools/st_world_policy/profile_pipeline.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TimingPlan')
        ns={'dataclass':dataclasses.dataclass,'__name__':__name__}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<unchanged-timing-plan>','exec'),ns)
        plan=ns['TimingPlan']()
        self.assertEqual(plan.updates,240)
        self.assertEqual(plan.steady_steps(0),list(range(21,121)))
        self.assertEqual(plan.steady_steps(1),list(range(141,241)))
        batch_indices=[list(range(i*32,(i+1)*32)) for i in range(240)]
        per_rank=[sum(len(batch[r::8]) for batch in batch_indices) for r in range(8)]
        self.assertEqual(sum(map(len,batch_indices)),7680)
        self.assertEqual(per_rank,[960]*8)

    def test_original_zero_disk_and_nonformal_guards_retained(self):
        # No real factory/admission inputs are opened. The exact unchanged
        # function ASTs retain the already reviewed downstream protections.
        def fn(text,name):
            return next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
        for name in ('_validate_full_population','validate_profile_recipe','_qualified_profile_execution','load_descriptor'):
            self.assertEqual(ast.dump(fn(source,name)),ast.dump(fn(parent,name)))
        self.assertIn("future.get('tensor_disk_bytes') != 0",source)
        self.assertIn("admission['formal_approved'] is not False",source)
        text=subprocess.check_output(['git','show','210be55c:tools/st_world_policy/check_formal_approval.py'],cwd=ROOT).decode()
        self.assertIn('reg["execution"]["scope"] != "FORMAL" and reg["weights_usable_for_formal_init"]',text)
        old=subprocess.check_output(['git','show','a8b0036f:tools/st_world_policy/check_formal_approval.py'],cwd=ROOT).decode()
        self.assertEqual(text,old)

suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(author),
                         unittest.defaultTestLoader.loadTestsFromTestCase(Independent)])
result=unittest.TextTestRunner(verbosity=2).run(suite)
io={k:int(v) for k,v in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}
ru=resource.getrusage(resource.RUSAGE_SELF)
assert 'torch' not in sys.modules
cost={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
      'wall_seconds':time.monotonic()-started,'peak_parent_rss_bytes':ru.ru_maxrss*1024,
      'cpu_affinity':sorted(os.sched_getaffinity(0)),'parent_rchar_including_startup_bytes':io['rchar'],
      'teacher_provider_sha256':hashlib.sha256(source.encode()).hexdigest(),
      'torch_imported':False,'actual_training_or_payload_validation':False,
      'read_accounting_note':'Parent rchar includes git stdout; read-only git children inspect exact small source blobs, internal git reads not separately measured.'}
(HERE/'TEST_RESULT.json').write_text(json.dumps(cost,indent=2,sort_keys=True)+'\n')
print(json.dumps(cost,sort_keys=True))
assert io['rchar']<64<<20
signal.alarm(0)
raise SystemExit(0 if result.wasSuccessful() else 1)
