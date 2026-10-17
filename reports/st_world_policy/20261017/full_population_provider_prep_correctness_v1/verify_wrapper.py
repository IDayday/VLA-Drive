"""Independent stdlib fixtures only. Never calls the real preparation/model/DB."""
import time
START = time.monotonic()
with open('/proc/self/io') as stream:
    BOOT = {k: int(v) for k, v in (line.split(':') for line in stream)}
import ast
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import resource
import signal
import sys
import tempfile
from contextlib import redirect_stdout, redirect_stderr
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('review wall budget')))
signal.alarm(110)
HERE = Path(__file__).resolve().parent
OWNER = HERE.parent / 'full_population_provider_preparation_v1'

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

author = load('independent_author_fixtures', OWNER / 'test_prepare.py')
m = author.m
bridge = load('independent_pinned_reads', HERE.parent / 'w_storage_bridge_v1/storage_bridge.py')
PLAN = json.loads((OWNER / 'PLAN.json').read_text())

def pin(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

class Independent(unittest.TestCase):
    def test_exact_source_pins_and_actual_parser_contract(self):
        self.assertEqual(pin(OWNER / 'prepare_once.py')['sha256'], '9ef4b038705613d38fee96a26ff5798d21ad7c4daa50009243685bd6a940ce24')
        self.assertEqual(pin(OWNER / 'launch_once.sh')['sha256'], '0a7083b66ddfc11bbdf53c41870be13c09e907c0a3c7bfb3735a37aa1e83a2aa')
        self.assertEqual(pin(OWNER / 'PLAN.json')['sha256'], m.PLAN_SHA)
        for item in PLAN['source_files'].values():
            p = Path(item['path'])
            self.assertLess(p.stat().st_size, 1 << 20)
            self.assertEqual(pin(p), item)
        qualify = Path(PLAN['source_files']['tools/st_world_policy/qualify_indexed_provider.py']['path'])
        provider = Path(PLAN['source_files']['tools/st_world_policy/teacher_provider.py']['path'])
        import argparse
        tree = ast.parse(qualify.read_text())
        parser = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'parser')
        ns = {'argparse': argparse, '__doc__': 'isolated actual parser'}
        exec(compile(ast.Module(body=[parser], type_ignores=[]), str(qualify), 'exec'), ns)
        ns['parser']().parse_args(m.commands(PLAN)[0][1])
        tree = ast.parse(provider.read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        statements = []
        for statement in main.body:
            if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'args' for t in statement.targets):
                break
            statements.append(statement)
        ns = {'argparse': argparse, '__doc__': 'isolated actual parser'}
        exec(compile(ast.Module(body=statements, type_ignores=[]), str(provider), 'exec'), ns)
        parsed = ns['p'].parse_args(m.commands(PLAN)[1][1])
        self.assertEqual(parsed.profile_future_complete, PLAN['inputs']['future_complete']['path'])
        self.assertTrue(parsed.profile_full_population)
        self.assertEqual(parsed.configured_output, str(m.ROOT / 'configured'))
        self.assertEqual(parsed.profile_source_migration, str(m.ROOT / 'SOURCE_MIGRATION.json'))

    def test_stale_index_config_and_source_pins_rejected_by_actual_reader(self):
        fixture = author.Preparation()
        fixture.setUp()
        with tempfile.TemporaryDirectory(prefix='st-provider-correctness-') as directory:
            base = Path(directory)
            for key in fixture.plan['inputs']:
                path = base / (key + '.json')
                path.write_text(json.dumps({'status': 'COMPLETE', 'stage': 'fit'}))
                fixture.plan['inputs'][key] = pin(path)
            for key in fixture.plan['source_files']:
                path = base / (Path(key).stem + '.py')
                path.write_text('# synthetic source\n')
                fixture.plan['source_files'][key] = pin(path)
            def validate():
                with patch.object(m.os, 'sched_getaffinity', return_value={0, 1}):
                    m.validate(fixture.ad, fixture.plan, None, bridge.Reads(), time.time())
            validate()
            for group, key in [('inputs', 'train_index_complete'), ('inputs', 'current_index_complete'),
                               ('inputs', 'reference_config'), ('inputs', 'teacher_config'),
                               ('inputs', 'future_complete'), ('inputs', 'pca_fit_index_complete'),
                               ('source_files', 'tools/st_world_policy/teacher_provider.py')]:
                path = Path(fixture.plan[group][key]['path'])
                original = path.read_bytes()
                path.write_bytes(original + b' ')
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                    validate()
                path.write_bytes(original)

    def child(self, failure=None, missing=None):
        with tempfile.TemporaryDirectory(prefix='st-provider-child-fixture-') as directory:
            root = Path(directory)
            plan = copy.deepcopy(PLAN)
            calls = []
            def atomic_json(path, value, immutable=False):
                self.assertTrue(immutable)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
            def cli(module, run_name):
                self.assertEqual(run_name, '__main__')
                calls.append((module, sys.argv[1:]))
                if len(calls) == 2:
                    self.assertTrue((root / 'SOURCE_MIGRATION.json').is_file())
                print(json.dumps({'original_phase': len(calls)}))
                print(json.dumps({'original_second_json': len(calls)}))
                if failure == len(calls):
                    raise SystemExit(7)
                names = ['SOURCE_MIGRATION.json'] if len(calls) == 1 else [
                    'PROVIDER.json', 'configured/training_config.json',
                    'configured/PROVIDER_CHANNEL_SCOPE.json', 'configured/PROVIDER_CONFIG_LINEAGE.json']
                for name in names:
                    if name != missing:
                        atomic_json(root / name, {'fixture': name}, immutable=True)
                raise SystemExit(0)
            class Reads:
                def document(self, item):
                    return plan if item['path'].endswith('PLAN.json') else {}
                def raw(self, path):
                    return Path(path).read_bytes()
            fake_core = SimpleNamespace()
            bootstrap = SimpleNamespace(CORE_SHA='fixture', load=lambda *args: fake_core, read_source=lambda *args: b'')
            metadata = ModuleType('tools.st_world_policy.metadata_storage')
            metadata.tree_inventory = lambda *args, **kwargs: {'charged_bytes': 0}
            parent = ModuleType('tools.st_world_policy'); parent.metadata_storage = metadata
            target = ModuleType('starVLA.dataloader.st_world_targets'); target.atomic_json = atomic_json
            modules = {'supervise_once': ModuleType('supervise_once'),
                'observed_cli': SimpleNamespace(proc_io=lambda: dict(m.BOOT_IO)),
                'tools': ModuleType('tools'), 'tools.st_world_policy': parent,
                'tools.st_world_policy.metadata_storage': metadata, 'starVLA': ModuleType('starVLA'),
                'starVLA.dataloader': ModuleType('starVLA.dataloader'), 'starVLA.dataloader.st_world_targets': target}
            out, err = io.StringIO(), io.StringIO()
            args = SimpleNamespace(child=True, admission='fixture-admission', admission_sha256='f' * 64)
            old_path = list(sys.path)
            try:
                with patch.dict(sys.modules, modules), patch.object(m, 'ROOT', root), \
                     patch.object(m, 'validate'), patch.object(m.os, 'fstat'), \
                     patch.object(m.os, 'getpid', return_value=1234), patch.object(m.os, 'getpgrp', return_value=1234), \
                     patch.dict(os.environ, {'ST_PROBE_SUPERVISOR_LOCK_FD': '999', 'CUDA_VISIBLE_DEVICES': ''}), \
                     patch.object(m.runpy, 'run_module', side_effect=cli), redirect_stdout(out), redirect_stderr(err):
                    code = m.run(args, bootstrap, None, SimpleNamespace(Reads=Reads), time.monotonic() + 20)
            finally:
                sys.path[:] = old_path
            envelope = json.loads(out.getvalue())
            self.assertEqual(out.getvalue().count('\n'), 1)
            self.assertEqual(envelope['schema'], 'st_full_future_child_result_v2')
            self.assertEqual(err.getvalue().count('original_phase'), len(calls))
            self.assertEqual(err.getvalue().count('original_second_json'), len(calls))
            return code, envelope, calls, (root / 'SOURCE_MIGRATION.json').exists()

    def test_actual_child_success_two_systemexit_zero_and_single_envelope(self):
        code, envelope, calls, migration = self.child()
        self.assertEqual(code, 0)
        self.assertEqual([c[0] for c in calls], [c[0] for c in m.commands(PLAN)])
        self.assertTrue(migration)
        self.assertIsNone(envelope['error'])
        result = envelope['cli_result']
        self.assertEqual(result['status'], 'PROVIDER_PREPARATION_COMPLETE')
        self.assertEqual(len(result['outputs']), 5)
        self.assertFalse(result['training_started'])

    def test_actual_child_first_cli_failure_stops_before_provider(self):
        code, envelope, calls, migration = self.child(failure=1)
        self.assertEqual(code, 2)
        self.assertEqual(len(calls), 1)
        self.assertFalse(migration)
        self.assertIsNone(envelope['cli_result'])
        self.assertEqual(envelope['error']['type'], 'RuntimeError')

    def test_actual_child_second_cli_failure_preserves_first_phase_no_retry(self):
        code, envelope, calls, migration = self.child(failure=2)
        self.assertEqual(code, 2)
        self.assertEqual(len(calls), 2)
        self.assertTrue(migration)
        self.assertIsNone(envelope['cli_result'])
        self.assertEqual(envelope['error']['type'], 'RuntimeError')

    def test_actual_child_missing_output_does_not_claim_complete(self):
        code, envelope, calls, migration = self.child(missing='configured/PROVIDER_CHANNEL_SCOPE.json')
        self.assertEqual(code, 2)
        self.assertEqual(len(calls), 2)
        self.assertTrue(migration)
        self.assertIsNone(envelope['cli_result'])
        self.assertEqual(envelope['error']['type'], 'FileNotFoundError')

    def test_real_runpy_system_exit_and_argv_restoration(self):
        with tempfile.TemporaryDirectory(prefix='st-provider-runpy-fixture-') as directory:
            path = Path(directory) / 'tiny_cli_fixture.py'
            path.write_text('import json,sys\nprint(json.dumps({"one":1}))\nprint(json.dumps({"two":2}))\nraise SystemExit(int(sys.argv[1]))\n')
            before = sys.argv
            sys.path.insert(0, directory)
            try:
                out, err = io.StringIO(), io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    m.original_cli('tiny_cli_fixture', ['0'])
                    with self.assertRaisesRegex(RuntimeError, 'Original CLI failed'):
                        m.original_cli('tiny_cli_fixture', ['3'])
                self.assertIs(sys.argv, before)
                self.assertEqual(out.getvalue(), '')
                self.assertEqual(len(err.getvalue().splitlines()), 4)
            finally:
                sys.path.remove(directory)

if __name__ == '__main__':
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(author.Preparation),
                               unittest.defaultTestLoader.loadTestsFromTestCase(Independent)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    with open('/proc/self/io') as stream:
        end = {k: int(v) for k, v in (line.split(':') for line in stream)}
    cost = {'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'wall_seconds': time.monotonic() - START, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'logical_read_bytes': end['rchar'] - BOOT['rchar'], 'logical_written_bytes': end['wchar'] - BOOT['wchar'],
        'torch_imported': 'torch' in sys.modules, 'actual_preparation': False,
        'real_payload_or_database_read': False, 'model_or_pca_read': False, 'ledger_access': False}
    print('COST_JSON=' + json.dumps(cost, sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
