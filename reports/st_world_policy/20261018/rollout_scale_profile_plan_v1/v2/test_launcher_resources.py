"""Independent synthetic controls for the actual frozen diagnostic launcher.

Extract only fresh_host AST. No imports from the trainer, subprocesses, CUDA,
remote resource inspection, model files or original training data are executed.
"""
import ast
import hashlib
import json
import pathlib
import types
import unittest

LAUNCHER = pathlib.Path('/mnt/project/st-world-policy-20261016/resources/launch_rollout_scale_diagnostic_v1.py')


class FreshHostControls(unittest.TestCase):
    def setUp(self):
        self.source = LAUNCHER.read_bytes()
        tree = ast.parse(self.source)
        constants = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                try:
                    constants[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue
        self.host = constants['HOST']
        self.sha = constants['SHA']
        self.busy = False
        self.real_app = False
        self.low_ram = False
        self.low_disk = False
        self.dirty = False
        self.foreign_host = False
        self.uuids = ['GPU-test-' + str(i) for i in range(4)]

        def command(args, **kwargs):
            if args[:3] == ['git', 'rev-parse', 'HEAD']:
                return self.sha + '\n'
            if args[:2] == ['git', 'status']:
                return ' M changed.py\n' if self.dirty else ''
            if args[0] == 'nvidia-smi':
                if any('query-compute-apps' in arg for arg in args):
                    fields = next(arg.split('=', 1)[1].split(',') for arg in args
                                  if arg.startswith('--query-compute-apps='))
                    values = {'pid': '123', 'gpu_uuid': self.uuids[0],
                              'used_gpu_memory': '1', 'process_name': 'real_job'}
                    return ', '.join(values[field] for field in fields) + '\n' if self.real_app else ''
                return '\n'.join(f'{i}, {uid}, NVIDIA A800-SXM4-80GB, 81920, '
                                 f'{100 if self.busy else 0}, {50 if self.busy else 0}'
                                 for i, uid in enumerate(self.uuids))
            raise AssertionError('Unmocked subprocess: ' + repr(args))

        def read(path):
            if str(path).endswith('server_dispatch_policy.json'):
                return {'allowed_aliases': [self.host]}
            if str(path).endswith('AUTHORIZATION_20261016T203442.json'):
                return {'allowed_hosts': [self.host.removesuffix('-worker-0')]}
            raise AssertionError('Unmocked JSON: ' + str(path))

        def fake_path(value):
            return types.SimpleNamespace(
                read_text=lambda: 'MemAvailable: %d kB\n' % ((1 if self.low_ram else 1024) * 2**20),
                is_dir=lambda: True)

        env = dict(constants, ROOT=pathlib.Path('/synthetic'), REPO=pathlib.Path('/synthetic/runtime'),
                   read=read, Path=fake_path,
                   socket=types.SimpleNamespace(gethostname=lambda: 'foreign' if self.foreign_host else self.host),
                   os=types.SimpleNamespace(sched_getaffinity=lambda _: set(range(128))),
                   subprocess=types.SimpleNamespace(check_output=command),
                   shutil=types.SimpleNamespace(disk_usage=lambda _: types.SimpleNamespace(
                       free=(1 if self.low_disk else 4096) * 2**30)))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'fresh_host')
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(LAUNCHER), 'exec'), env)
        self.check = env['fresh_host']

    def test_healthy_metadata(self):
        rows, _ = self.check()
        self.assertEqual(len(rows), 4)

    def test_foreign_host_rejected(self):
        self.foreign_host = True
        with self.assertRaises((AssertionError, ValueError)):
            self.check()

    def test_busy_gpu_rejected(self):
        self.busy = True
        with self.assertRaises((AssertionError, ValueError)):
            self.check()

    def test_low_memory_rejected(self):
        self.low_ram = True
        with self.assertRaises((AssertionError, ValueError)):
            self.check()

    def test_low_disk_rejected(self):
        self.low_disk = True
        with self.assertRaises((AssertionError, ValueError)):
            self.check()

    def test_dirty_runtime_rejected(self):
        self.dirty = True
        with self.assertRaises((AssertionError, ValueError)):
            self.check()

    def test_zero_usage_real_gpu_process_rejected(self):
        self.real_app = True
        with self.assertRaises((AssertionError, ValueError)):
            self.check()


if __name__ == '__main__':
    print(json.dumps({'launcher': str(LAUNCHER), 'sha256': hashlib.sha256(LAUNCHER.read_bytes()).hexdigest(),
                      'scope': 'synthetic AST controls, no real subprocess/GPU/model/data access'}), flush=True)
    unittest.main(verbosity=2)
