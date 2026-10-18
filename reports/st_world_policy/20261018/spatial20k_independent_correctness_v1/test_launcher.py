"""CPU-only launcher branch tests; external approval/GPU/process APIs are doubles.

Only filesystem constants are redirected in the parsed launcher. The execution
body is unchanged. This does not test model resume or claim a real dispatch.
"""
import ast
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

SOURCE = Path('/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261018/spatial20k_actual_preparation_v1/launch20k_once.py')
PIN = 'e175da58b7e9f0b128c3059c13ab3a2dd7b0c65063daead80a6253165efe79e2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LauncherTests(unittest.TestCase):
    def exercise(self, mutation=None, expected=None):
        self.assertEqual(sha(SOURCE), PIN)
        with tempfile.TemporaryDirectory(prefix='spatial20k-launch-review-') as tmp:
            root = Path(tmp)
            repo = root / 'repo'
            regdir = root / 'formal/spatial/registrations/SP2_fulltrain_seed42_official_20k_v1'
            regdir.mkdir(parents=True)
            (root / 'resources').mkdir()
            parent_cfg = {'execution': {'scope': 'FORMAL', 'stop_after': 5000,
                'max_updates': 20000, 'world_size': 4, 'global_batch': 32},
                'runtime': {'output_root': str(root/'formal/spatial/runs'), 'max_seconds': 28800}}
            cfg = copy.deepcopy(parent_cfg)
            cfg['execution']['stop_after'] = 20000
            cfg['continuation'] = {'parent_registration': {'path': str(root/'parent.json')}}
            parent = {'artifacts': {'config': {'path': str(root/'parent_config.json')}}}
            reg = {'run_id': regdir.name, 'source_sha': 'a'*40, 'artifacts': {
                'config': {'path': str(regdir/'CONFIG.json')},
                'resource_lease': {'path': str(regdir/'LEASE.json')}}}
            cp = {'schema': 'st_checkpoint_storage_commitments_v1', 'status': 'CURRENT',
                  'checked_at_unix': 900, 'valid_until_unix': 40000,
                  'remaining_bytes': 4143*2**30, 'free_floor_bytes': 1024*2**30}
            lease = {'run_id': regdir.name, 'source_sha': 'a'*40,
                     'first_continuation_only': True, 'no_automatic_resume': True,
                     'hostnames': ['test-worker-0'], 'valid_from_unix': 900,
                     'valid_until_unix': 40000, 'max_invocation_seconds': 28800,
                     'maximum_optimizer_updates_this_stage': 15000,
                     'cuda_visible_devices_by_host': {'test-worker-0': '0,1,2,3'},
                     'gpu_uuids_by_host': {'test-worker-0': ['GPU-0','GPU-1','GPU-2','GPU-3']},
                     'checkpoint_commitments': {'path': str(root/'cp.json'), 'sha256': ''},
                     'remaining_cache_growth_bytes': 100*2**30,
                     'separate_conservative_metadata_buffer_bytes': 10*2**30}
            rows = '0,GPU-0,0,0\n1,GPU-1,0,0\n2,GPU-2,0,0\n3,GPU-3,0,0\n'
            state = {'cfg': cfg, 'parent_cfg': parent_cfg, 'reg': reg,
                     'lease': lease, 'cp': cp, 'rows': rows, 'free': 6000*2**30,
                     'host': 'test-worker-0', 'approval_error': None, 'lock_error': None,
                     'root': root, 'regdir': regdir, 'bad_cp_hash': False}
            if mutation:
                mutation(state)
            def put(path, value):
                Path(path).write_text(json.dumps(value))
            put(root/'cp.json', cp)
            lease['checkpoint_commitments']['sha256'] = '0'*64 if state['bad_cp_hash'] else sha(root/'cp.json')
            put(root/'parent.json', parent)
            if 'continuation' in cfg:
                cfg['continuation']['parent_registration']['sha256'] = ('0'*64 if state.get('bad_parent_hash')
                                                                       else sha(root/'parent.json'))
            put(regdir/'CONFIG.json', cfg)
            put(regdir/'LEASE.json', lease)
            put(regdir/'REGISTRATION.json', reg)
            put(root/'parent_config.json', parent_cfg)
            put(root/'policy.json', {'allowed_aliases': ['test-worker-0']})
            put(root/'resources/AUTHORIZATION_20261016T203442.json', {'allowed_hosts': ['test']})
            put(regdir/'RUN_APPROVAL.json', {'already_written_by_lead': True})
            events = []
            def approval(*args, **kwargs):
                events.append('approval')
                if state['approval_error']:
                    raise state['approval_error']
                return {'status': 'VERIFIED'}
            gate = types.ModuleType('tools.st_world_policy.check_formal_approval')
            gate.file_hash = sha
            gate.verify_registration = lambda *a, **kw: {'test': 'separate-real-fingerprint-audit-required'}
            gate.check_formal_approval = approval
            target = types.ModuleType('starVLA.dataloader.st_world_targets')
            def atomic(path, value, **kw):
                events.append(('write', str(path)))
                put(path, value)
            target.atomic_json = atomic
            budget = types.ModuleType('tools.st_world_policy.budget_extension')
            def recipe(config):
                result = copy.deepcopy(config)
                result.pop('continuation', None)
                result['execution'].pop('stop_after')
                result['runtime'].pop('output_root')
                return result
            budget._recipe = recipe
            tree = ast.parse(SOURCE.read_text())
            # Redirect only literal paths; retain every guard, branch, and argv.
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and node.value == '/mnt/project/st-world-policy-20261016':
                    node.value = str(root)
                elif isinstance(node, ast.Constant) and node.value == '/mnt/project/VLA-Drive-st-official-runtime-v2-20261016':
                    node.value = str(repo)
                elif isinstance(node, ast.Constant) and node.value == '/mnt/project/server_dispatch_policy.json':
                    node.value = str(root/'policy.json')
            compiled = compile(tree, str(SOURCE), 'exec')
            opened = []
            real_open = open
            def opened_lock(path, *args, **kwargs):
                if str(path).startswith('/tmp/foresight_gpu_'):
                    f = tempfile.TemporaryFile(mode='w+')
                    opened.append(f)
                    return f
                return real_open(path, *args, **kwargs)
            calls = []
            def popen(argv, **kwargs):
                events.append('popen')
                calls.append((argv, kwargs))
                return types.SimpleNamespace(pid=12345)
            mods = {gate.__name__: gate, target.__name__: target, budget.__name__: budget}
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.dict('sys.modules', mods))
                stack.enter_context(patch('sys.argv', [str(SOURCE), '--registration', str(regdir/'REGISTRATION.json')]))
                stack.enter_context(patch('socket.gethostname', return_value=state['host']))
                stack.enter_context(patch('time.time', return_value=1000))
                stack.enter_context(patch('fcntl.flock', side_effect=state['lock_error']))
                stack.enter_context(patch('builtins.open', side_effect=opened_lock))
                stack.enter_context(patch('subprocess.check_output', return_value=state['rows']))
                stack.enter_context(patch('subprocess.Popen', side_effect=popen))
                stack.enter_context(patch('shutil.disk_usage', return_value=types.SimpleNamespace(free=state['free'])))
                stack.enter_context(patch.dict(os.environ, {'FLASH_ATTENTION_DETERMINISTIC': '1'}))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                try:
                    if expected:
                        with self.assertRaises(expected):
                            exec(compiled, {'__name__': '__main__'})
                        self.assertEqual(calls, [])
                        self.assertFalse((regdir/'LAUNCH.json').exists() and not state.get('preexisting_launch'))
                    else:
                        exec(compiled, {'__name__': '__main__'})
                        self.assertEqual(len(calls), 1)
                        argv, kwargs = calls[0]
                        self.assertNotIn('--resume', argv)
                        self.assertIn('29400s', argv)
                        self.assertIn('--kill-after=180s', argv)
                        self.assertIn('--nproc_per_node=4', argv)
                        self.assertEqual(len(kwargs['pass_fds']), 4)
                        self.assertEqual(kwargs['cwd'], repo)
                        self.assertEqual(kwargs['env']['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
                        self.assertEqual(kwargs['env']['PYTHONDONTWRITEBYTECODE'], '1')
                        self.assertNotIn('FLASH_ATTENTION_DETERMINISTIC', kwargs['env'])
                        self.assertLess(events.index('approval'), events.index('popen'))
                        writes = [x[1] for x in events if isinstance(x, tuple)]
                        self.assertEqual(writes, [str(regdir/'LAUNCH.json')])
                finally:
                    for f in opened:
                        f.close()

    def test_first_continuation_argv_locks_and_external_approval(self):
        self.exercise()

    def test_rejected_branches_never_dispatch(self):
        cases = {
            'foreign_run': lambda s: s['reg'].update(run_id='SP0_other'),
            'old_stop': lambda s: s['cfg']['execution'].update(stop_after=5000),
            'wrong_global_batch': lambda s: s['cfg']['execution'].update(global_batch=16),
            'new_schedule': lambda s: s['cfg']['execution'].update(max_updates=35000),
            'new_world': lambda s: s['cfg']['execution'].update(world_size=8),
            'missing_parent': lambda s: s['cfg'].pop('continuation'),
            'wrong_parent_hash': lambda s: s.update(bad_parent_hash=True),
            'short_lease': lambda s: s['lease'].update(valid_until_unix=30579),
            'extra_stage_updates': lambda s: s['lease'].update(maximum_optimizer_updates_this_stage=20000),
            'wrong_lease_run': lambda s: s['lease'].update(run_id='foreign'),
            'wrong_lease_source': lambda s: s['lease'].update(source_sha='b'*40),
            'automatic_resume': lambda s: s['lease'].update(no_automatic_resume=False),
            'short_wall': lambda s: s['lease'].update(max_invocation_seconds=7200),
            'unapproved_host': lambda s: s.update(host='unknown-worker-0'),
            'occupied_gpu': lambda s: s.update(rows=s['rows'].replace('GPU-0,0,0','GPU-0,1024,50')),
            'wrong_uuid': lambda s: s.update(rows=s['rows'].replace('GPU-0','GPU-X')),
            'stale_cp': lambda s: s['cp'].update(valid_until_unix=30579),
            'wrong_cp_total': lambda s: s['cp'].update(remaining_bytes=2703*2**30),
            'wrong_cp_hash': lambda s: s.update(bad_cp_hash=True),
            'insufficient_cache_headroom': lambda s: s.update(free=(4143+1024+99)*2**30),
            'insufficient_metadata_headroom': lambda s: s.update(free=(4143+1024+100+9)*2**30),
            'existing_output': lambda s: (s['root']/'formal/spatial/runs'/s['reg']['run_id']).mkdir(parents=True),
        }
        for name, mutate in cases.items():
            with self.subTest(case=name):
                self.exercise(mutate, AssertionError)
        self.exercise(lambda s: s.update(approval_error=ValueError('stale approval')), ValueError)
        self.exercise(lambda s: s.update(lock_error=BlockingIOError('GPU lock owned')), BlockingIOError)
        def existing_launch(state):
            (state['regdir']/'LAUNCH.json').write_text('{}')
            state['preexisting_launch'] = True
        self.exercise(existing_launch, AssertionError)


if __name__ == '__main__':
    unittest.main(verbosity=2)
