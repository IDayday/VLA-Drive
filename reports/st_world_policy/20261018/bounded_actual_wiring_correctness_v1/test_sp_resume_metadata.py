"""Independent metadata-only review; never calls main/native resolver/Popen."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import unittest

SOURCE = Path('/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261018/spatial20k_resume_invocation_v1/launch.py')
SHA = 'c4593d8e3e5444ee2b31afc2043836a5742658ba777be0ed1176030e681ce6a8'
spec = importlib.util.spec_from_file_location('independent_sp_resume', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ResumeMetadata(unittest.TestCase):
    def check_arm(self, arm, step):
        self.assertEqual(m.sha(SOURCE), SHA)
        rp = m.ROOT / 'formal/spatial/registrations' / f'{arm}_fulltrain_seed42_official_20k_v1_resource_v2/REGISTRATION.json'
        reg = json.loads(rp.read_text())
        cfg = m.pinned(reg['artifacts']['config'])
        lease = m.pinned(reg['artifacts']['resource_lease'])
        host = lease['hostnames'][0]
        now = lease['valid_from_unix'] + 100
        a = dict(schema='st_spatial20k_resume_invocation_v1', status='APPROVED',
                 run_id=reg['run_id'], registration=dict(path=str(rp), sha256=m.sha(rp)),
                 source_sha=m.SOURCE, helper=dict(path=str(SOURCE), sha256=SHA),
                 no_automatic_resume=True, max_invocation_seconds=28800,
                 valid_from_unix=now-1, valid_until_unix=now+40000, hostnames=[host],
                 cuda_visible_devices_by_host=lease['cuda_visible_devices_by_host'],
                 gpu_uuids_by_host=lease['gpu_uuids_by_host'], expected_completed=step,
                 remaining_cache_growth_bytes=768*2**30,
                 separate_conservative_metadata_buffer_bytes=10*2**30)
        # This synthetic admission exists only in memory, never as an actual receipt.
        m.validate(a, rp, reg, cfg, lease, host, now)
        for key, value in [('expected_completed', 5000), ('expected_completed', 20000),
                           ('run_id', 'wrong'), ('source_sha', '0'*40), ('status', 'TEMPLATE'),
                           ('no_automatic_resume', False), ('valid_from_unix', now+1),
                           ('valid_until_unix', now+29579), ('remaining_cache_growth_bytes', 0),
                           ('separate_conservative_metadata_buffer_bytes', 0), ('hostnames', []),
                           ('helper', dict(path=str(SOURCE), sha256='0'*64))]:
            bad = copy.deepcopy(a)
            bad[key] = value
            with self.subTest(arm=arm, key=key, value=value), self.assertRaises(AssertionError):
                m.validate(bad, rp, reg, cfg, lease, host, now)
        for key, value in [('scope', 'PROFILE'), ('world_size', 2), ('micro_batch', 1),
                           ('global_batch', 16), ('max_updates', 240), ('stop_after', 12000)]:
            bad = copy.deepcopy(cfg)
            bad['execution'][key] = value
            with self.subTest(arm=arm, config=key), self.assertRaises(AssertionError):
                m.validate(a, rp, reg, bad, lease, host, now)

    def test_sp2_actual_small_metadata(self):
        self.check_arm('SP2', 11762)

    def test_sp3_actual_small_metadata(self):
        self.check_arm('SP3', 11582)

    def test_resume_resolution_and_lock_wiring(self):
        source = SOURCE.read_text()
        main = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        constants = [n.value for n in ast.walk(main) if isinstance(n, ast.Constant)]
        for value in ['--resume', '29400s', '--kill-after=180s', 'BUDGET_PAUSED',
                      'RESUME_DISPATCH.lock', 'RUN.lock']:
            self.assertIn(value, constants)
        self.assertIn("resolution['selected']['receipt'] == expected", source)
        self.assertLess(source.index('resolve_latest(out /'), source.index('invocation.mkdir('))
        self.assertLess(source.index('invocation.mkdir('), source.index('subprocess.Popen('))
        self.assertIn('exist_ok=False', source)
        self.assertIn('pass_fds=tuple(f.fileno() for f in locks)', source)


if __name__ == '__main__':
    unittest.main(verbosity=2)
