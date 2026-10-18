"""Source/control tests only. Never dispatch a scorer or load model/data assets."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('score_once', HERE / 'score_once.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
RAW = adapter.SOURCE.read_bytes()


def control_functions():
    tree = adapter.prepare(RAW)
    names = {'canonical_host_v3', 'check_route_time_v3', 'check_score_budget'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {'ROUTE_HOSTS_V3': adapter.OLD_HOSTS + (adapter.ADDITIONAL_HOST,)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(adapter.SOURCE), 'exec'), ns)
    return ns


class ThinRouteTests(unittest.TestCase):
    def test_original_hash(self):
        self.assertEqual(hashlib.sha256(RAW).hexdigest(), adapter.SOURCE_SHA)

    def test_only_one_allowlist_literal_changes(self):
        old = ast.parse(RAW)
        new = adapter.prepare(RAW)
        assigns = [n for n in new.body if isinstance(n, ast.Assign)
                   and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                   and n.targets[0].id == 'ROUTE_HOSTS_V3']
        self.assertEqual(len(assigns), 1)
        self.assertEqual(ast.literal_eval(assigns[0].value), adapter.OLD_HOSTS + (adapter.ADDITIONAL_HOST,))
        assigns[0].value = ast.parse(repr(adapter.OLD_HOSTS), mode='eval').body
        self.assertEqual(ast.dump(old, include_attributes=False), ast.dump(new, include_attributes=False))

    def test_changed_original_source_rejected(self):
        with self.assertRaisesRegex(ValueError, 'launcher changed'):
            adapter.prepare(RAW + b'\n')

    def test_unexpected_allowlist_rejected_even_with_test_pin(self):
        changed = RAW.replace(repr(adapter.OLD_HOSTS).encode(), b"('training-vlawm-tsinghua-zt6',)")
        self.assertNotEqual(changed, RAW)
        saved = adapter.SOURCE_SHA
        try:
            adapter.SOURCE_SHA = hashlib.sha256(changed).hexdigest()
            with self.assertRaisesRegex(ValueError, 'allowlist'):
                adapter.prepare(changed)
        finally:
            adapter.SOURCE_SHA = saved

    def test_actual_zt4_requires_policy_and_campaign_authorization(self):
        ns = control_functions()
        policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
        auth = json.loads(Path('/mnt/project/st-world-policy-20261016/resources/AUTHORIZATION_20261016T203442.json').read_text())
        host = 'training-vlawm-tsinghua-zt4-worker-0'
        self.assertEqual(ns['canonical_host_v3'](host, policy, auth), adapter.ADDITIONAL_HOST)
        bad_auth = copy.deepcopy(auth)
        bad_auth['allowed_hosts'].remove(adapter.ADDITIONAL_HOST)
        with self.assertRaises(AssertionError):
            ns['canonical_host_v3'](host, policy, bad_auth)
        bad_policy = copy.deepcopy(policy)
        bad_policy['allowed_aliases'].remove(host)
        with self.assertRaises(AssertionError):
            ns['canonical_host_v3'](host, bad_policy, auth)
        bad_policy = copy.deepcopy(policy)
        bad_policy['global_dispatch_prohibitions'][adapter.ADDITIONAL_HOST] = {'aliases': [host]}
        with self.assertRaises(AssertionError):
            ns['canonical_host_v3'](host, bad_policy, auth)

    def test_not_arbitrary_tsinghua_or_withdrawn_host(self):
        ns = control_functions()
        policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
        auth = json.loads(Path('/mnt/project/st-world-policy-20261016/resources/AUTHORIZATION_20261016T203442.json').read_text())
        for host in ['training-vlawm-tsinghua-zt6-worker-0', 'training-vlawm-zt3-worker-0', 'unknown']:
            with self.subTest(host=host), self.assertRaises(AssertionError):
                ns['canonical_host_v3'](host, policy, auth)

    def test_route_time_boundary_and_expiry_unchanged(self):
        f = control_functions()['check_route_time_v3']
        f({'valid_from_unix': 1, 'valid_until_unix': 5260}, 1000, 4260)
        for route, now in [({'valid_from_unix': 1, 'valid_until_unix': 5259}, 1000),
                           ({'valid_from_unix': 1001, 'valid_until_unix': 6000}, 1000),
                           ({'valid_from_unix': 1, 'valid_until_unix': float('inf')}, 1000)]:
            with self.subTest(route=route), self.assertRaises(AssertionError):
                f(route, now, 4260)

    def test_completed_output_and_two_invocations_still_refused(self):
        f = control_functions()['check_score_budget']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); reg = root/'reg'; out = root/'out'
            reg.mkdir(); out.mkdir()
            self.assertEqual(f(reg, out, False), [])
            (reg/'score_invocations'/'one').mkdir(parents=True)
            with self.assertRaises(AssertionError):
                f(reg, out, False)
            (out/'official_score').mkdir()
            (out/'official_score'/'IDENTITY.json').write_text('{}')
            self.assertEqual(len(f(reg, out, True)), 1)
            (reg/'score_invocations'/'two').mkdir()
            with self.assertRaises(AssertionError):
                f(reg, out, True)
            (out/'official_score'/'COMPLETE.json').write_text('{}')
            with self.assertRaises(AssertionError):
                f(reg, out, True)

    def test_current_immutable_routes_are_for_requested_zt4_jobs(self):
        pins = {'C1': 'f5507b9cb2a96ac728b555c6d66f5ebb335cd8f93e23f6391b8783ceb6bc8647',
                'C4': '53e27c1f49c94b05ef44230ba16dac644339d419f4998dcddf8b1a6f7bc1a099'}
        for arm, pin in pins.items():
            job = f'{arm}_step004096_dev1696_seed42_v1'
            path = Path('/mnt/project/st-world-policy-20261016/formal/evaluation/registrations') / job / 'ROUTE.json'
            raw = path.read_bytes(); route = json.loads(raw)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), pin)
            self.assertEqual(route['job_id'], job)
            self.assertEqual(route['host'], adapter.ADDITIONAL_HOST)
            self.assertEqual(route['inference']['score_workers'], 2)
            self.assertEqual(route['inference']['score_max_calls'], 1728)


if __name__ == '__main__':
    unittest.main(verbosity=2)
