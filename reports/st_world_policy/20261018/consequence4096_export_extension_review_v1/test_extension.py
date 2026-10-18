"""Source-only independent review: do not import/execute the exporter main."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

WRAPPER = Path('/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261018/consequence4096_route_extension_v1/export_once.py')
WRAPPER_SHA = 'c69fa15e42e8c43973ceba41ed05d4caec21fae5d193176aea5fe3a97d8ef5a0'
assert hashlib.sha256(WRAPPER.read_bytes()).hexdigest() == WRAPPER_SHA
spec = importlib.util.spec_from_file_location('reviewed_extension', WRAPPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
RAW = helper.SOURCE.read_bytes()
ORIGINAL = ast.parse(RAW)
PATCHED = helper.prepare(RAW)
POLICY_PATH = Path('/mnt/project/server_dispatch_policy.json')
AUTH_PATH = Path('/mnt/project/st-world-policy-20261016/resources/AUTHORIZATION_20261016T203442.json')
POLICY = json.loads(POLICY_PATH.read_text())
AUTH = json.loads(AUTH_PATH.read_text())


def functions(tree):
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in ('canonical_host_v3', 'check_route_time_v3')]
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'ROUTE_HOSTS_V3'
                              for t in node.targets))
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[assignment, *selected], type_ignores=[])),
                 'source_only_review', 'exec'), namespace)
    return namespace


class ReviewTests(unittest.TestCase):
    def test_exactly_one_allowlist_change(self):
        reverted = copy.deepcopy(PATCHED)
        assignment = next(n for n in reverted.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'ROUTE_HOSTS_V3' for t in n.targets))
        self.assertEqual(ast.literal_eval(assignment.value), helper.OLD_HOSTS + (helper.ADDITIONAL_HOST,))
        assignment.value = ast.parse(repr(helper.OLD_HOSTS), mode='eval').body
        self.assertEqual(ast.dump(ORIGINAL, include_attributes=False), ast.dump(reverted, include_attributes=False))

    def test_modified_source_rejected(self):
        with self.assertRaises(ValueError):
            helper.prepare(RAW + b'\n# changed\n')

    def test_current_authorized_host_accepts_and_old_rejects(self):
        host = 'training-vlawm-tsinghua-zt4-worker-0'
        with self.assertRaises(AssertionError):
            functions(ORIGINAL)['canonical_host_v3'](host, POLICY, AUTH)
        self.assertEqual(functions(PATCHED)['canonical_host_v3'](host, POLICY, AUTH), helper.ADDITIONAL_HOST)

    def test_absent_authorization_and_alias_ban_rejected(self):
        canonical = functions(PATCHED)['canonical_host_v3']
        host = 'training-vlawm-tsinghua-zt4-worker-0'
        auth = copy.deepcopy(AUTH)
        auth['allowed_hosts'].remove(helper.ADDITIONAL_HOST)
        with self.assertRaises(AssertionError):
            canonical(host, POLICY, auth)
        policy = copy.deepcopy(POLICY)
        policy['global_dispatch_prohibitions']['review-fixture-ban'] = {'aliases': [host]}
        with self.assertRaises(AssertionError):
            canonical(host, policy, AUTH)
        with self.assertRaises(AssertionError):
            canonical('unapproved-host', POLICY, AUTH)

    def test_original_time_rejections_preserved(self):
        check = functions(PATCHED)['check_route_time_v3']
        check({'valid_from_unix': 0, 'valid_until_unix': 8000}, 100, 7860)
        for route in ({'valid_from_unix': 101, 'valid_until_unix': 10000},
                      {'valid_from_unix': 0, 'valid_until_unix': 7900},
                      {'valid_from_unix': 0, 'valid_until_unix': float('inf')}):
            with self.assertRaises(AssertionError):
                check(route, 100, 7860)

    def test_actual_wrapper_identity_is_retained(self):
        text = WRAPPER.read_text()
        self.assertIn("__file__=str(Path(__file__).resolve())", text)
        self.assertIn("'launcher_sha256':sha(__file__)", RAW.decode())


if __name__ == '__main__':
    unittest.main(verbosity=2)
