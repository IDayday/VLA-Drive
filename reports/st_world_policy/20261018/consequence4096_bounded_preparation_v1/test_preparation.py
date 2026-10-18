"""Source/metadata-only frontend checks. No analysis, CSV, tensor or GPU IO."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location('common4096_frontend_test', HERE / 'run_common_once.py')
front = importlib.util.module_from_spec(spec); spec.loader.exec_module(front)


class PreparationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = front.original()
        cls.core, _, _ = cls.module.dependencies()

    def test_original_commands_all_four(self):
        doc = self.core.pinned(front.FIXED)
        for arm in front.ARMS:
            command, plan = front.fixed_command(self.module, self.core, arm)
            expected = doc['arms'][arm]['original_command_unchanged']
            for field in ('argv', 'env', 'cwd', 'max_seconds'):
                self.assertEqual(command[field], expected[field])
            self.assertEqual(self.module.option(command, 'stage'), 'COMMON_ENDPOINT')
            self.assertEqual(plan['populations']['dev']['scenes'], 1696)
            self.assertEqual(plan['populations']['dev']['logs'], 16)

    def test_no_arbitrary_arm(self):
        with self.assertRaises(ValueError): front.fixed_command(self.module, self.core, 'C5')

    def test_command_tampering_rejected(self):
        original = self.core.pinned
        def changed(pin, *args, **kwargs):
            value = original(pin, *args, **kwargs)
            if pin == front.FIXED:
                value = copy.deepcopy(value)
                value['arms']['C2']['original_command_unchanged']['argv'][-1] = '/tmp/not-the-endpoint'
            return value
        with patch.object(self.core, 'pinned', side_effect=changed):
            with self.assertRaises(ValueError): front.fixed_command(self.module, self.core, 'C2')

    def fixture_inputs(self, complete_step=4096, valid=1696, source=None):
        command, _ = front.fixed_command(self.module, self.core, 'C2')
        budget, items = {'runs': {}}, {}
        for side, arm in (('control', 'C0'), ('treatment', 'C2')):
            refs = {name: {'path': self.module.option(command, side + '-' + directory) + '/' + file,
                           'sha256': arm + '-' + name}
                    for name, directory, file in (('score_identity','score','IDENTITY.json'),
                                                  ('export_identity','export','IDENTITY.json'),
                                                  ('score_complete','score','COMPLETE.json'))}
            budget['runs'][arm] = refs
            checkpoint = {'completed': complete_step, 'run': '/runs/' + arm + '_fulltrain_fixed'}
            export = {'identity': arm + 'e', 'checkpoint': checkpoint, 'source_sha': source or self.module.SOURCE}
            score = {'identity': arm + 's', 'checkpoint': checkpoint, 'export_identity': arm + 'e',
                     'export_file_sha256': refs['export_identity']['sha256']}
            complete = {'status':'COMPLETE','requested_scenes':1696,'valid_scored':valid,
                        'invalid':0,'unresolved':1696-valid,'identity':arm+'s'}
            for key, value in (('score_identity',score),('export_identity',export),('score_complete',complete)):
                items[refs[key]['path']] = value
        return command, budget, items

    def test_complete4096_receipts_accepted_without_payload(self):
        command, budget, items = self.fixture_inputs()
        with patch.object(self.core, 'pinned', side_effect=lambda pin: items[pin['path']]):
            front.verify_inputs(self.module, self.core, command, budget)

    def test_old_endpoint_and_partial_rejected(self):
        for kwargs in ({'complete_step':1024}, {'valid':1695}, {'source':'changed'}):
            command, budget, items = self.fixture_inputs(**kwargs)
            with patch.object(self.core, 'pinned', side_effect=lambda pin: items[pin['path']]):
                with self.assertRaises(ValueError): front.verify_inputs(self.module, self.core, command, budget)

    def test_output_verifier_only_changes_stage(self):
        # AST equality proves no previous full-payload verification was removed.
        def function(path):
            tree = ast.parse(Path(path).read_text())
            return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'complete_output')
        old = function(front.WRAPPER['path'])
        new = function(HERE / 'run_common_once.py')
        new.args.args.pop(0)
        class Unqualify(ast.NodeTransformer):
            def visit_Attribute(self, node):
                self.generic_visit(node)
                if isinstance(node.value, ast.Name) and node.value.id == 'module':
                    return ast.Name(id=node.attr, ctx=node.ctx)
                return node
            def visit_Constant(self, node):
                if node.value == 'COMMON_ENDPOINT': return ast.Constant(value='DIAGNOSTIC_1024')
                return node
        new = Unqualify().visit(new)
        self.assertEqual(ast.dump(old, include_attributes=False), ast.dump(new, include_attributes=False))

    def test_reviewed_monitor_is_not_reimplemented(self):
        self.assertEqual(self.module.run_pair.__module__, 'common4096_reviewed_wrapper')
        self.assertEqual(self.module.PAIR_LIMITS['rss_bytes'], 2 << 30)
        self.assertEqual(self.module.PAIR_LIMITS['wall_seconds'], 1800)
        self.assertEqual(self.module.PAIR_LIMITS['cleanup_seconds'], 40)
        self.assertEqual(self.module.TOTAL_READ, 1 << 30)
        self.assertEqual(self.module.OUTPUT_LIMIT, 32 << 20)
        self.assertEqual(self.module.ANALYSIS_LIMIT + self.module.CONTROL_RESERVE
                         + (16 + 8 + 8 + 32) * 1024, self.module.OUTPUT_LIMIT)

    def test_consumed_session_refuses_new_allowance(self):
        command, _ = front.fixed_command(self.module, self.core, 'C2')
        with patch.object(Path, 'exists', return_value=True):
            with self.assertRaises(ValueError): self.module.session_state([command])

    def test_child_routes_only_fixed_binding(self):
        class Child:
            def child(self):
                module = self.original()
                command, budget = self.fixed_command(module, object())
                return command, budget, self.BOOT_IO
        child = Child()
        with patch.object(front, 'load_source', return_value=child), \
                patch.object(front, 'original', return_value=object()), \
                patch.object(front, 'read_binding', return_value=({'comparison':'C3_minus_C0'}, {'stage':'COMMON_ENDPOINT'}, {})) as read:
            command, budget, io = front.child('C3')
        self.assertEqual(command['comparison'], 'C3_minus_C0')
        self.assertEqual(budget['stage'], 'COMMON_ENDPOINT')
        self.assertEqual(read.call_args.args[-1], 'C3')
        self.assertIs(io, front.BOOT_IO)


if __name__ == '__main__': unittest.main(verbosity=2)
