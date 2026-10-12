"""Bounded CPU negative controls for the dedicated transport closure."""
import unittest
import recover_completed_navtest_transport as recovery


class Controls(unittest.TestCase):
    def test_exact_transport_pattern(self):
        recovery.failure_contract({'classification': 'unclassified', 'exit_codes': [255]*8+[0]*16})
        for bad in [{'classification': 'scientific', 'exit_codes': [255]*8+[0]*16},
                    {'classification': 'unclassified', 'exit_codes': [1]*8+[0]*16},
                    {'classification': 'unclassified', 'exit_codes': [255]*8}]:
            with self.assertRaises(RuntimeError): recovery.failure_contract(bad)

    def test_bad_score_never_recovered(self):
        row = {'token': 't', 'log': 'l', 'variant': 'q0', 'status': 'ok', 'score': 0.8}
        recovery.score_row(row, 't', 'l', 'q0', ['score'])
        for change in [{'status': 'failed'}, {'score': float('nan')}, {'score': float('inf')},
                       {'score': -0.1}, {'token': 'foreign'}, {'variant': 'q_final'}]:
            with self.assertRaises(RuntimeError): recovery.score_row(dict(row, **change), 't', 'l', 'q0', ['score'])

    def test_frozen_validator_bound(self):
        guard = recovery.load_guard()
        self.assertEqual(guard.EXECUTED_TOOL_SHA256, recovery.TOOL_SHA)


if __name__ == '__main__': unittest.main()
