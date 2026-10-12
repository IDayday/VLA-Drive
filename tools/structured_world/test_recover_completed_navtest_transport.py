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

    def test_new_case_cannot_reinterpret_old_exit_codes(self):
        case = recovery.CASES['g3_90k']
        recovery.failure_contract({'classification': 'unclassified', 'exit_codes': [0]*8+[255]*8}, case)
        for codes in [[255]*8+[0]*16, [0]*7+[1]+[255]*8, [0]*16, [255]*16]:
            with self.assertRaises(RuntimeError):
                recovery.failure_contract({'classification': 'unclassified', 'exit_codes': codes}, case)

    def test_worker_identity_and_status_cannot_be_repaired(self):
        case = recovery.CASES['g3_90k']
        report = dict(shard=8, shards=16, attempt=case['attempt'], prediction_identity='identity',
            routing_tool_sha256=case['tool_sha'], status='INFERENCE', scenes_done=288)
        launch = dict(shard=8, shards=16, command=['ssh'])
        recovery.report_contract(report, launch, 8, 'identity', case)
        for changed in [dict(status='FAILED'), dict(error='CUDA error'), dict(prediction_identity='other'),
                        dict(attempt='other'), dict(shards=8), dict(status='COMPLETE')]:
            with self.assertRaises(RuntimeError):
                recovery.report_contract(dict(report, **changed), launch, 8, 'identity', case)
        live_report = dict(report, shard=0, status='COMPLETE')
        recovery.report_contract(live_report, dict(launch, shard=0), 0, 'identity', case)
        with self.assertRaises(RuntimeError):
            recovery.report_contract(dict(live_report, status='INFERENCE'), dict(launch, shard=0), 0, 'identity', case)

    def test_frozen_validator_bound(self):
        guard = recovery.load_guard()
        self.assertEqual(guard.EXECUTED_TOOL_SHA256, recovery.TOOL_SHA)
        guard = recovery.load_guard(recovery.CASES['g3_90k'])
        self.assertEqual(guard.EXECUTED_TOOL_SHA256, recovery.CASES['g3_90k']['tool_sha'])


if __name__ == '__main__': unittest.main()
