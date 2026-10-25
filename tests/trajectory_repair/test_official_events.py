import sys
import unittest
import numpy as np
from starVLA.rl.trajectory_repair.official_events import OfficialEventTrace, event_assignment_line


class TinyOfficialScorer:
    def __init__(self):
        self._collision_time_idcs = [float('inf'), float('inf')]
        self._ttc_time_idcs = [float('inf'), float('inf')]

    def _calculate_no_at_fault_collision(self):
        proposal_idx, token, collision_type, no_at_fault_collision_score = 1, 'object', 'FRONT', 0.
        for time_idx in [10, 11]:
            self._collision_time_idcs[proposal_idx] = min(time_idx, self._collision_time_idcs[proposal_idx])

    def _calculate_ttc(self):
        proposal_idx, token, future_time_idx = 1, 'object', 3
        for time_idx in [5, 6]:
            current_time_idx = time_idx + future_time_idx
            self._ttc_time_idcs[proposal_idx] = min(time_idx, self._ttc_time_idcs[proposal_idx])


class OfficialTraceTest(unittest.TestCase):
    def test_original_method_results_unchanged_and_full_events_retained(self):
        baseline = TinyOfficialScorer()
        baseline._calculate_no_at_fault_collision()
        baseline._calculate_ttc()
        traced = TinyOfficialScorer()
        with OfficialEventTrace(TinyOfficialScorer) as capture:
            traced._calculate_no_at_fault_collision()
            traced._calculate_ttc()
        self.assertEqual(baseline._collision_time_idcs, traced._collision_time_idcs)
        self.assertEqual(baseline._ttc_time_idcs, traced._ttc_time_idcs)
        rows = capture.attach([dict(simulated_states=np.zeros((41, 11)))])
        self.assertEqual(np.flatnonzero(rows[0]['official_at_fault_collision_by_time']).tolist(), [10, 11])
        self.assertEqual(np.flatnonzero(rows[0]['official_ttc_rule_infraction_by_time']).tolist(), [5, 6])
        self.assertEqual(rows[0]['official_event_trace']['TTC'][0]['object_time_index'], 8)
        self.assertIsNone(sys.gettrace())

    def test_exception_restores_tracing_state(self):
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with OfficialEventTrace(TinyOfficialScorer):
                raise RuntimeError('injected')
        self.assertIsNone(sys.gettrace())

    def test_existing_debugger_is_not_silently_replaced(self):
        previous = lambda *_: None
        sys.settrace(previous)
        try:
            with self.assertRaises(RuntimeError):
                with OfficialEventTrace(TinyOfficialScorer):
                    pass
            self.assertIs(sys.gettrace(), previous)
        finally:
            sys.settrace(None)

    def test_changed_assignment_contract_rejected(self):
        with self.assertRaises(ValueError):
            event_assignment_line(TinyOfficialScorer._calculate_ttc, '_wrong_field')


if __name__ == '__main__':
    unittest.main()
