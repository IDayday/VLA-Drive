"""CPU controls for prospective starts and the sole historical-slot exemption."""
import copy
import unittest

import enroll_fullgrad_remaining11 as registration


class ChronologyControls(unittest.TestCase):
    def fixture(self):
        jobs = [
            {'case': f'{group}/{stage}', 'repeat': repeat,
             'folder': f'/new/{group}/{stage}/{repeat}'}
            for group in ('G1', 'G3')
            for stage in ('initial', 'mid')
            for repeat in range(3)
        ]
        jobs[0]['folder'] = '/old/first'
        return jobs, {'job': copy.deepcopy(jobs[0])}

    def test_one_historical_prefix_and_eleven_prospective_slots(self):
        jobs, prefix = self.fixture()
        before = copy.deepcopy((jobs, prefix))
        result = registration.chronology_rule(jobs, prefix, '2026-10-11T17:00:00+00:00')
        self.assertEqual(len(result['remaining_slots_before_A']), 11)
        self.assertNotIn(['G1', 'initial', 0], result['remaining_slots_before_A'])
        self.assertFalse(result['prefix_exemption']['new_enrollment_applies'])
        self.assertFalse(result['prefix_exemption']['GPU_reexecution_allowed'])
        self.assertIn('NOT before this old A', result['prefix_exemption']['reason'])
        self.assertIn('All actual A/AA/B role starts', result['remaining_slot_start_requirement'])
        self.assertEqual((jobs, prefix), before)

    def test_cannot_exempt_a_second_already_run_slot(self):
        jobs, prefix = self.fixture()
        prefix['job'] = copy.deepcopy(jobs[1])
        with self.assertRaisesRegex(ValueError, 'Only actual first'):
            registration.chronology_rule(jobs, prefix, 'now')

    def test_equal_count_does_not_hide_missing_slot(self):
        jobs, prefix = self.fixture()
        jobs[-1] = copy.deepcopy(jobs[-2])
        with self.assertRaisesRegex(ValueError, 'coverage/order'):
            registration.chronology_rule(jobs, prefix, 'now')

    def test_short_phase_cannot_claim_original_twelve(self):
        jobs, prefix = self.fixture()
        with self.assertRaisesRegex(ValueError, 'coverage/order'):
            registration.chronology_rule(jobs[:-1], prefix, 'now')

    def test_first_prefix_cannot_be_relabelled_as_a_later_repeat(self):
        jobs, prefix = self.fixture()
        jobs[0], jobs[1] = jobs[1], jobs[0]
        prefix['job'] = copy.deepcopy(jobs[0])
        with self.assertRaisesRegex(ValueError, 'coverage/order'):
            registration.chronology_rule(jobs, prefix, 'now')


if __name__ == '__main__':
    unittest.main()
