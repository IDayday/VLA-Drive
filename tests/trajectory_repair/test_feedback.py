import copy
import unittest
import numpy as np

from starVLA.rl.trajectory_repair.feedback import candidate_evidence, read_feedback


def feedback(*, nc=1., dac=1., ttc=1., ep=1., speed=1.):
    states = np.zeros((41, 11))
    states[:, 3] = speed
    mask = np.zeros(41, dtype=bool)
    if dac < 1:
        mask[20:25] = True
    return dict(status='OK',
                requested_trajectory=np.zeros((8, 3)), simulated_states=states,
                non_drivable_body_by_time=mask,
                official_first_at_fault_collision_seconds=None if nc == 1 else 2.,
                official_first_ttc_infraction_seconds=None if ttc == 1 else 1.5,
                raw_progress_meters=4 * speed,
                metrics=dict(no_at_fault_collisions=nc, drivable_area_compliance=dac,
                             time_to_collision_within_bound=ttc, ego_progress=ep, comfort=1.,
                             driving_direction_compliance=1., score=nc * dac * (5 * ep + 5 * ttc + 2) / 12))


class FeedbackContractTest(unittest.TestCase):
    def test_distinct_request_and_simulation_time_axes(self):
        parsed = read_feedback(feedback())
        self.assertEqual(parsed.requested_local_poses.shape, (8, 3))
        self.assertEqual(parsed.simulated_global_states.shape, (41, 11))
        self.assertAlmostEqual(parsed.labels['request_time_seconds'][0], .5)
        self.assertAlmostEqual(parsed.labels['physical_time_seconds'][0], 0.)
        self.assertAlmostEqual(parsed.labels['physical_time_seconds'][-1], 4.)

    def test_missing_event_is_not_no_event(self):
        value = feedback()
        del value['official_first_ttc_infraction_seconds']
        parsed = read_feedback(value)
        self.assertFalse(parsed.available['first_ttc_rule_infraction'])
        self.assertIsNone(parsed.labels['first_ttc_rule_infraction_seconds'])
        self.assertTrue(read_feedback(feedback()).available['first_ttc_rule_infraction'])

    def test_first_event_does_not_invent_duration(self):
        parsed = read_feedback(feedback(nc=0., ttc=0.))
        self.assertEqual(parsed.labels['first_ttc_rule_infraction_seconds'], 1.5)
        self.assertFalse(parsed.available['full_ttc_rule_mask'])
        self.assertFalse(parsed.available['full_collision_mask'])
        self.assertIsNone(parsed.labels['full_collision_mask'])

    def test_geometric_overlap_mask_cannot_become_official_collision(self):
        value = feedback()
        value['full_collision_mask'] = np.zeros(41)
        with self.assertRaises(ValueError):
            read_feedback(value)

    def test_explicit_official_mask_is_available(self):
        value = feedback()
        value['official_at_fault_collision_by_time'] = np.zeros(41)
        parsed = read_feedback(value)
        self.assertTrue(parsed.available['full_collision_mask'])
        value['official_at_fault_collision_by_time'][10] = 1
        with self.assertRaises(ValueError):
            read_feedback(value)

    def test_nc_half_is_not_fully_safe(self):
        result = candidate_evidence([feedback(nc=.5)])
        self.assertFalse(result['feasible'][0])

    def test_infrastructure_error_rejected(self):
        value = feedback()
        value['status'] = 'WORKER_ERROR'
        with self.assertRaises(ValueError):
            read_feedback(value)

    def test_invalid_shapes_or_nonfinite_values_rejected(self):
        for field, value in [('requested_trajectory', np.zeros((8, 4))),
                             ('simulated_states', np.full((41, 11), np.nan)),
                             ('non_drivable_body_by_time', np.full(41, .5))]:
            record = feedback()
            record[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                read_feedback(record)

    def test_event_metric_consistency(self):
        value = feedback(ttc=0.)
        value['official_first_ttc_infraction_seconds'] = None
        with self.assertRaises(ValueError):
            read_feedback(value)

    def test_unknown_road_mask_is_not_safe_label(self):
        value = feedback()
        del value['non_drivable_body_by_time']
        parsed = read_feedback(value)
        self.assertFalse(parsed.available['road_body_mask'])
        self.assertIsNone(parsed.labels['road_body_outside_mask'])

    def test_dac_t0_included_in_official_metric(self):
        value = feedback(dac=0.)
        value['non_drivable_body_by_time'][:] = False
        value['non_drivable_body_by_time'][0] = True
        parsed = read_feedback(value)
        self.assertEqual(parsed.labels['road_body_outside_mask'][1:].sum(), 0)

    def test_slow_speed_alone_is_not_wrong(self):
        result = candidate_evidence([feedback(speed=0.), feedback(speed=4.)])
        self.assertFalse(result['has_safe_higher_scoring_alternative'].any())
        self.assertFalse(result['has_safe_higher_scoring_higher_EP_alternative'].any())

    def test_safe_alternative_is_evidence_not_edit_direction(self):
        result = candidate_evidence([feedback(ep=.2), feedback(ep=.8)])
        self.assertEqual(result['has_safe_higher_scoring_alternative'].tolist(), [True, False])
        self.assertTrue(result['has_safe_higher_scoring_higher_EP_alternative'][0])
        self.assertFalse(result['parsed'][0].available['verified_edit_direction'])

    def test_tiny_absolute_progress_gain_not_success(self):
        result = candidate_evidence([feedback(ep=.8), feedback(ep=.80001)])
        self.assertFalse(result['has_safe_higher_scoring_alternative'].any())

    def test_source_reward_cannot_replace_canonical_metric(self):
        value = feedback(ep=.1)
        value['metrics']['score'] = (10 * .1 + 5 + 2) / 17
        with self.assertRaises(ValueError):
            read_feedback(value)

    def test_input_not_overwritten_by_label_extraction(self):
        original = feedback(dac=0.)
        before = copy.deepcopy(original)
        read_feedback(original)
        np.testing.assert_array_equal(original['simulated_states'], before['simulated_states'])
        np.testing.assert_array_equal(original['requested_trajectory'], before['requested_trajectory'])


if __name__ == '__main__':
    unittest.main()
