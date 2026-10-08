import pytest
import torch
from starVLA.model.modules.structured_world.grid import GridSpec
from tools.structured_world.score_scene_fields import binary_counts, endpoint_time_counts, score_scene, summarize_counts


@pytest.mark.parametrize('steps', [6, 8])
def test_perfect_fields_unknown_labels_and_fixed_body_metrics(steps):
    grid = GridSpec(xmin=-10, xmax=10, ymin=-10, ymax=10)
    h, w = grid.height, grid.width
    occupancy = torch.zeros(steps+1, h, w, dtype=torch.uint8)
    occupancy[:, 22:24, 25:27] = 1
    occupancy[2:, 18:20, 26:28] = 1
    valid = torch.ones_like(occupancy, dtype=torch.bool)
    valid[:, :2] = False
    occupancy[~valid] = 255
    gt = torch.stack((torch.linspace(0, 3, steps), torch.zeros(steps), torch.zeros(steps)), 1)
    labels = {'occupancy': occupancy, 'occupancy_valid': valid,
        'road_distance': torch.full((h, w), 3.), 'road_valid': valid[0],
        'ego_physical': gt, 'future_valid': torch.ones(steps, dtype=torch.bool),
        'instances': torch.zeros_like(occupancy, dtype=torch.long)}
    p0, pt = (occupancy[0:1] == 1).float(), (occupancy[1:, None] == 1).float()
    prediction = {'road_distance_m': labels['road_distance'][None], 'p0': p0, 'pt': pt,
        'p_enter': (1-p0[None])*pt, 'p_release': p0[None]*(1-pt), 'q0': gt.clone(), 'q_final': gt.clone()}
    metadata = {'ego_body': {'length': 2., 'width': 1., 'rear_axle_to_center': 0.}}
    scores = score_scene(prediction, labels, metadata, grid)
    future = summarize_counts(scores['global_valid_cells/future_occupancy'])
    assert future['IoU'] == 1 and future['Brier'] == 0
    assert future['valid'] == int(valid[1:].sum())
    body = scores['fixed_logged_GT_body_relations']
    assert body['body_road_error_sum'] == 0 and body['body_occupancy_error_sum'] == 0
    assert body['registered_body_queries'] == 3*steps
    # OOR predictions remain in the complete trajectory count; fixed metric
    # queries do not become the particular model's easy proposal distribution.
    prediction['q0'][:, 0] = 100
    changed = score_scene(prediction, labels, metadata, grid)
    assert changed['fixed_logged_GT_body_relations'] == body
    assert changed['fixed_logged_GT_body_neighborhood/future_occupancy'] == scores['fixed_logged_GT_body_neighborhood/future_occupancy']
    assert changed['complete_deployment_trajectory_range']['q0_points'] == steps
    assert changed['complete_deployment_trajectory_range']['q0_out_of_range_points'] == steps


def test_endpoint_time_keeps_missed_events_and_never_calls_them_free():
    truth = torch.tensor([[[0, 0]], [[1, 1]], [[0, 0]], [[0, 0]]], dtype=torch.bool)
    probability = torch.tensor([[[0., 0.]], [[0., 0.]], [[1., 0.]], [[0., 0.]]])
    valid = torch.ones_like(truth)
    result = endpoint_time_counts(probability, truth, valid)
    assert result['GT_to_prediction_endpoint_time_error_sum'] == .5
    assert result['GT_to_prediction_endpoint_time_count'] == 1
    assert result['GT_to_prediction_endpoint_time_unmatched_positive_endpoints'] == 1
    assert result['prediction_to_GT_endpoint_time_error_sum'] == .5
    # No labels is not a perfect score or a fabricated positive.
    empty = summarize_counts(binary_counts(probability, truth, torch.zeros_like(valid)))
    assert empty['valid'] == 0 and empty['Brier'] is None and empty['recall'] is None
