import numpy as np
import pytest
from iqe.robust_scorer.evaluation import score_diagnostics, predictions
from iqe.specialist_study.metrics import select


def test_diagnostics_preserve_cost_weight_and_role_boundary():
    truth = np.array([[.95, 0.], [.02, 0.]])
    pred = {'values': np.array([[.8, .2], [.2, .8]])}
    bank = {'scores': truth, 'valid': np.ones_like(truth, dtype=bool)}
    out = score_diagnostics(pred, bank)
    assert out['comparison_pairs'] == 2
    assert out['gap_weighted_pair_accuracy'] == pytest.approx(.95/(.95+.02))
    with pytest.raises(ValueError, match='no test access'):
        predictions(None, 'R00', 'final_test', 'cpu')


def test_diagnostic_invalid_labels_and_empty_pair_population():
    truth = np.array([[.5, .5], [.3, np.nan]])
    pred = {'values': np.array([[.5, .5], [.3, .9]])}
    bank = {'scores': truth, 'valid': np.array([[True, True], [True, False]])}
    out = score_diagnostics(pred, bank)
    assert out['valid_candidates'] == 3
    assert out['mse'] == 0
    assert out['comparison_pairs'] == 0
    assert out['gap_weighted_pair_accuracy'] is None
    assert sum(b['count'] for b in out['reliability_bins']) == 3
    assert out['candidate_columns'][1]['valid'] == 1
    assert out['candidate_columns'][1]['zeros'] == 0
    assert out['binned_mean_calibration_error'] == 0


def test_raw_argmax_stable_base_tie_and_margin_are_distinct():
    pred = np.array([[.8, .8, .8], [.79, .8, .8]])
    assert select(pred, expert_ids=['expert_0', 'expert_2', 'expert_1']).tolist() == [0, 2]
    assert select(pred, margin=.02, expert_ids=['expert_0', 'expert_2', 'expert_1']).tolist() == [0, 0]


def test_zero_failures_remain_in_per_candidate_diagnostics():
    bank = {'scores': np.array([[0., 1.], [1., 0.]]), 'valid': np.ones((2, 2), bool)}
    result = score_diagnostics({'values': np.array([[.99, .95], [.95, .5]])}, bank)
    assert result['candidate_columns'][0]['zeros'] == 1
    assert result['candidate_columns'][0]['mean_prediction_on_zero'] == .99
    assert result['candidate_columns'][1]['mean_prediction_on_zero'] == .5


def test_real_inference_comparison_rejects_wrong_selection_and_nonfinite():
    import torch
    from iqe.robust_scorer.qualification import compare_outputs
    expected = dict(trajectory=torch.zeros(1, 8, 3), candidates=torch.zeros(1, 4, 8, 3),
                    predicted_scores=torch.ones(1, 4), selected=torch.zeros(1, dtype=torch.long))
    actual = {k: v.clone() for k, v in expected.items()} | {'calls': {'visual': 1, 'language': 1}}
    assert all(value == 0 for value in compare_outputs(actual, expected).values())
    actual['selected'][0] = 1
    with pytest.raises(ValueError, match='selected candidate'):
        compare_outputs(actual, expected)
    actual['selected'][0] = 0
    actual['predicted_scores'][0, 0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        compare_outputs(actual, expected)


def test_finalizer_waits_for_every_terminal_result_and_live_lock(tmp_path):
    import fcntl
    import json
    from types import SimpleNamespace
    from scripts.iqe.finish_robust_scorer import ready
    from iqe.robust_scorer.config import CELLS
    config = SimpleNamespace(root=tmp_path, updates=6000)
    assert not ready(config)
    for cell in CELLS:
        folder = tmp_path/'scorers'/cell
        folder.mkdir(parents=True)
        (folder/'result.json').write_text(json.dumps(dict(status='COMPLETE', optimizer_steps=6000,
                                                        termination={'terminal': True})))
    assert ready(config)
    path = tmp_path/'scorers/R00/result.json'
    path.write_text(json.dumps(dict(status='STOPPED_AT_BOUNDARY', optimizer_steps=580, termination={'terminal': False})))
    assert not ready(config)
    path.write_text(json.dumps(dict(status='COMPLETE', optimizer_steps=6000, termination={'terminal': True})))
    with open(path.parent/'RUN.lock', 'a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not ready(config)
    assert ready(config)


@pytest.mark.parametrize('cell', ['R00', 'R11'])
def test_research_scorer_payload_roundtrip_and_external_rejection(cell):
    from dataclasses import asdict
    import torch
    from iqe.contracts import TrajectoryContract
    from iqe.robust_scorer.config import CELLS
    from iqe.robust_scorer.model import RobustScorer
    from iqe.robust_scorer.bundle import scorer_from_payload
    contract = TrajectoryContract(8, .5, 'normalized_xy_sincos', 'ego_relative', 'rear_axle',
                                  'radian', (10., .3, 0., 0.), (8., 2., 1., 1.), 'test')
    model = RobustScorer(8, 8, contract, **CELLS[cell]).eval()
    payload = dict(kind='iqe_robust_scorer_research_v1', scorer_cell=cell, scorer_spec=CELLS[cell],
                   contract={'trajectory': asdict(contract)}, scorer=model.state_dict(),
                   initialization={'external_scorer_weights_loaded': False, 'loaded_scorer_weights': []})
    restored = scorer_from_payload(payload)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    assert not any(p.requires_grad for p in restored.parameters())
    payload['initialization']['external_scorer_weights_loaded'] = True
    with pytest.raises(ValueError, match='external'):
        scorer_from_payload(payload)
