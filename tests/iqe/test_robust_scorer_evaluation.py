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


def test_raw_argmax_stable_base_tie_and_margin_are_distinct():
    pred = np.array([[.8, .8, .8], [.79, .8, .8]])
    assert select(pred, expert_ids=['expert_0', 'expert_2', 'expert_1']).tolist() == [0, 2]
    assert select(pred, margin=.02, expert_ids=['expert_0', 'expert_2', 'expert_1']).tolist() == [0, 0]


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
