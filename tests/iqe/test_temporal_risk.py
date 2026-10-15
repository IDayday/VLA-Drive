import numpy as np
import pytest
import torch
from iqe.io import ContractError
from iqe.temporal_risk.labels import event_labels
from iqe.temporal_risk.objectives import terms, denominators


def labels(collision, offroad, ttc, nc=1., dac=1., tc=1.):
    return event_labels(collision, offroad, ttc, simulator_dt=.1, horizon=8, dt=.5,
        components=dict(no_at_fault_collisions=nc, drivable_area_compliance=dac,
                        time_to_collision_within_bound=tc))


def test_actual_first_events_and_nc_soft_failure():
    area = np.zeros(41, bool); area[11] = True
    out = labels(5, area, 0, nc=.5, dac=0., tc=0.)
    y = np.asarray(out['cumulative_event'])
    assert out['first_event_s'] == [0.5, 1.1, 0.]
    assert y[:, 0].all() and y[:, 2].all()
    assert y[:2, 1].sum() == 0 and y[2:, 1].all()


def test_no_event_is_not_fabricated_as_final_failure():
    out = labels(np.inf, np.zeros(41, bool), np.inf)
    assert out['first_event_s'] == [None]*3
    assert not np.asarray(out['cumulative_event']).any()
    with pytest.raises(ContractError, match='disagree'):
        labels(np.inf, np.zeros(41, bool), np.inf, nc=0.)
    with pytest.raises(ContractError, match='out-of-range'):
        labels(41, np.zeros(41, bool), np.inf, nc=0.)


def test_missing_trace_and_invalid_horizon_rejected():
    with pytest.raises(ContractError, match='horizon'):
        labels(np.inf, np.zeros(40, bool), np.inf)
    with pytest.raises(ContractError, match='invalid event'):
        labels(np.nan, np.zeros(41, bool), np.inf)


def test_event_loss_scene_normalization_and_masked_gradients():
    b, k, t = 3, 2, 8
    batch = dict(valid=torch.ones(b, k, dtype=torch.bool), scores=torch.ones(b, k),
        components=torch.ones(b, k, 6), component_valid=torch.ones(b, k, 6, dtype=torch.bool),
        natural=torch.tensor([True, True, False]), comparison=torch.zeros(b, dtype=torch.bool),
        events=torch.zeros(b, k, t, 3), event_valid=torch.ones(b, k, t, 3, dtype=torch.bool))
    batch['event_valid'][1, 1] = False
    prediction = dict(values=torch.full((b, k), .5, requires_grad=True),
        component_logits=torch.zeros(b, k, 6, requires_grad=True),
        event_logits=torch.zeros(b, k, t, 3, requires_grad=True))
    loss = terms(prediction, batch)['events']
    assert loss.denominator == 2 and denominators(batch)['events'] == 2
    (loss.numerator/loss.denominator).backward()
    g = prediction['event_logits'].grad
    assert g[0].sum() == g[1].sum()
    assert g[2].abs().sum() == 0 and g[1, 1].abs().sum() == 0
    assert g[0].sum() > 0


def test_common_initial_weights_output_and_single_decoder_are_preserved():
    from test_geometric_readout import scorer_example
    from iqe.temporal_risk.model import TemporalRiskScorer
    control, context, trajectory, contract = scorer_example()
    torch.manual_seed(17)
    model = TemporalRiskScorer(12,12,contract,visual_dim=12,event_init_seed=91).eval()
    for name, tensor in control.state_dict().items():
        assert torch.equal(tensor, model.state_dict()[name]), name
    calls=[]
    hook=model.decoder.register_forward_hook(lambda *unused:calls.append(1))
    expected=control(context,trajectory)
    result=model(context,trajectory)
    assert len(calls)==1
    for key in ('values','component_logits','valid'):
        assert torch.equal(expected[key],result[key])
    reordered=model(context,trajectory[:,[1,0]])
    torch.testing.assert_close(result['event_logits'],reordered['event_logits'][:,[1,0]])
    (result['event_logits'].square().mean()+result['values'].mean()).backward()
    assert model.event_head.weight.grad.abs().sum()>0
    assert model.decoder.layers[0].linear1.weight.grad.abs().sum()>0
    for key in ('world','action','current_ego','visual'):
        assert context[key].grad is None
    hook.remove()
    with pytest.raises(ContractError,match='whitelist'):
        model(context|{'events':torch.zeros(1,2,8,3)},trajectory)


def test_event_microbatch_sums_match_and_empty_mask_stays_graph_connected():
    b,k,t=2,2,8
    batch=dict(valid=torch.ones(b,k,dtype=torch.bool),scores=torch.ones(b,k),
        components=torch.ones(b,k,6),component_valid=torch.ones(b,k,6,dtype=torch.bool),
        natural=torch.tensor([True,True]),comparison=torch.zeros(b,dtype=torch.bool),
        events=torch.ones(b,k,t,3),event_valid=torch.ones(b,k,t,3,dtype=torch.bool))
    batch['event_valid'][1]=False
    pred=dict(values=torch.full((b,k),.5),component_logits=torch.zeros(b,k,6),
              event_logits=torch.zeros(b,k,t,3,requires_grad=True))
    full=terms(pred,batch)['events']
    sums=[terms({key:v[i:i+1] for key,v in pred.items()},
                {key:v[i:i+1] for key,v in batch.items()})['events'] for i in range(b)]
    assert sum(x.denominator for x in sums)==full.denominator==1
    torch.testing.assert_close(sum(x.numerator for x in sums),full.numerator)
    sums[1].numerator.backward()
    assert pred['event_logits'].grad is not None and pred['event_logits'].grad.abs().sum()==0
