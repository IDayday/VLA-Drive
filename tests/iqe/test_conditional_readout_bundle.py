from dataclasses import asdict
from pathlib import Path
import pytest
import torch
from iqe.io import file_hash
from iqe.conditional_readout.bundle import scorer_from_payload
from iqe.conditional_readout import model as source
from test_conditional_readout import model, inputs, contract


def test_scorer_payload_strict_roundtrip_and_no_external_weight_source():
    # Match inference flags on both sides; requires_grad changes MHA fastpath
    # dispatch even inside no_grad, so a trainable source is not this contract.
    m=model().eval().requires_grad_(False);x,t=inputs()
    payload=dict(kind='iqe_conditioned_readout_research_v1',conditioned=True,
        source_hash=file_hash(source.__file__),scorer=m.state_dict(),
        initialization=dict(external_scorer_weights_loaded=False,loaded_scorer_weights=[]),
        contract=dict(trajectory=asdict(contract()),input=dict(views=['front','left','right'])))
    restored=scorer_from_payload(payload)
    with torch.no_grad():
        assert torch.equal(m(x,t)['values'],restored(x,t)['values'])
    with pytest.raises(ValueError,match='wrong readout'):
        scorer_from_payload(payload|dict(conditioned=False))
    with pytest.raises(ValueError,match='implementation changed'):
        scorer_from_payload(payload|dict(source_hash='wrong'))
    with pytest.raises(ValueError,match='external'):
        scorer_from_payload(payload|dict(initialization=dict(external_scorer_weights_loaded=True,loaded_scorer_weights=['foreign'])))
    with pytest.raises(RuntimeError):
        scorer_from_payload(payload|dict(scorer={k:v for k,v in m.state_dict().items() if not k.startswith('trajectory_encoder')}))
