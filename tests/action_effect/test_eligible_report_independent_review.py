"""Independent reporting admission negatives; all fixtures are synthetic."""
from copy import deepcopy

import pytest
import torch

from tools.action_effect import compare_eligible_effect_models as derived
from tests.action_effect.test_derived_eligible_report import fixtures,model_report


@pytest.mark.parametrize('field',['candidate_id','kind'])
def test_prediction_cannot_borrow_another_native_candidate_identity(monkeypatch,field):
    original,data,records=fixtures(monkeypatch)
    records[0][field]='not-the-native-'+field
    with pytest.raises(ValueError,match='exact native pair'):
        derived.admit_population(original,data,records)


def test_invalid_score_scene_cannot_be_reclassified_as_structural_no_intervention(monkeypatch):
    original,data,records=fixtures(monkeypatch)
    # b still has its native pair and locked prediction. No valid score does not
    # make it a source-defined no-pair scene and cannot earn a zero extension.
    for key in ('before_valid','after_valid','valid'):records[-1][key][0]=False
    with pytest.raises(ValueError,match='valid scores'):
        derived.admit_population(original,data,records)


@pytest.mark.parametrize('field',['scene_sampler_sha256','stateful_names','optimizer_step'])
def test_same_declared_recipe_cannot_hide_different_actual_streams(monkeypatch,field):
    reports={mode:model_report(monkeypatch,mode) for mode in derived.MODES}
    value=reports['terminal_pdms']['actual_sampling_proof'][field]
    reports['terminal_pdms']['actual_sampling_proof'][field]=value+['foreign'] if isinstance(value,list) else value+1 if isinstance(value,int) else 'foreign-state'
    with pytest.raises(ValueError,match='sampler recipe'):
        derived.match_controls(reports,step=1000)


@pytest.mark.parametrize('mutation',['absent_rng','float_rng','matrix_rng','extra_rng','boolean_consumption','zero_step'])
def test_sampling_proof_rejects_absent_or_wrong_checkpoint_contract(mutation):
    state={'sampler':torch.Generator().manual_seed(143).get_state(),
        'stateful':{'pair_rng':torch.Generator().manual_seed(244).get_state()},
        'consumed_scenes':32000,'optimizer_step':1000}
    if mutation=='absent_rng':state['stateful']={}
    elif mutation=='float_rng':state['stateful']['pair_rng']=state['stateful']['pair_rng'].float()
    elif mutation=='matrix_rng':state['sampler']=state['sampler'][None]
    elif mutation=='extra_rng':state['stateful']['foreign_rng']=state['stateful']['pair_rng']
    elif mutation=='boolean_consumption':state['consumed_scenes']=True
    else:state['optimizer_step']=0
    with pytest.raises(ValueError):derived.sampling_proof(state)
