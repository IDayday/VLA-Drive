from copy import deepcopy
from pathlib import Path
import pytest
from iqe.io import read_json, ContractError, file_hash
from iqe.training.il_only import derive_contract
from iqe.training.online_base import OnlineBasePolicy
from iqe.base_training import inputs, configuration

REPO = Path(__file__).resolve().parents[2]


def test_separate_objective_keeps_planner_input_and_normalizer(tmp_path):
    parent = read_json(REPO / 'reports/iqe/S0_CONTRACT.json')
    snapshot = deepcopy(parent)
    new = derive_contract(parent, tmp_path)
    assert parent == snapshot
    assert parent['framework_contract_hash'] != new['framework_contract_hash']
    for key in ('input', 'trajectory', 'normalizer_hash', 'query_architecture', 'query_model_code_hashes'):
        assert new[key] == parent[key]
    cfg = new['source_config']['foresight']
    assert cfg['ablation'] == cfg['arm'] == 'W_ACTION_ONLY'
    assert not any(cfg[k] for k in ('enable_current_dino', 'enable_future_dino', 'enable_interaction'))
    assert [cfg[k] for k in ('lambda_cur', 'lambda_fut', 'lambda_int')] == [0., 0., 0.]
    assert new['action']['retained_auxiliary_losses'] == []
    assert new['source_config']['framework'] == parent['source_config']['framework']
    assert derive_contract(parent, tmp_path) == new


def test_no_old_four_task_or_foreign_runtime_resume():
    policy = OnlineBasePolicy()
    state = {'learned_driving_weights_loaded': False, 'kind': 'iqe_query_base',
             'resume_boundary': 'optimizer_step', 'performance_identity': {'old': True}}
    with pytest.raises(ContractError, match='separate experiment'):
        policy.validate_state(state, {'identity_hash': 'new'})
    state['online_base_identity'] = {'identity_hash': 'new'}
    assert policy.validate_state(state, {'identity_hash': 'new'})['exact_resume']


@pytest.mark.parametrize('options', [dict(visual_precision='bf16_compute'), dict(query_training_semantics='native_repeat1'),
                                    dict(execution_weights='bf16_master')])
def test_public_runtime_does_not_silently_change_training(options):
    with pytest.raises(ContractError):
        OnlineBasePolicy(**options)


def test_all_protected_model_files_unchanged():
    contract = read_json(REPO / 'reports/iqe/S0_CONTRACT.json')
    for relative, checksum in contract['query_model_code_hashes'].items():
        assert file_hash(REPO / relative) == checksum
