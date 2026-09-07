from types import SimpleNamespace
import pytest
from scripts.export_planreg_current_only_navtest import CurrentOnlyDataset


class Loader:
    tokens = ['scene_a']

    def __len__(self):
        return 1

    def get_agent_input_from_token(self, token):
        assert token == 'scene_a'
        return 'current_input'

    def get_scene_from_token(self, token):
        raise AssertionError('Future Scene annotations must not be constructed')


def test_export_dataset_reads_only_agent_input():
    def build(value):
        assert value == 'current_input'
        return {'status_feature': [1, 2]}
    dataset = CurrentOnlyDataset(Loader(), [SimpleNamespace(compute_features=build)])
    features, targets, token = dataset[0]
    assert features == {'status_feature': [1, 2], 'scenario_token': 'scene_a'}
    assert targets == {}
    assert token == 'scene_a'


@pytest.mark.parametrize('key', ['future_image_paths', 'target_registers'])
def test_export_dataset_rejects_future_feature_keys(key):
    dataset = CurrentOnlyDataset(Loader(), [SimpleNamespace(compute_features=lambda _: {key: 1})])
    with pytest.raises(RuntimeError, match='Future/target'):
        dataset[0]
