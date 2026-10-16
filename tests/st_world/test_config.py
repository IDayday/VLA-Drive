from copy import deepcopy
import pytest
from starVLA.dataloader.st_world_schema import TeacherFeatureContract
from starVLA.model.modules.st_world.config import SpaceTimeConfig, SPATIAL_CELLS, TEMPORAL_CELLS


def registered(cell='S-P2', parent=None):
    spec = SPATIAL_CELLS[parent or cell]
    height, width = spec['image_hw']
    pool = spec['pool']
    result = dict(cell=cell, world_queries_per_view=spec['world'], sampled_positions=spec['positions'],
        teacher=dict(image_hw=[height, width], patch_hw=[16, 16],
            native_grid_hw=[height // 16, width // 16],
            output_grid_hw=[height // 16 // pool, width // 16 // pool],
            channels=256, compute_dtype='float32', storage_dtype='float16',
            view_names=['CAM_F0', 'CAM_L0', 'CAM_R0'], teacher_identity='a' * 64,
            transform_identity='b' * 64,
            pooling=dict(kind='none' if pool == 1 else 'avg', kernel_hw=[pool, pool], stride_hw=[pool, pool])))
    if parent is not None:
        history, future, delta = TEMPORAL_CELLS[cell]
        result.update(spatial_parent=parent, history=history, future=future, delta=delta,
                      future_weight=.2 if future else 0., delta_weight=.05 if delta else 0.)
    return result


@pytest.mark.parametrize('cell', list(SPATIAL_CELLS))
def test_registered_spatial_cells_have_independent_output_grid(cell):
    config = SpaceTimeConfig.from_dict(registered(cell))
    assert isinstance(config.teacher, TeacherFeatureContract)
    assert config.world_queries == (288 if cell == 'S-P3' else 144)
    assert config.teacher.output_grid_hw == ((6, 8) if cell in ('S-P0', 'S-P1') else (18, 24))
    assert config.teacher.channels == 256  # Read from contract, never silently assume 1024.
    assert SpaceTimeConfig.from_dict({**registered(cell), 'teacher': config.teacher}) == config


@pytest.mark.parametrize('cell', list(TEMPORAL_CELLS))
def test_temporal_objectives_and_parent_are_explicit(cell):
    config = SpaceTimeConfig.from_dict(registered(cell, 'S-P2'))
    assert (config.history, config.future, config.delta) == TEMPORAL_CELLS[cell]
    assert config.current and config.current_weight == 1


@pytest.mark.parametrize('patch', [
    {'cell': 'SP2'}, {'private_default': True}, {'sampled_positions': 48},
    {'world_queries_per_view': 96}, {'current': False}, {'future': True, 'future_weight': .2},
    {'current_weight': 2}, {'current_weight': float('nan')}, {'future_weight': .2},
    {'history': 'false'}, {'readout_dim': 257}, {'sampled_positions': True},
    {'future_times_s': [.5, 1., 2., 8.]}, {'spatial_parent': 'S-P1'},
    {'future_times_s': None}, {'cell': []},
])
def test_unregistered_or_unresolved_configuration_is_rejected(patch):
    with pytest.raises(ValueError):
        SpaceTimeConfig.from_dict({**registered(), **patch})


@pytest.mark.parametrize('patch', [
    {'output_grid_hw': [9, 12]}, {'channels': 0}, {'compute_dtype': 'bfloat16'},
    {'storage_dtype': 'bfloat16'}, {'feature_layer': 'cls'}, {'unknown': 1},
    {'view_names': ['CAM_L0', 'CAM_F0', 'CAM_R0']}, {'teacher_identity': 'path-not-a-hash'},
])
def test_teacher_contract_cannot_silently_change(patch):
    value = registered()
    value['teacher'].update(patch)
    with pytest.raises(ValueError):
        SpaceTimeConfig.from_dict(value)


def test_temporal_weight_must_be_calibrated_explicitly():
    value = registered('T4', 'S-P3')
    value.pop('delta_weight')
    with pytest.raises(ValueError, match='delta objective'):
        SpaceTimeConfig.from_dict(value)
    value = registered('T2', 'S-P2')
    value['history'] = True
    with pytest.raises(ValueError, match='objectives disagree'):
        SpaceTimeConfig.from_dict(value)


def test_resolved_contract_roundtrip_has_stable_identity_and_no_hidden_defaults():
    value = registered()
    original = deepcopy(value)
    config = SpaceTimeConfig.from_dict(value)
    assert value == original
    assert TeacherFeatureContract.from_dict(config.teacher.to_dict()).identity == config.teacher.identity
    assert SpaceTimeConfig.from_dict(config.to_dict()) == config
    value.pop('teacher')
    with pytest.raises(ValueError, match='Incomplete'):
        SpaceTimeConfig.from_dict(value)
