import copy
import json
import pytest
from tools.full_foresight.summarize_training import completed_rows, check_rows, TASKS


def rows():
    result = []
    for update in range(1, 1002):
        w = dict(ego_fm=1., current_dino=1., future_dino=.9 * min(update / 1000, 1), interaction=.8 * min(update / 1000, 1))
        result.append(dict(update=update, exposure=update * 32, counts=dict(ego_scenes=32, current_dino=96, future_dino=0, interaction=0),
                           horizon_scene_requests=[10, 11, 11], raw_losses=dict(ego_fm=1., current_dino=1., future_dino=0., interaction=0.),
                           losses=dict(ego_fm=1., current_dino=1., future_dino=0., interaction=0.), effective_weights=w))
    return result


def test_full_weight_and_legal_missing_auxiliary():
    assert check_rows(rows(), .9, .8) == 1001 * 32


@pytest.mark.parametrize('change', ['weight', 'exposure', 'nan', 'duplicated_step'])
def test_reject_actual_pipeline_changes(change):
    r = copy.deepcopy(rows())
    if change == 'weight':r[999]['effective_weights']['interaction'] = 0.
    if change == 'exposure':r[2]['horizon_scene_requests'] = [1, 1, 1]
    if change == 'nan':r[0]['raw_losses']['future_dino'] = float('nan')
    if change == 'duplicated_step':r[3]['update'] = 3
    with pytest.raises(ValueError):check_rows(r, .9, .8)


def test_live_append_partial_tail(tmp_path):
    p = tmp_path / 'rows.jsonl'
    p.write_text(json.dumps({'update': 1}) + '\n{"update":')
    assert completed_rows(p) == [{'update': 1}]
