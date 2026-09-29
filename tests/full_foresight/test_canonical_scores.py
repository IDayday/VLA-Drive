import csv
import pytest
from tools.full_foresight.canonical_scores import runtime_effect
from tools.local_interaction_mask_v2.compare_pdms import METRICS


def table(path,change=None):
    path.mkdir()
    rows=[]
    for i in range(2):
        row={'token':str(i),'log':'same_log','proposal_sha256':'trajectory'+str(i),
             'metric_cache_sha256':'cache'+str(i),'status':'ok',**{k:1. for k in METRICS}}
        if i==1 and change:row.update(change)
        rows.append(row)
    with (path/'scenes.csv').open('w') as stream:
        writer=csv.DictWriter(stream,list(rows[0]));writer.writeheader();writer.writerows(rows)


def test_same_inputs_runtime_metric_change_remains_visible(tmp_path):
    table(tmp_path/'a');table(tmp_path/'b',{'score':.75,'drivable_area_compliance':0.})
    result=runtime_effect(tmp_path/'a',tmp_path/'b')
    assert result['scenes']==2 and result['same_prediction_and_cache']
    assert result['metrics']['score']['mean_difference']==-.125
    assert result['metrics']['drivable_area_compliance']['changed_scenes_above_1e_12']==1


@pytest.mark.parametrize('change',[{'proposal_sha256':'different_prediction'},
                                  {'metric_cache_sha256':'different_cache'},
                                  {'log':'different_log'},{'status':'failed'}])
def test_input_or_failure_change_cannot_be_called_runtime_parity(tmp_path,change):
    table(tmp_path/'a');table(tmp_path/'b',change)
    with pytest.raises(ValueError):runtime_effect(tmp_path/'a',tmp_path/'b')


def test_unchanged_results_are_exact_parity(tmp_path):
    table(tmp_path/'a');table(tmp_path/'b')
    result=runtime_effect(tmp_path/'a',tmp_path/'b')
    assert all(v['max_abs_difference']==0 for v in result['metrics'].values())
