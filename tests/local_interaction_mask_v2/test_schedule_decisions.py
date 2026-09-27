import json
import pytest
from tools.local_interaction_mask_v2.convergence import planner_extension


def planner_runs(tmp_path):
    paths=[]
    common=dict(code_sha='source',current_identity='current',labels='train',holdout_labels='holdout',graph_config={},
                foundation_sha256='public',trainable_parameters=100,frozen_DiT_parameters=200)
    for mode in ('current','all','mask'):
        path=tmp_path/mode;path.mkdir();paths.append(path)
        (path/'manifest.json').write_text(json.dumps(dict(common,arguments=dict(mode=mode,batch=32,seed=42,schedule_epochs=16))))
        (path/'status.json').write_text(json.dumps(dict(status='complete',epochs=8,step=80,presentations=2560)))
        for epoch,value in [(6,5.),(8,4.99)]:
            (path/f'holdout_{epoch}.json').write_text(json.dumps(dict(ego_ADE_m=value,scenes=64,failed=0)))
    return paths


def test_planner_extension_is_joint_and_not_a_pdms_selection(tmp_path):
    paths=planner_runs(tmp_path)
    assert planner_extension(paths)['decision']=='STOP_ALL_THREE_AT_8'
    (paths[2]/'holdout_8.json').write_text(json.dumps(dict(ego_ADE_m=4.7,scenes=64,failed=0)))
    result=planner_extension(paths)
    assert result['decision']=='EXTEND_ALL_THREE_TO_16'
    assert result['uses_PDMS'] is False


def test_planner_extension_rejects_unequal_exposure(tmp_path):
    paths=planner_runs(tmp_path)
    (paths[0]/'status.json').write_text(json.dumps(dict(status='complete',epochs=8,step=80,presentations=2559)))
    with pytest.raises(ValueError,match='unequal'):planner_extension(paths)
