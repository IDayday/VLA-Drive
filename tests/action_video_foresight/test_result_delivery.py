import copy
import math

import pytest

from tools.action_video_foresight.summarize_c_s_results import endpoint_ego


@pytest.mark.parametrize('arm,field',[('C1','ego_fit'),('S2','ego')])
def test_ego_endpoint_uses_recorded_schema_and_rejects_wrong_identity(arm,field):
    cp={'completed':100000,'sha256':'exact-scored-checkpoint'}
    result=dict(status='COMPLETE',arm=arm,update=100000,checkpoint=cp,
                **{field:dict(valid=True,failed=0,scenes=12146,ADE=.8,FDE=1.8,yaw_MAE_rad=.02)})
    assert endpoint_ego(result,arm,cp)['ADE_m']==.8
    for bad_cp in [dict(cp,sha256='another-model'),dict(cp,completed=90000)]:
        with pytest.raises(ValueError):endpoint_ego(result,arm,bad_cp)
    for invalid in [dict(valid=False),dict(failed=1),dict(scenes=12145),dict(ADE=math.nan),dict(FDE=-1.)]:
        bad=copy.deepcopy(result);bad[field].update(invalid)
        with pytest.raises(ValueError):endpoint_ego(bad,arm,cp)
