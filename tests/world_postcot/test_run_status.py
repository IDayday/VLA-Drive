import copy

import pytest

from tools.qwen_world_postcot.update_status import completed_scores


def test_real_controller_final_schema_and_no_partial_or_qualification_score():
    assert completed_scores(None) is None
    assert completed_scores({'status':'RUNNING'}) is None
    result=dict(status='COMPLETE',cohort='NAVTEST',qualification=False,seed=2101,PDMS=50.)
    campaign=dict(status='COMPLETE',primary_seed2101=dict(status='COMPLETE',primary_sampling_seed=2101,
        IL=result,RL=dict(result,PDMS=60.)))
    assert completed_scores(campaign)==dict(IL=50.,RL=60.)
    for replacement in ({'status':'RUNNING'},{'qualification':True},{'seed':2102},
                        {'cohort':'DEV'},{'PDMS':float('nan')}):
        bad=copy.deepcopy(campaign);bad['primary_seed2101']['RL'].update(replacement)
        with pytest.raises(ValueError,match='Incomplete/diagnostic'):
            completed_scores(bad)
