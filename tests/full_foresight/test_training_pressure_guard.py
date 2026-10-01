import pytest
from tools.full_foresight.training_pressure_guard import eligible


@pytest.mark.parametrize('change,allowed',[
    ({},True),({'status':'PAUSED'},False),({'status':'COMPLETE'},False),
    ({'updated_unix':0},False),({'identity':'other'},False),
    ({'source_sha':'other'},False),({'host':'other'},False),
])
def test_only_fresh_registered_running_training_releases(change,allowed):
    state=dict(status='RUNNING',identity='run',source_sha='sha',host='authorized',updated_unix=99)
    state.update(change)
    model=dict(run_identity='run',training_source_sha='sha',hostname='authorized')
    assert eligible(state,model,100)==allowed
