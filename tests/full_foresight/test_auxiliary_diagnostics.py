import torch
import pytest
from tools.full_foresight.evaluate_auxiliary import spatial_statistics, summarize


def test_current_and_future_errors_keep_denominators_and_copy_reference():
    current=torch.zeros(3,4,2,3);future=torch.ones_like(current)
    valid=torch.ones(3,2,3,dtype=torch.bool);valid[1]=False
    future[1]=float('nan')
    result=spatial_statistics(current,future,current,valid)
    assert result[0]['squared_error']==result[0]['copy_current_squared_error']==24
    assert result[1]['elements']==0 and result[1]['squared_error']==0
    row={'token':'a','log':'l','failure':None,**{f'h1_v{i}':x for i,x in enumerate(result)}}
    total=summarize([row],1)
    assert total['visual']['h1_v0']['mse']==1 and total['visual']['h1_v0']['relative_improvement_over_copy']==0
    assert total['visual']['h0_v0']['elements']==0 and total['valid']
    failed={'token':'b','log':'m','failure':'missing target'}
    assert not summarize([row,failed],2)['valid']
    future[0,0,0,0]=float('nan')
    with pytest.raises(ValueError):spatial_statistics(current,future,current,valid)
