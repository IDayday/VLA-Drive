import torch
import pytest
from tools.action_video_foresight.visualization_metrics import (
    clean_features, reference_metrics, affinity_map, select_representatives,
    projected_vehicle_centres, relative_gain)
from tools.full_foresight.evaluate_auxiliary import FixedPositionMoments


def test_padding_nan_and_effective_denominator():
    target=torch.tensor([[[1.,2.],[float('nan'),float('nan')]]])
    pred=torch.tensor([[[3.,4.],[float('nan'),1e30]]])
    mask=torch.tensor([[True,False]])
    r=reference_metrics(pred,target,mask)
    assert r['patches']==1 and r['mse']==4
    assert torch.isfinite(clean_features(target,mask,True)).all()
    with pytest.raises(ValueError):clean_features(target,torch.ones_like(mask))


def test_fixed_coordinate_variance_rejects_rich_template_claim():
    value=torch.arange(24).reshape(2,3,4).float()
    m=FixedPositionMoments()
    for _ in range(5):m.add(value,torch.ones(2,3,1,dtype=torch.bool))
    assert m.result()==0
    assert value.var()>0  # Pooled spatial/channel variance is misleading here.


def test_selection_does_not_follow_model_error():
    rows=[dict(token=str(i),log=str(i%4),navigation=i%3,ego_motion='moving',
               peer_motion='moving_peer' if i%2 else 'no_valid_peer',clip_valid=True) for i in range(32)]
    first=[r['token'] for r in select_representatives(rows,12)]
    for i,r in enumerate(rows):r['model_error']=1000-i
    assert first==[r['token'] for r in select_representatives(list(reversed(rows)),12)]


def test_common_anchor_affinity_and_calibration():
    feat=torch.eye(3).reshape(1,3,3)
    assert affinity_map(feat,feat[0,1]).argmax()==1
    # Identity camera: z is depth; intrinsic principal point at image centre.
    record={'current_calibration':dict(intrinsics=[[[100,0,512],[0,100,288],[0,0,1]]],
            extrinsics=[torch.eye(4).tolist()],distortion=[[0,0,0,0,0]])}
    current=torch.zeros(3,8);current[1,2]=10;current[2,2]=-10
    anchors=projected_vehicle_centres(record,current.numpy(),[True,True,True])
    assert len(anchors)==1 and anchors[0]['row']==3 and anchors[0]['column']==4
    assert relative_gain(1,0) is None


def test_summary_keeps_current_future_change_groups_separate():
    from tools.action_video_foresight.summarize_visualization import task_score, clustered_difference
    rows=[{'log':str(i%2),'scores':{
        'current/v0/model':dict(patches=1,squared_channel_mean_sum=2),
        'current/v0/train_mean':dict(patches=1,squared_channel_mean_sum=4),
        'future/v0/t0/model':dict(patches=3,squared_channel_mean_sum=9),
        'future/v0/t0/static':dict(patches=3,squared_channel_mean_sum=12),
        'future/v0/t0/high_change/model':dict(patches=1,squared_channel_mean_sum=8)}} for i in range(4)]
    assert task_score(rows[0],'current','model')==2
    assert task_score(rows[0],'future','model')==3
    assert task_score(rows[0],'future','model','high_change')==8
    result=clustered_difference(rows,'future','static',100)
    assert result['scenes']==4 and result['log_cluster_95']==[-1,-1]
