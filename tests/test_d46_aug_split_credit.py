import copy
import numpy as np
import pytest
import torch
from starVLA.rl.d46_augustus.config import defaults
from starVLA.rl.d46_augustus.runtime import AugustusExtension
from starVLA.rl.d46_augustus.split_credit import normalize_centered,reverse_travel,action_reward
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec


def test_released_population_std_and_hard_small_variance_gate():
    values=np.array([[0.,1.,2.,3.],[1.,1.,1.,1.],[0.,0.,0.,1e-7]],dtype=np.float64)
    raw=torch.from_numpy(values-values.mean(1,keepdims=True))
    result=normalize_centered(raw)
    np.testing.assert_allclose(result[0],[-1.3416407865,-.4472135955,.4472135955,1.3416407865],rtol=1e-6)
    assert result[1:].count_nonzero()==0
    assert not torch.allclose(result[0],raw[0].float()/raw[0].std(unbiased=True).float())


def test_release_gate_counts_first_waypoint_and_preserves_official_scores():
    physical=torch.zeros(4,8,3)
    physical[0,:,0]=torch.arange(1,9)
    physical[1,:,0]=-1.  # Starts behind ego, then constant: still reverse.
    physical[2,:,0]=torch.arange(1,9);physical[2,2,0]=1.5
    physical[3,:,0]=-.005  # Native 1 cm dead-zone.
    np.testing.assert_allclose(reverse_travel(physical).numpy(),[0.,1.,.5,0.])
    chain=torch.zeros(4,4,8,3);chain[:,-1]=ActionCodec.norm_odo(physical)
    scene=dict(chain=chain,records=[{'metrics':{'score':.9}}for _ in range(4)])
    records=copy.deepcopy(scene['records'])
    torch.testing.assert_close(action_reward(scene),torch.tensor([4.5,0.,0.,4.5],dtype=torch.float64))
    assert scene['records']==records


def test_split_recipe_reuses_deployment_and_mask_without_batch_rescaling():
    old=defaults('AUG7_NATIVE_FP32');cfg=defaults('AUG9_SPLIT_ACTION')
    assert old['beta']==cfg['beta']==.1
    for key in ['training_sigma_grid','deterministic_prefix_grid','noise_scale','scope','text_action_precision']:
        assert old['augustus'][key]==cfg['augustus'][key]
    ext=AugustusExtension();ext.cfg=cfg
    raw=torch.stack([torch.linspace(-.01,.01,16),torch.linspace(-1,1,16),torch.linspace(-100,100,16)]).double()
    normalized,_=ext.advantages(raw,torch.tensor([True,True,False]),16,1)
    torch.testing.assert_close(normalized[0],normalized[1])
    assert normalized[2].count_nonzero()==0
