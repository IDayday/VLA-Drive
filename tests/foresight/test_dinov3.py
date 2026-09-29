import copy
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
from starVLA.model.modules.foresight.config import ForesightConfig
from starVLA.model.modules.foresight.dino_feature_head import DINOFeatureHead
from starVLA.model.modules.foresight.dinov3_target_encoder import preprocess_image
from starVLA.model.modules.foresight.losses import masked_regression,request_future_horizons
from tools.foresight.student_state import optimizer_batch_counts


def targets(current=True):
    g=torch.Generator().manual_seed(5)
    data={'ego':torch.zeros(7,8,4),'future_dino':torch.randn(7,3,3,8,2,3,generator=g),
          'future_dino_valid':torch.ones(7,3,3,2,3,dtype=torch.bool)}
    data['future_dino_valid'][0]=False
    if current:data.update(current_dino=torch.randn(7,3,8,2,3,generator=g),current_dino_valid=torch.ones(7,3,2,3,dtype=torch.bool))
    return data


def test_current_addition_does_not_change_future_rng_or_missing_request():
    a,b=targets(False),targets(True)
    g1=torch.Generator().manual_seed(42);g2=torch.Generator().manual_seed(42)
    for _ in range(5):
        ca=optimizer_batch_counts(a,g1,'cpu');cb=optimizer_batch_counts(b,g2,'cpu')
        assert torch.equal(a['dino_horizon'],b['dino_horizon'])
        assert torch.equal(g1.get_state(),g2.get_state())
        assert ca['future_dino']==cb['future_dino'] and ca['ego_scenes']==cb['ego_scenes']==7
    g1=torch.Generator().manual_seed(1);g2=torch.Generator().manual_seed(1)
    a['future_dino_valid'][:]=False;b['future_dino_valid'][:]=True
    for _ in range(5):
        optimizer_batch_counts(a,g1,'cpu');optimizer_batch_counts(b,g2,'cpu')
        assert torch.equal(a['dino_horizon'],b['dino_horizon'])
    assert set(request_future_horizons(100,g1).tolist())=={0,1,2}


def test_current_future_normalize_independently_ignore_nan_and_backward():
    d=targets();d['current_dino_valid'][1:]=False
    counts=optimizer_batch_counts(d,torch.Generator().manual_seed(0),'cpu')
    assert counts['current_dino']==3*2*3*8
    pred=torch.ones_like(d['current_dino'],requires_grad=True)
    target=torch.zeros_like(pred);target[1:]=float('nan')
    pred.data[1:]=float('nan')
    loss,count=masked_regression(pred,target,d['current_dino_valid'][:,:,None],global_count=counts['current_dino'])
    loss.backward();assert loss==1 and torch.isfinite(pred.grad).all() and not pred.grad[1:].any()
    target[0,0,0,0,0]=float('nan')
    with pytest.raises(ValueError):masked_regression(pred,target,d['current_dino_valid'][:,:,None])


def test_physical_horizon_shared_head_no_answers_save_load(tmp_path):
    torch.manual_seed(5);head=DINOFeatureHead(24,8,dim=32,layers=2,heads=4).eval()
    w=torch.randn(2,64,24,requires_grad=True)
    current=head(w,torch.zeros(2),(2,5));future=head(w,torch.tensor([1.,4.]),(2,5))
    assert current.shape==(2,3,8,2,5)
    assert torch.equal(current,head(w,torch.zeros(2),(2,5)))
    assert not torch.equal(current,future)
    for prediction in (current,future):
        grad,=torch.autograd.grad(prediction.square().mean(),w,retain_graph=True)
        assert grad.abs().sum()>0 and torch.isfinite(grad).all()
    with pytest.raises(ValueError):head(w,torch.tensor([0.,3.]),(2,5))
    path=tmp_path/'head.pt';torch.save(head.state_dict(),path)
    restored=DINOFeatureHead(24,8,dim=32,layers=2,heads=4).eval()
    restored.load_state_dict(torch.load(path,weights_only=True),strict=True)
    assert torch.equal(current,restored(w,torch.zeros(2),(2,5)))
    # Unified task head is stateless across calls at other physical horizons.
    head(w,torch.tensor([2.,2.]),(2,5))
    assert torch.equal(future,head(w,torch.tensor([1.,4.]),(2,5)))


def test_rectangular_rgb_preprocessing_has_full_field_and_explicit_patch_mask(tmp_path):
    image=np.zeros((1080,1920,3),np.uint8);image[:,:,0]=255
    path=tmp_path/'current.png';Image.fromarray(image).save(path)
    pixels,valid,meta=preprocess_image(path)
    assert pixels.shape==(3,256,464) and valid.shape==(16,29)
    assert meta['crop_xyxy']==[0,0,1920,1080] and meta['resized_wh']==[455,256]
    assert valid[:,:28].all() and not valid[:,28].any()
    assert pixels[0,0,0].item()==pytest.approx((1-.485)/.229)
    assert not pixels[:,:,455:].any()
    again=preprocess_image(path);assert torch.equal(pixels,again[0]) and torch.equal(valid,again[1])
    wrong=tmp_path/'square.png';Image.new('RGB',(256,256)).save(wrong)
    with pytest.raises(ValueError):preprocess_image(wrong)


def test_explicit_arm_config_rejects_unmatched_flags_or_legacy_flux():
    ForesightConfig(arm='W_FULL',enable_current_dino=True,enable_future_dino=True,enable_interaction=True).validate()
    ForesightConfig(arm='W_ONLY').validate()
    ForesightConfig(arm='R_NATIVE',num_queries=0).validate()
    with pytest.raises(ValueError):ForesightConfig(arm='W_FULL',enable_future_dino=True).validate()
    with pytest.raises(ValueError):ForesightConfig(arm='W_CUR',enable_current_dino=True,lambda_vis=1).validate()
    with pytest.raises(ValueError):ForesightConfig(arm='W_FUT',enable_future_dino=True,future_horizons_s=(0,1,2)).validate()
