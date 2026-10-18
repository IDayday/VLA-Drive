import copy
import json
from dataclasses import replace
import numpy as np
import pytest
import torch
from starVLA.model.modules.decodable_st.contracts import AllowedObservation,SpatialContract,StudentRepresentation
from starVLA.model.modules.decodable_st.attention_masks import world_attention_mask,append_mrope_positions
from starVLA.model.modules.decodable_st.frozen_decoders import sgdrive_logits_layout
from starVLA.model.modules.decodable_st.coordinates import transform_points,future_lidar_to_current,match_timestamp,ray_evidence
from starVLA.model.modules.decodable_st.student_queries import QueryBank
from starVLA.model.modules.decodable_st.vigt_readout import SpatialReadout
from starVLA.model.modules.decodable_st.world_to_action_bridge import WorldToActionBridge
from starVLA.model.modules.decodable_st.losses import scene_mean,occupancy_loss


def test_observation_rejects_targets_and_identifiers():
    value=dict(images=((None,None,None),),navigation=("straight",),ego_state=torch.zeros(1,1,4),camera_ids=("F","L","R"))
    AllowedObservation.from_mapping(value).validate(("F","L","R"))
    for forbidden in ("teacher_cache","future_images","lidar","sample_token","log_id","gt_action"):
        with pytest.raises(ValueError):AllowedObservation.from_mapping({**value,forbidden:None})


@pytest.mark.parametrize("valid", [[[0,0,1,1,1],[1,1,1,1,1]],[[1,1,1,0,0],[1,1,1,1,1]]])
def test_mask_padding_groups_and_mrope(valid):
    valid=torch.tensor(valid);action=torch.tensor([[2],[3]])
    mask=world_attention_mask(valid,action,2,3,torch.float32)[:,0]
    assert (mask[:,:5,5:]<0).all()
    assert (mask[:,5:7,7:]<0).all()
    for b in range(2):assert (mask[b,5:,action[b]]<0).all()
    assert mask[0,7,5]==0 and mask[1,9,8]==0
    pos=torch.arange(5)[None,None].expand(3,2,5)
    out=append_mrope_positions(pos,valid,5)
    assert torch.equal(out[:,:,:5],pos)
    assert (out[:,:,6:]-out[:,:,5:-1]==1).all()


def test_query_initialization_preserves_global_rng():
    torch.manual_seed(92);before=torch.random.get_rng_state().clone()
    bank=QueryBank(32,(4,4),42)
    assert torch.equal(torch.random.get_rng_state(),before)
    bank().sum().backward();assert bank.queries.grad is not None


def test_native_sgdrive_axis_layout():
    raw=torch.arange(2*16*3*5).reshape(1,32,3,5)
    out=sgdrive_logits_layout(raw)
    for c in range(2):
        for z in range(16):
            assert torch.equal(out[0,c,:,:,z],raw[0,c*16+z])


def test_ego_compensation_and_timestamps():
    t0=np.eye(4);t1=np.eye(4);t1[0,3]=10
    np.testing.assert_allclose(future_lidar_to_current(t0,t1,np.eye(4),np.array([[5,0,1]])),[[15,0,1]])
    assert match_timestamp(1000000,[1500000,2001000],1.,.01)["actual_horizon_s"]==1.001
    assert not match_timestamp(1000000,[1500000],1.,.01)["valid"]


def test_ray_unknown_is_not_free_and_roi_is_explicit():
    q,y,v=ray_evidence(np.array([[0,0,0]]),np.array([[10,0,0]]))
    assert y.tolist()==[1,0,0,0] and (q[:,0]<=10).all()
    c=SpatialContract(8,4,4,(-1,-2,-1),(20,2,3))
    assert c.in_roi(torch.tensor([[[0.,0.,0.],[20.,0.,0.],[float('nan'),0,0]]])).tolist()==[[True,False,False]]


def test_zero_reader_and_actual_student_memory_gradient():
    contract=SpatialContract(8,16,16,(-1,-2,-1),(20,2,3))
    head=SpatialReadout(32,contract,(4,4));bridge=WorldToActionBridge(32,contract)
    hidden=torch.randn(2,16,32,requires_grad=True);action=torch.randn(2,8,32,requires_grad=True)
    rep=StudentRepresentation(action,hidden,predicted_spatial_context=head(hidden))
    torch.testing.assert_close(bridge(rep),action,rtol=0,atol=0)
    bridge.spatial_reader.output.weight.data.normal_(0,.01)
    bridge(rep).square().mean().backward()
    assert hidden.grad.abs().sum()>0 and action.grad.abs().sum()>0


def test_scene_normalization_missing_tasks_and_empty():
    error=torch.tensor([[1.,9.],[4.,4.]],requires_grad=True);valid=torch.tensor([[True,False],[True,True]])
    loss,n=scene_mean(error,valid);assert n==2;torch.testing.assert_close(loss,torch.tensor(2.5))
    loss.backward();torch.testing.assert_close(error.grad,torch.tensor([[.5,0],[.25,.25]]))
    logits=torch.randn(2,3,requires_grad=True)
    empty,n=occupancy_loss(logits,torch.full((2,3),-1),torch.zeros(2,3,dtype=torch.bool))
    empty.backward();assert empty==0 and logits.grad.eq(0).all() and n==0
