import numpy as np
import pytest
import torch
from iqe.geometric_readout.geometry import (project_points, candidate_anchors,
                                           neighborhood_indices, gather_neighborhood)


def calibration():
    k = torch.tensor([[[[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]]]], dtype=torch.float64)
    ext = torch.eye(4, dtype=torch.float64)[None, None]
    size = torch.tensor([[[100., 100.]]], dtype=torch.float64)
    return k, ext, size


def test_official_pinhole_inverse_transform_and_visibility():
    from navsim.visualization.camera import _transform_pcs_to_images
    k, e, size = calibration()
    e[..., :3, 3] = torch.tensor([2., -1., .5])
    points = torch.tensor([[[[2., -1., 5.], [4., -1., 5.], [0., 0., -1.], [30., 0., 5.]]]], dtype=torch.float64)
    uv, valid = project_points(points, k, e, size)
    expected, mask = _transform_pcs_to_images(points[0, 0].numpy().T, e[0, 0, :3, :3].numpy(),
                                             e[0, 0, :3, 3].numpy(), k[0, 0].numpy(), img_shape=(100, 100))
    np.testing.assert_allclose(uv[0, 0, 0].numpy(), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(valid[0, 0, 0].numpy(), mask)
    assert valid.sum() == 2


def test_brown_distortion_requires_explicit_model_and_matches_opencv():
    import cv2
    k, e, size = calibration()
    points = torch.tensor([[[[.2, .1, 2.], [.5, -.2, 3.]]]], dtype=torch.float64)
    d = torch.tensor([[[-.3, .1, .001, .002, -.01]]], dtype=torch.float64)
    pinhole, _ = project_points(points, k, e, size)
    uv, _ = project_points(points, k, e, size, camera_model="brown_conrady", distortion=d)
    expected, _ = cv2.projectPoints(points[0, 0].numpy(), np.zeros(3), np.zeros(3), k[0, 0].numpy(), d[0, 0].numpy())
    np.testing.assert_allclose(uv[0, 0, 0].numpy(), expected[:, 0], rtol=1e-12, atol=1e-12)
    assert not torch.equal(pinhole, uv)
    with pytest.raises(ValueError):
        project_points(points, k, e, size, camera_model="auto")


def test_patch_centers_row_major_masks_and_gather_gradient():
    size = torch.tensor([[[32., 24.]]])
    grid = torch.tensor([[[4, 3]]])
    pixel = torch.tensor([[[[[3.5, 3.5], [27.5, 19.5], [50., 50.]]]]])
    valid = torch.tensor([[[[True, True, False]]]])
    ix, mask = neighborhood_indices(pixel, valid, size, grid)
    assert ix[0, 0, 0, :, 4].tolist() == [0, 11, 11]
    assert mask[0, 0, 0].sum(-1).tolist() == [4, 4, 0]
    memory = torch.arange(12., dtype=torch.float64).reshape(1, 1, 12, 1).requires_grad_()
    values = gather_neighborhood(memory, ix)
    (values*mask[..., None]).sum().backward()
    assert memory.grad.sum() == 8
    assert memory.grad[0, 0, 0] == 1 and memory.grad[0, 0, 11] == 1


def test_candidate_order_and_height_hypotheses_no_future_inputs():
    trajectory = torch.tensor([[[[5., 1., 0.], [10., 2., .2]], [[6., 0., 0.], [12., 1., .3]]]])
    anchors = candidate_anchors(trajectory)
    assert anchors.shape == (1, 2, 4, 3)
    assert anchors[0, 0, :, 2].tolist() == [0., 1., 0., 1.]
    assert torch.equal(candidate_anchors(trajectory[:, [1, 0]]), anchors[:, [1, 0]])


def test_invalid_calibration_is_rejected():
    k, e, size = calibration()
    e[..., 0, 0] = 2.
    with pytest.raises(ValueError):
        project_points(torch.zeros(1, 1, 8, 3, dtype=torch.float64), k, e, size)


def scorer_example():
    from iqe.contracts import TrajectoryContract
    from iqe.geometric_readout.model import GeometricReadoutScorer
    contract=TrajectoryContract(horizon=8,dt=.5,raw_representation='normalized_xy_sincos',
        frame='ego_relative',origin='rear_axle',yaw_unit='radian',
        mean=(10.172484,.360762,0.,0.),scale=(8.805105,2.277741,1.,1.),statistics_source='test fixture')
    torch.manual_seed(17)
    model=GeometricReadoutScorer(12,12,contract,visual_dim=12).eval()
    c={'world':torch.randn(1,3,12,requires_grad=True),'action':torch.randn(1,8,12,requires_grad=True),
       'current_ego':torch.randn(1,4,requires_grad=True),'world_valid':torch.ones(1,3,dtype=torch.bool),
       'action_valid':torch.ones(1,8,dtype=torch.bool),'visual':torch.randn(1,3,576,12,requires_grad=True),
       'visual_valid':torch.ones(1,3,576,dtype=torch.bool),'visual_xy':torch.randn(1,3,576,2)}
    e=torch.eye(4).expand(1,3,4,4).clone()
    e[...,:3,:3]=torch.tensor([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])
    e[...,2,3]=1.5
    c.update(camera_intrinsics=torch.tensor([[824.,0.,512.],[0.,824.,288.],[0.,0.,1.]]).expand(1,3,3,3),
             camera_to_ego=e,camera_image_sizes=torch.tensor([1024.,576.]).expand(1,3,2))
    t=torch.zeros(1,2,8,3);t[0,:,:,0]=torch.arange(1,9)*5.;t[0,1,:,1]=1.
    return model,c,t,contract


def test_same_random_parameterization_no_external_weights_and_input_gradients():
    from iqe.specialist_study.models import StudyScorer
    from iqe.evaluation.retention import module_hash
    model,c,t,contract=scorer_example()
    torch.manual_seed(17)
    control=StudyScorer(12,12,contract,spatial=True,visual_dim=12).eval()
    assert module_hash(model)==module_hash(control)
    t.requires_grad_()
    result=model(c,t)
    (result['values'].sum()+result['component_logits'].sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert c['world'].grad is c['action'].grad is c['current_ego'].grad is c['visual'].grad is t.grad is None
    assert model.spatial_reader.projection[1].weight.grad.abs().sum()>0


def test_candidate_independence_single_projection_and_missing_view_finite():
    model,c,t,_=scorer_example();calls=[]
    h=model.spatial_reader.projection.register_forward_hook(lambda *_:calls.append(1))
    with torch.no_grad():
        a=model(c,t)['values'];assert len(calls)==1
        b=model(c,t[:,[1,0]])['values']
        d=model(c,t[:,[0,1,0]])['values']
        torch.testing.assert_close(a,b[:,[1,0]],rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(a,d[:,:2],rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(d[:,0],d[:,2],rtol=1e-5,atol=1e-6)
        t[...,0]=-10.
        assert torch.isfinite(model(c,t)['values']).all()
    h.remove()
    with pytest.raises(ValueError):model(c|{'GT':t},t)
