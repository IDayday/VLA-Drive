from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from starVLA.dataloader.structured_world.adapters import box_container, transform_boxes_to_ego, FrameAnnotation
from starVLA.dataloader.structured_world.centered_boxes import CenteredOccFlowLabels
from starVLA.dataloader.structured_world.labels import occupancy
from starVLA.model.modules.structured_world.grid import GridSpec


def test_tilted_sensor_box_center_is_rigid_and_input_unchanged():
    original = box_container([[10., -3., 1., 5., 2., 3., .4]])
    before = original.tensor.clone()
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('xyz', [.04, -.09, .6]).as_matrix()
    transform[:3, 3] = [1.2, -.4, 1.7]
    actual = transform_boxes_to_ego(original, transform)
    expected = original.gravity_center.numpy()@transform[:3, :3].T+transform[:3, 3]
    np.testing.assert_allclose(actual.gravity_center.numpy(), expected, atol=2e-6)
    assert torch.equal(original.tensor, before)


def test_mature_future_reframe_preserves_physical_center_and_clone():
    original = box_container([[15., 2., 1., 4., 1.8, 2.4, .7]])
    before = original.tensor.clone()
    identity = dict(l2e_r=np.eye(3), l2e_t=np.zeros(3), e2g_r=np.eye(3), e2g_t=np.zeros(3))
    current = {**identity, 'e2g_r': Rotation.from_euler('xyz', [.1, -.15, -.7]).as_matrix(), 'e2g_t': np.array([4., 3., 0.])}
    generator = CenteredOccFlowLabels(dict(xbound=[-20., 80., .5], ybound=[-40., 40., .5], zbound=[-4., 4., 8.]), compute_flow=False)
    transformed = generator.reframe_boxes(original, identity, current)
    expected = original.gravity_center.numpy()@current['e2g_r'].T+current['e2g_t']
    np.testing.assert_allclose(transformed.gravity_center.numpy(), expected, atol=2e-6)
    assert torch.equal(original.tensor, before)


def test_truncated_label_lidar_is_unknown_and_keeps_ego_eligibility():
    frame = FrameAnnotation(box_container([[10., 0., 0., 4., 2., 2., 0.]]), np.array([0]), ['static'], np.array([1]),
                            np.eye(4), 0, True, '/unavailable/truncated.pcd', np.eye(4))
    scene = SimpleNamespace(frames=[frame, frame], inverse_transform=np.eye(4), future_valid=np.array([True]))
    with patch('starVLA.dataloader.structured_world.labels.lidar_ego', side_effect=RuntimeError('Incomplete pointcloud stream')):
        labels, valid, _, current_points, quality = occupancy(scene, GridSpec())
    assert np.all(labels == 255) and not valid.any()
    assert current_points.shape == (0, 3)
    assert scene.future_valid.all()
    assert all(row['reason'] == 'invalid_label_only_lidar' for row in quality)
