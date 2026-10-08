import ast
from pathlib import Path
import numpy as np
import torch
from starVLA.dataloader.structured_world.nuscenes_adapter import to_internal, to_lidar, INTERNAL_TO_LIDAR
from starVLA.model.modules.structured_world.nuscenes_evaluation import scene_metrics
from starVLA.model.modules.structured_world.queries import body_points


def test_nuscenes_axes_roundtrip_and_forward_left():
    source = np.array([[0., 10.], [-2., 10.], [3., -1.]])
    assert np.array_equal(to_lidar(to_internal(source)), source)
    assert np.array_equal(to_internal(source[0]), [10., 0.])
    assert np.array_equal(to_internal(source[1]), [10., 2.])
    assert np.allclose(INTERNAL_TO_LIDAR.T@INTERNAL_TO_LIDAR, np.eye(3))
    assert np.linalg.det(INTERNAL_TO_LIDAR) == 1.


def test_original_uniad_metric_class_is_unchanged():
    root = Path(__file__).resolve().parents[2]
    source = root/'third_party/uniad/upstream/projects/mmdet3d_plugin/uniad/dense_heads/planning_head_plugin/planning_metrics.py'
    port = root/'third_party/uniad/ported/planning_metrics.py'
    def cls(path):
        return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)), include_attributes=False)
    assert cls(source) == cls(port)


def test_native_metric_wrapper_preserves_predictions_and_exact_gt_L2():
    internal = np.column_stack((np.arange(1, 7)*2., np.zeros(6), np.zeros(6))).astype(np.float32)
    native = to_lidar(internal[:, :2]); before = internal.copy(); gt_before = native.copy()
    result = scene_metrics(internal, native, np.zeros((6, 200, 200), np.uint8))
    assert result['L2_3s'] == 0 and result['box_collision_3s'] == 0
    assert np.array_equal(internal, before) and np.array_equal(native, gt_before)
    shifted = internal.copy(); shifted[:, 1] += 1.
    assert scene_metrics(shifted, native, np.zeros((6, 200, 200), np.uint8))['L2_3s'] == 1.


def test_native_GT_collision_exclusion_and_outside_population_are_disclosed():
    internal = np.zeros((6, 3), np.float32); internal[:, 0] = 60.
    result = scene_metrics(internal, np.zeros((6, 2)), np.ones((6, 200, 200), np.uint8))
    assert result['native_out_of_range_steps'] == 6
    # The original UniAD rule suppresses predicted collisions at GT collision
    # steps, while L2 still counts the scene. This test locks that behavior.
    assert result['box_collision_3s'] == 0. and result['L2_3s'] == 60.


def test_body_corners_rotation_offset_and_registered_spacing():
    pose = torch.tensor([[0., 0., torch.pi/2]])
    body = {'length': 4.084, 'width': 1.85, 'rear_axle_to_center': .5}
    points = body_points(pose, body)[0]
    assert torch.isclose(points[:, 1].mean(), torch.tensor(.5), atol=1e-6)
    assert torch.isclose(points[:, 0].max()-points[:, 0].min(), torch.tensor(1.85))
    assert torch.isclose(points[:, 1].max()-points[:, 1].min(), torch.tensor(4.084))
    assert points.shape[0] >= 50
