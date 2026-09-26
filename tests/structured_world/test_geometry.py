import numpy as np
from starVLA.model.modules.structured_world.geometry import future_track_positions, crop_resize_intrinsics


def test_static_world_track_does_not_move_with_ego():
    frames = []
    point = np.array([20., 3., 0.])
    for t in range(8):
        a = t * .1
        pose = np.eye(4)
        pose[:2,:2] = [[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]
        pose[:3,3] = [t, .2*t, 0]
        local = (point - pose[:3,3]) @ pose[:3,:3]
        frames.append({'ego2global':pose,'anns':{'track_tokens':['car'], 'gt_boxes':local[None]}})
    xy, mask = future_track_positions(['car','missing'], frames, np.eye(4))
    np.testing.assert_allclose(xy[0], np.tile(point[:2], (8,1)), atol=1e-5)
    assert mask[0].all() and not mask[1].any()


def test_crop_projection():
    k = np.array([[1545.,0,960],[0,1545,560],[0,0,1]])
    new, affine = crop_resize_intrinsics(k,(1920,1080),(1024,576))
    p = np.array([.2,.1,1.])
    np.testing.assert_allclose(new @ p, affine @ (k @ p))
