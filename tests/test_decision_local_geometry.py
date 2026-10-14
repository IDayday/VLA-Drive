import numpy as np
import pytest
from starVLA.model.modules.structured_world.grid import GridSpec
from starVLA.model.modules.decision_local.consequence_geometry import obstacle_distance, read_consequences, current_cv_occupancy, carry_static_ground_support


def test_current_frustum_uses_positive_depth_raw_roi_and_sensor_transform():
    from tools.decision_local.audit_current_frustum import points_in_current_camera
    K=np.array([[100.,0.,400.],[0.,100.,300.],[0.,0.,1.]])
    sensor=np.eye(4);sensor[:3,3]=[2.,-3.,1.]
    camera=np.array([[0.,0.,10.],[0.,23.,10.],[50.,0.,10.],[0.,0.,-10.]])
    original=camera.copy()
    # 800x600 source becomes a y=[75,524] crop. The second point is in
    # the original image but outside the allowed S0 ROI.
    mask=points_in_current_camera(camera+sensor[:3,3],sensor,K,None,(800,600))
    np.testing.assert_array_equal(mask,[True,False,False,False])
    np.testing.assert_array_equal(camera,original)
    # Positive radial distortion can push a formerly valid ray beyond the ROI;
    # this must not be treated as a rectified image or clamped to the border.
    p=np.array([[30.,0.,10.]])+sensor[:3,3]
    assert points_in_current_camera(p,sensor,K,None,(800,600))[0]
    assert not points_in_current_camera(p,sensor,K,np.array([.1,0,0,0,0]),(800,600))[0]


def test_static_ground_carry_never_clears_untrusted_objects_or_missing_frames():
    occ=np.zeros((3,4,4),np.uint8);valid=np.ones_like(occ,bool);instance=np.zeros_like(occ)
    occ[1:]=255;valid[1:]=False;instance[1,2,2]=42;valid[0,0,0]=False
    before=occ.copy()
    result,support,receipt=carry_static_ground_support(occ,valid,instance,[True,False],[{}, {}, {}])
    assert result[1,1,1]==0 and support[1,1,1]
    assert result[1,2,2]==255 and not support[1,2,2]
    assert result[1,0,0]==255 and not support[1,0,0]
    assert not support[2].any() and (result[2]==255).all()
    np.testing.assert_array_equal(occ,before)


def test_signed_distance_and_unknown():
    g=GridSpec(xmin=-10,xmax=10,ymin=-10,ymax=10)
    occ=np.zeros((g.height,g.width),np.uint8); valid=np.ones_like(occ,bool)
    occ[16:24,16:24]=1
    sdf,support=obstacle_distance(occ,valid,g)
    assert sdf[20,20]<0 and sdf[20,25]>0 and support[20,20]
    valid[:,26:]=False
    _,support=obstacle_distance(occ,valid,g)
    assert not support[20,27]
    occ[:]=0; valid[:]=True
    sdf,support=obstacle_distance(occ,valid,g,clip_m=2.)
    assert sdf[20,20]==2 and support[20,20] and not support[0,0]


def test_full_body_crosses_road_rotated_and_outside():
    g=GridSpec(xmin=-10,xmax=10,ymin=-10,ymax=10)
    x=g.xmin+(np.arange(g.width)+.5)*g.dx
    # Exact SDF to half-plane x<2; bilinear raster is exact away from border.
    road=np.broadcast_to(2-x,(g.height,g.width)).astype(np.float32).copy()
    occ=np.zeros((1,g.height,g.width),np.uint8); valid=np.ones_like(occ,bool)
    body={'length':4.,'width':2.,'rear_axle_to_center':1.}
    poses=np.array([[[0,0,0]],[[0,0,np.pi/2]],[[9,0,0]]],np.float32)
    value,mask,meta=read_consequences(road,valid[0],occ,valid,poses,body,grid=g,clip_m=2.)
    np.testing.assert_allclose(value[:2,0,0],[-1.,1.],atol=2e-6)
    assert mask[0,0,0] and not mask[2,0,0] and not meta['body_in_grid'][2,0]


def test_empty_current_frame_not_fake_obstacle_or_empty_unknown():
    g=GridSpec(xmin=-10,xmax=10,ymin=-10,ymax=10)
    frame={'ego2global_translation':[0,0,0],'ego2global_rotation':[1,0,0,0],
           'lidar2ego_translation':[0,0,0],'lidar2ego_rotation':[1,0,0,0],
           'anns':{'gt_names':np.array([],dtype=str),'gt_boxes':np.empty((0,7)), 'gt_velocity_3d':np.empty((0,3))}}
    current=np.zeros((g.height,g.width),np.uint8); valid=np.ones_like(current,bool);valid[:3]=False
    occ,mask,receipt=current_cv_occupancy(frame,current,valid,[.5,1.],grid=g)
    assert receipt['objects']==0 and (occ[:,:3]==255).all() and not mask[:,:3].any()
    assert (occ[:,3:]==0).all()
