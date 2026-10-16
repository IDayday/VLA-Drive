import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from starVLA.dataloader.st_world_history import (
    build_history_record, causal_frames, crop_resize_affine, load_history_images, validate_history_record)
from starVLA.dataloader.st_world_schema import identity_hash
from tools.st_world_policy.prepare_history import build_target_scene, stable_selection


def real_fixture(tmp_path):
    frames=[]
    for i in range(5):
        cams={}
        for name in ('CAM_F0','CAM_L0','CAM_R0'):
            path=tmp_path/('%d_%s.png'%(i,name));Image.new('RGB',(32,18),(i*30,20,40)).save(path)
            cams[name]={'data_path':path.name,'sensor2lidar_rotation':np.eye(3).tolist(),
                        'sensor2lidar_translation':[0,0,1], 'cam_intrinsic':np.eye(3).tolist(),'distortion':[0]*5}
        frames.append({'token':'t%d'%i,'timestamp':i*500000,'ego2global_rotation':[1,0,0,0],
                       'ego2global_translation':[i,0,0],'lidar2ego':np.eye(4).tolist(),'cams':cams,
                       'anns':{'DO_NOT_READ':'future-label'}})
    current={'token':'t2','log':'log','timestamp':1000000,'image_paths':[str(tmp_path/('2_%s.png'%c)) for c in cams]}
    return frames,current


def make(tmp_path):
    frames,current=real_fixture(tmp_path)
    return frames,current,build_history_record(frames,'t2','log',current,[tmp_path],raw_log_sha256='a'*64)


def test_history_true_offsets_geometry_and_pixels(tmp_path):
    frames,current,r=make(tmp_path)
    assert [f['token'] for f in r['frames']]==['t0','t1']
    assert r['frames'][0]['relative_ego_planar'][0][3]==-2
    assert r['frames'][0]['views'][0]['camera_to_current_planar_ego'][2][3]==1
    images,mask=load_history_images(r)
    assert mask.shape==(2,3) and mask.all()
    assert images[0][0].size==(1024,576)
    assert np.array(images[0][0])[0,0,0]!=np.array(images[1][0])[0,0,0]


def test_future_labels_do_not_affect_history(tmp_path):
    frames,current,r=make(tmp_path)
    altered=copy.deepcopy(frames)
    for f in altered:
        f['anns']={'future':float('nan'),'trajectory':[9999]}
        f['score']=1e9
    for f in altered[3:]:
        f['cams']={'MUST_NOT_READ':None}
        f['ego2global_rotation']='poison'
    after=build_history_record(altered,'t2','log',current,[tmp_path],raw_log_sha256='a'*64)
    assert after==r


def test_missing_stays_absent_partial_camera(tmp_path):
    frames,current=real_fixture(tmp_path)
    r=build_history_record(frames[2:],'t2','log',current,[tmp_path],raw_log_sha256='a'*64)
    images,mask=load_history_images(r)
    assert not mask.any() and all(x is None for row in images for x in row)
    Path(tmp_path/'0_CAM_L0.png').unlink()
    partial=build_history_record(frames,'t2','log',current,[tmp_path],raw_log_sha256='a'*64)
    assert not partial['frames'][0]['views'][1]['valid']
    assert partial['frames'][0]['views'][0]['valid']


def test_future_duplicate_log_wrong_view_and_corruption_reject(tmp_path):
    frames,current,r=make(tmp_path)
    with pytest.raises(ValueError): causal_frames(frames,'t2',offsets_s=[.5])
    with pytest.raises(ValueError): causal_frames(frames+[frames[0]],'t2')
    with pytest.raises(ValueError): build_history_record(frames,'t2','other',current,[tmp_path],raw_log_sha256='a'*64)
    changed=copy.deepcopy(r);changed['frames'][0]['timestamp_us']=1000000
    changed['identity']=identity_hash({k:v for k,v in changed.items() if k!='identity'})
    with pytest.raises(ValueError):validate_history_record(changed)
    changed=copy.deepcopy(r);changed['frames'][0]['views'][0]['camera']='CAM_R0'
    changed['identity']=identity_hash({k:v for k,v in changed.items() if k!='identity'})
    with pytest.raises(ValueError):validate_history_record(changed)
    Image.new('RGB',(32,18),'red').save(tmp_path/'0_CAM_F0.png')
    with pytest.raises(ValueError):load_history_images(r)


def test_crop_geometry_and_target_boundary(tmp_path):
    frames,current,r=make(tmp_path)
    crop,a=crop_resize_affine((1920,1080));assert crop==[0,0,1920,1080]
    assert np.allclose(a@np.array([960,540,1]),[512,288,1])
    label=build_target_scene(frames,'t2','log','train',[tmp_path],times_s=[0.,.5,1.,2.])
    assert label['frames'][1]['frame_token']=='t3' and label['frames'][-1]['timestamp_us'] is None
    assert 'teacher' not in r and 'future' not in r


def test_fixed_selection_is_score_free():
    index=[{'token':str(i),'log':str(i//4)} for i in range(20)]
    a=stable_selection(index,5,42);b=stable_selection(list(reversed(index)),5,42)
    assert a==b and len({r['log'] for r in a})==5


def test_collation_and_actual_sensor_time_guard(tmp_path):
    import torch
    from starVLA.dataloader.st_world_history import collate_history_inputs
    frames,current,r=make(tmp_path)
    batch=collate_history_inputs([r,r])
    assert batch['history_valid'].shape==(2,2,3) and batch['history_valid'].all()
    assert torch.equal(batch['history_times_s'][0],torch.tensor([-1.,-.5]))
    assert batch['history_ego_relative'].shape==(2,2,4,4)
    frames[0]['cams']['CAM_F0']['timestamp']=1000001
    with pytest.raises(ValueError,match='after current'):
        build_history_record(frames,'t2','log',current,[tmp_path],raw_log_sha256='a'*64)
