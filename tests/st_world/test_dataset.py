import copy
import json

import numpy as np
from PIL import Image
import pytest
import torch

from starVLA.dataloader.st_world_dataset import STWorldTrainingDataset,collate_st_world,projected_teacher_contract
from starVLA.dataloader.st_world_history import image_reference,file_sha256
from starVLA.dataloader.st_world_schema import TeacherFeatureContract,identity_hash
from starVLA.dataloader.st_world_targets import TeacherCacheWriter


def setup_data(tmp_path, times=(0., .5, 1., 2., 4.)):
    data=tmp_path/'data';(data/'current').mkdir(parents=True);(data/'ego').mkdir()
    index=[{'token':'a','log':'log_a'}]
    source={'schema':'ddpolicy_current_cameras_v1','split':'train','index_sha256':identity_hash(index),
            'partition_sha256':'c'*64};source['identity']=identity_hash(source)
    (data/'identity.json').write_text(json.dumps(source));(data/'index.json').write_text(json.dumps(index))
    image=tmp_path/'image.png';Image.new('RGB',(32,18),'blue').save(image)
    record={'identity':source['identity'],'token':'a','log':'log_a','timestamp':1000000,
            'image_paths':[str(image)]*3,'global_pose_history':[[0,0,0],[0,0,0],[1,0,0],[2,0,0]],'navigation':1}
    (data/'current/a.json').write_text(json.dumps(record))
    ego={'schema':'foresight_ego_labels_v1','split':'train','index_sha256':source['index_sha256']}
    ego['identity']=identity_hash(ego);(data/'ego_identity.json').write_text(json.dumps(ego))
    torch.save({'identity':ego['identity'],'token':'a','ego':torch.zeros(8,4)},data/'ego/a.pt')
    selection={'schema':'st_fixed_scene_list_v1','split':'train','seed':1,'source_index_hash':source['index_sha256'],
        'scenes':[dict(index[0],current_record_sha256=file_sha256(data/'current/a.json'),ego_record_sha256=file_sha256(data/'ego/a.pt'))]}
    manifest=tmp_path/'SELECTED_SCENES.json';manifest.write_text(json.dumps(selection))
    c=TeacherFeatureContract(image_hw=(32,32),patch_hw=(16,16),native_grid_hw=(2,2),output_grid_hw=(2,2),
        channels=8,storage_dtype='float32',compute_dtype='float32',view_names=['CAM_F0','CAM_L0','CAM_R0'],
        teacher_identity='a'*64,transform_identity='b'*64,pooling={'kind':'none','kernel_hw':[1,1],'stride_hw':[1,1]})
    ref=image_reference(image)
    m={'schema':'st_teacher_requests_v1','split':'train','times_s':list(times),
       'view_names':list(c.view_names),'timestamp_tolerance_s':.05,'source_student_identity':source['identity'],
       'source_index_sha256':source['index_sha256'],'partition_sha256':source['partition_sha256'],
       'scenes':[{'token':'a','log':'log_a','split':'train','timestamp_us':1000000,'frames':[
        {'requested_s':0.,'timestamp_us':1000000,'views':[ref]*3},
        *[{'requested_s':t,'timestamp_us':None,'views':[None]*3} for t in times[1:]]]}]}
    m['identity']=identity_hash(m)
    w=TeacherCacheWriter(tmp_path/'cache',c,m);w.commit_shard(0,torch.arange(32,dtype=torch.float32).reshape(1,2,2,8),torch.ones(1,2,2,dtype=torch.bool));w.finalize()
    return {'data_root':data,'manifest_path':manifest,'teacher_root':w.root,'expected_teacher_identity':w.identity['identity'],'expected_contract':c}


def test_original_inputs_targets_and_current_only(tmp_path):
    args=setup_data(tmp_path);ds=STWorldTrainingDataset(**args)
    inputs,targets=ds[0]
    assert set(inputs.current)=={'image','state','lang','token'}
    assert len(inputs.current['image'])==3 and inputs.current['image'][0].size==(1024,576)
    assert targets['ego'].shape==(8,4) and targets['future_dense_features'].shape==(0,3,2,2,8)
    images,batched=collate_st_world([ds[0],ds[0]])
    assert len(images)==2 and batched['current_dense_features'].shape==(2,3,2,2,8)
    future=STWorldTrainingDataset(**args,need_future=True)[0][1]
    assert future['future_dense_features'].shape==(4,3,2,2,8) and not future['future_valid'].any()
    assert torch.equal(future['future_times_s'],torch.tensor([.5,1.,2.,4.]))


def test_future_time_contract_refused_even_with_matching_shape(tmp_path):
    args=setup_data(tmp_path,times=(0.,.5,1.,3.,4.))
    with pytest.raises(ValueError,match='exact times'):STWorldTrainingDataset(**args,need_future=True)
    # Current-only SP0 does not require or read the four future horizons.
    current=STWorldTrainingDataset(**args,need_future=False)[0][1]
    assert current['future_times_s'].numel()==0


def test_current_only_cache_is_legal_for_sp0_not_future_supervision(tmp_path):
    args=setup_data(tmp_path,times=(0.,))
    assert STWorldTrainingDataset(**args)[0][1]['future_dense_features'].shape[0]==0
    with pytest.raises(ValueError,match='exact times'):STWorldTrainingDataset(**args,need_future=True)


def test_target_changes_cannot_change_inputs_and_are_pinned(tmp_path):
    args=setup_data(tmp_path);ds=STWorldTrainingDataset(**args);old=ds[0][0]
    path=args['data_root']/'ego/a.pt';z=torch.load(path,weights_only=True);z['ego']+=5;torch.save(z,path)
    with pytest.raises(ValueError,match='Ego target changed'):ds[0]
    # A newly registered dataset records the different supervised label identity,
    # but its original visual/history input remains bitwise unchanged.
    with pytest.raises(ValueError,match='fixed manifest'):STWorldTrainingDataset(**args)
    new_manifest=json.loads(args['manifest_path'].read_text())
    new_manifest['scenes'][0]['ego_record_sha256']=file_sha256(path)
    new_path=tmp_path/'NEW_SELECTION.json';new_path.write_text(json.dumps(new_manifest))
    changed=STWorldTrainingDataset(**dict(args,manifest_path=new_path));new=changed[0][0]
    assert np.array_equal(old.current['state'],new.current['state'])
    assert old.current['lang']==new.current['lang']
    assert all(a.tobytes()==b.tobytes() for a,b in zip(old.current['image'],new.current['image']))
    assert ds.identity['identity']!=changed.identity['identity']


def test_foreign_role_source_or_mutated_current_refused(tmp_path):
    args=setup_data(tmp_path)
    for role in ['dev','test','navtest']:
        with pytest.raises(ValueError):STWorldTrainingDataset(**args,require_split=role)
    ds=STWorldTrainingDataset(**args)
    p=args['data_root']/'current/a.json';r=json.loads(p.read_text());r['navigation']=2;p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='Current JSON changed'):ds[0]
    with pytest.raises(ValueError,match='Current record changed'):STWorldTrainingDataset(**args)


def test_manifest_never_silently_shrinks_and_pca_contract_is_new(tmp_path):
    args=setup_data(tmp_path);m=json.loads(args['manifest_path'].read_text())
    m['scenes'].append({'token':'missing','log':'new'});args['manifest_path'].write_text(json.dumps(m))
    with pytest.raises(ValueError,match='outside original role'):STWorldTrainingDataset(**args)
    c=args['expected_contract'];p=projected_teacher_contract(c,pca_identity='e'*64,dimension=4)
    assert p.channels==4 and p.transform_identity!=c.transform_identity and p.identity!=c.identity
    with pytest.raises(ValueError):STWorldTrainingDataset(**dict(args,expected_contract=p))


def test_legacy_pin_upgrade_is_explicit_preserves_prior_and_refuses_changed_pin(tmp_path):
    from types import SimpleNamespace
    from tools.st_world_policy.prepare_history import upgrade_selection
    args=setup_data(tmp_path);old=args['manifest_path'];value=json.loads(old.read_text())
    del value['scenes'][0]['ego_record_sha256'];old.write_text(json.dumps(value));old_hash=file_sha256(old)
    with pytest.raises(ValueError,match='Unpinned legacy'):STWorldTrainingDataset(**args)
    upgrade=SimpleNamespace(pin_selection=str(old),output=str(tmp_path/'upgraded'),student_root=str(args['data_root']),
        max_scenes=1,split='train')
    receipt=upgrade_selection(upgrade)
    assert receipt['old_evidence_modified'] is False and file_sha256(old)==old_hash
    assert upgrade_selection(upgrade)==receipt
    upgraded=tmp_path/'upgraded/SELECTED_SCENES.json'
    assert len(STWorldTrainingDataset(**dict(args,manifest_path=upgraded)))==1
    with pytest.raises(ValueError,match='new output namespace'):
        upgrade_selection(SimpleNamespace(**dict(vars(upgrade),output=str(old.parent))))
    value=json.loads(upgraded.read_text());value['scenes'][0]['ego_record_sha256']='a'*64;upgraded.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='fixed manifest'):STWorldTrainingDataset(**dict(args,manifest_path=upgraded))
