import pytest
import torch

from tools.st_world_policy.teacher_pca import StreamingCovariance,TeacherPCA,make_patch_plan


def fit(x,k):
    s=StreamingCovariance(x.shape[1])
    for chunk in x.split(17):s.update(chunk)
    return s.basis(k),s


def test_streaming_statistics_equal_full_and_nonwhiten():
    rng=torch.Generator().manual_seed(42)
    x=torch.randn(400,8,generator=rng)*torch.arange(1,9)+torch.arange(8)
    p,s=fit(x,8)
    assert torch.allclose(s.mean,x.double().mean(0),atol=1e-10)
    assert torch.allclose(p['channel_std'],x.std(0,unbiased=False),atol=1e-6)
    t=TeacherPCA(p,dimension=8,identity='test',teacher_contract_identity='a'*64)
    z=t.transform(x);reconstructed=t.inverse(z)
    assert torch.allclose(x,reconstructed,atol=6e-6)
    # Nonwhitening preserves eigenvalue-dependent coordinate variance.
    assert torch.allclose(z.var(0),p['eigenvalues'],atol=2e-6)
    assert not torch.allclose(z.var(0),torch.ones(8),atol=.01)


def test_dimensions_share_stats_and_reconstruction_nested():
    x=torch.randn(200,8,generator=torch.Generator().manual_seed(1))
    p,_=fit(x,6)
    a=TeacherPCA(p,dimension=2,identity='test',teacher_contract_identity='a'*64)
    b=TeacherPCA(p,dimension=6,identity='test',teacher_contract_identity='a'*64)
    assert torch.equal(a.transform(x),b.transform(x)[:,:2])
    assert (b.inverse(b.transform(x))-x).square().mean()<(a.inverse(a.transform(x))-x).square().mean()
    # Delta uses the same linear transform: offsets cancel; magnitude is retained.
    y=x*.8
    expected=((y-x)/p['channel_std'])@p['components'][:2].T
    assert torch.allclose(a.transform(y)-a.transform(x),expected,atol=1e-6)


def test_zero_variance_and_nonfinite():
    x=torch.randn(80,5);x[:,2]=4
    p,_=fit(x,4)
    assert p['constant_channels'][2] and p['channel_std'][2]>0
    assert torch.isfinite(TeacherPCA(p,dimension=4,identity='x',teacher_contract_identity='a'*64).transform(x)).all()
    s=StreamingCovariance(5)
    with pytest.raises(ValueError):s.update(torch.full((5,5),float('nan')))
    with pytest.raises(ValueError):s.basis(4)


def test_role_guard_precedes_any_cache_read():
    class Reader:
        manifest={'split':'dev','scenes':[]}
    with pytest.raises(ValueError,match='train-only'):make_patch_plan(Reader(),seed=1,max_patches=32,purpose='fit')


def scoped_fixture(tmp_path):
    import json
    from PIL import Image
    from safetensors.torch import save_file
    from starVLA.dataloader.st_world_history import image_reference,file_sha256
    from starVLA.dataloader.st_world_schema import TeacherFeatureContract,identity_hash
    from starVLA.dataloader.st_world_targets import TeacherCacheWriter,tensor_digest
    image=tmp_path/'input.png';Image.new('RGB',(32,18),'blue').save(image);ref=image_reference(image)
    teacher={'repository':'registered','revision':'same','weight_sha256':'a'*64,'config_sha256':'b'*64,
        'encoder_source_sha256':'c'*64,'timm':'1.0','torch':'2.5','feature':'last_block_post_norm_dense_patch',
        'extra_norm':'none','prefix_tokens_excluded':5,'compute_dtype':'float32','tf32':False,'autocast':False}
    transform={'source':'complete original16:9 RGB','crop':'none','resize':'PIL_BICUBIC','augmentation':'none',
        'padding':'none','image_hw':[288,384],'normalization':'ImageNet_once','mean':[.485,.456,.406],'std':[.229,.224,.225]}
    m={'schema':'st_teacher_requests_v1','split':'train','times_s':[0.],'timestamp_tolerance_s':.05,
        'view_names':['CAM_F0','CAM_L0','CAM_R0'],'scenes':[{'token':'a','log':'l','split':'train','timestamp_us':0,
        'frames':[{'requested_s':0.,'timestamp_us':0,'views':[ref]*3}]}]};m['identity']=identity_hash(m)
    readers=[]
    for name,pool in [('fit',1),('pooled',3)]:
        c=TeacherFeatureContract(image_hw=[288,384],patch_hw=[16,16],native_grid_hw=[18,24],
            output_grid_hw=[18//pool,24//pool],channels=1024,storage_dtype='float32',compute_dtype='float32',
            view_names=m['view_names'],teacher_identity=identity_hash(teacher),transform_identity=identity_hash(transform),
            pooling={'kind':'none' if pool==1 else 'avg','kernel_hw':[pool,pool],'stride_hw':[pool,pool]})
        writer=TeacherCacheWriter(tmp_path/name,c,m,execution={'teacher':teacher,'transform':transform})
        writer.commit_shard(0,torch.zeros(1,*c.output_grid_hw,1024),torch.ones(1,*c.output_grid_hw,dtype=torch.bool));writer.finalize()
        readers.append(writer)
    pca=tmp_path/'pca';pca.mkdir()
    t={'channel_mean':torch.zeros(1024),'channel_std':torch.ones(1024),'pca_mean':torch.zeros(1024),
       'components':torch.eye(1024)[:4].contiguous(),'eigenvalues':torch.ones(1024),'constant_channels':torch.zeros(1024,dtype=torch.bool)}
    save_file(t,str(pca/'statistics.safetensors'))
    receipt={'schema':'st_teacher_pca_v1','fit_role':'train','fit_cache_identity':readers[0].identity['identity'],
        'teacher_contract_identity':readers[0].contract.identity,'dimensions':[2,4],'whiten':False,'standardize_channels':True,
        'tensor_sha256':file_sha256(pca/'statistics.safetensors'),'tensor_hashes':{k:tensor_digest(v) for k,v in t.items()}}
    receipt['identity']=identity_hash(receipt);(pca/'PCA.json').write_text(json.dumps(receipt))
    (pca/'COMPLETE.json').write_text(json.dumps({'identity':receipt['identity'],'pca_json_sha256':file_sha256(pca/'PCA.json')}))
    return readers,pca,receipt


def test_cross_grid_pca_requires_explicit_exact_cache_scope(tmp_path):
    from tools.st_world_policy.teacher_pca import register_channel_scope
    readers,pca,receipt=scoped_fixture(tmp_path);fit,pooled=readers
    kwargs=dict(dimension=2,expected_identity=receipt['identity'],expected_teacher_contract=pooled.contract.identity)
    with pytest.raises(ValueError,match='exact teacher contract mismatch'):TeacherPCA.load(pca,**kwargs)
    scope=register_channel_scope(pca,fit.root,[pooled.root],pca/'CHANNEL_SCOPE.json')
    loaded=TeacherPCA.load(pca,**kwargs,channel_scope=pca/'CHANNEL_SCOPE.json',target_cache_identity=pooled.identity['identity'])
    assert loaded.channel_scope_identity==scope['identity']
    assert loaded.fit_teacher_contract_identity==fit.contract.identity
    assert loaded.teacher_contract_identity==pooled.contract.identity
    with pytest.raises(ValueError,match='not explicitly admitted'):
        TeacherPCA.load(pca,**kwargs,channel_scope=pca/'CHANNEL_SCOPE.json',target_cache_identity='f'*64)
    # Original fitted tensors/receipt are immutable; no refit or edited source.
    assert scope['statistics_refitted'] is False


def test_family_does_not_admit_precision_normalization_or_teacher_changes(tmp_path):
    import copy
    from tools.st_world_policy.teacher_pca import teacher_channel_family
    from starVLA.dataloader.st_world_schema import identity_hash
    readers,_,_=scoped_fixture(tmp_path);source=readers[0].identity
    original=teacher_channel_family(source)
    changed=copy.deepcopy(source);changed['execution']['teacher']['tf32']=True
    changed['contract']['teacher_identity']=identity_hash(changed['execution']['teacher'])
    with pytest.raises(ValueError,match='precision/preprocessing'):teacher_channel_family(changed)
    changed=copy.deepcopy(source);changed['execution']['transform']['augmentation']='random_crop'
    changed['contract']['transform_identity']=identity_hash(changed['execution']['transform'])
    with pytest.raises(ValueError,match='precision/preprocessing'):teacher_channel_family(changed)
    changed=copy.deepcopy(source);changed['execution']['teacher']['weight_sha256']='f'*64
    changed['contract']['teacher_identity']=identity_hash(changed['execution']['teacher'])
    assert teacher_channel_family(changed)!=original


def test_linear_pooling_and_one_common_pca_commute_without_refit():
    x=torch.randn(40,8,generator=torch.Generator().manual_seed(41));p,_=fit(x,4)
    pca=TeacherPCA(p,dimension=4,identity='x',teacher_contract_identity='a'*64)
    grid=x[:36].reshape(2,3,2,3,8)
    pooled=grid.mean((1,3))
    assert torch.allclose(pca.transform(pooled),pca.transform(grid).mean((1,3)),atol=2e-7)


def test_namespace_clone_preserves_old_scope_and_all_fixed_statistics(tmp_path,monkeypatch):
    from tools.st_world_policy.teacher_pca import register_channel_scope,clone_basis_namespace
    from starVLA.dataloader.st_world_history import file_sha256
    readers,source,receipt=scoped_fixture(tmp_path);fit,pooled=readers
    register_channel_scope(source,fit.root,[],source/'CHANNEL_SCOPE.json')
    old_scope=file_sha256(source/'CHANNEL_SCOPE.json')
    monkeypatch.setattr(StreamingCovariance,'update',lambda *a:(_ for _ in ()).throw(AssertionError('No refit allowed')))
    target=tmp_path/'new_pca'
    result=clone_basis_namespace(source,target,fit.root,[pooled.root])
    assert result['statistics_refitted'] is False and result['old_namespace_modified'] is False
    assert file_sha256(source/'CHANNEL_SCOPE.json')==old_scope
    for name in ['PCA.json','COMPLETE.json','statistics.safetensors']:
        assert (source/name).read_bytes()==(target/name).read_bytes()
    assert clone_basis_namespace(source,target,fit.root,[pooled.root])==result
    with pytest.raises(ValueError,match='new namespace'):clone_basis_namespace(source,source,fit.root,[])
    with pytest.raises(ValueError,match='Immutable artifact conflict'):clone_basis_namespace(source,target,fit.root,[])
    loaded=TeacherPCA.load(target,dimension=2,expected_identity=receipt['identity'],
        expected_teacher_contract=pooled.contract.identity,channel_scope=target/'CHANNEL_SCOPE.json',
        target_cache_identity=pooled.identity['identity'])
    assert loaded.channel_scope_identity==result['channel_scope_identity']


def test_explicit_scope_is_checked_even_for_same_grid_contract(tmp_path):
    from tools.st_world_policy.teacher_pca import register_channel_scope
    readers,root,receipt=scoped_fixture(tmp_path);fit,_=readers
    scope=register_channel_scope(root,fit.root,[],root/'CHANNEL_SCOPE.json')
    kwargs=dict(dimension=2,expected_identity=receipt['identity'],expected_teacher_contract=fit.contract.identity,
        channel_scope=root/'CHANNEL_SCOPE.json')
    assert TeacherPCA.load(root,**kwargs,target_cache_identity=fit.identity['identity']).channel_scope_identity==scope['identity']
    with pytest.raises(ValueError,match='not explicitly admitted'):
        TeacherPCA.load(root,**kwargs,target_cache_identity='f'*64)


def test_probe_raw_distortion_and_full_field_resize_projection():
    import cv2
    import numpy as np
    from reports.st_world_policy.history_data.spatial_probes import project
    camera={'sensor2lidar_rotation':np.eye(3),'sensor2lidar_translation':[1.,2.,0.],
            'cam_intrinsic':[[200.,0.,384.],[0.,200.,288.],[0.,0.,1.]],
            'distortion':[.1,-.05,.001,.002,0.]}
    xyz=np.array([[2.,3.,10.],[0.,0.,-1.]])
    uv,z,valid=project(xyz,camera,(768,576))
    expected=cv2.projectPoints(xyz[:1]-[1.,2.,0.],np.zeros(3),np.zeros(3),
        np.asarray(camera['cam_intrinsic']),np.asarray(camera['distortion']))[0].reshape(-1,2)
    assert np.allclose(uv[0],(expected[0]+.5)*.5-.5)
    assert valid.tolist()==[True,False] and np.isnan(uv[1]).all() and z.tolist()==[10.,-1.]


def test_probe_requires_lidar_visibility_and_never_makes_missing_depth_negative():
    import numpy as np
    from reports.st_world_policy.history_data.spatial_probes import current_labels
    camera={'sensor2lidar_rotation':np.eye(3),'sensor2lidar_translation':[0.,0.,0.],
            'cam_intrinsic':[[100.,0.,192.],[0.,100.,144.],[0.,0.,1.]],'distortion':[0.]*5}
    pixel=np.array([[x,y] for y in range(142,147) for x in range(190,195)])
    def points(depth):return np.c_[(pixel[:,0]-192)*depth/100,(pixel[:,1]-144)*depth/100,np.full(len(pixel),depth)]
    box=np.array([[0.,0.,10.,.8,.8,2.,0.]])
    visible=current_labels(points(10),box,camera,(384,288))
    assert visible['small_vehicle'].sum()>0 and visible['statistics']['supported_small_boxes']==1
    occluded=current_labels(np.r_[points(10),points(5)],box,camera,(384,288))
    assert occluded['small_vehicle'].sum()==0 and occluded['statistics']['supported_small_boxes']==0
    # The foreground pixels are not mislabeled as background inside an
    # occluded projected box; absent LiDAR elsewhere is never a negative.
    assert not occluded['small_valid'].any()
    missing=current_labels(np.empty((0,3)),np.empty((0,7)),camera,(384,288))
    assert not missing['depth_valid'].any() and not missing['small_valid'].any()
