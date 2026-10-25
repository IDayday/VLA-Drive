from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from starVLA.model.modules.qwen_world_postcot.readouts import (
    DINOReadout,FutureReadout,CurrentObjectReadout,current_object_loss,
    valid_scene_mse,sample_future_queries,gather_future_targets)
from starVLA.model.modules.qwen_world_postcot.memory import StudentMemory
from starVLA.model.modules.qwen_world_postcot.rgb_decoder import DINORGBDecoder,rgb_reconstruction_loss
from starVLA.model.modules.qwen_world_postcot.teacher_targets import (
    BinaryTeacherCacheWriter,TeacherCacheReader,finalize_teacher_cache,tensor_layout,
    validate_clip_timestamps,clip_timing_valid,WAFutureTargetBatch,
    current_depth_from_ego,visible_current_centers,identity_hash)
from starVLA.model.modules.qwen_world_postcot.teachers import FrozenTargetModule


@pytest.fixture(autouse=True)
def small_threads():
    previous=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def memory(batch=2,width=16):
    return StudentMemory(torch.randn(batch,12,width,requires_grad=True),
        torch.randn(batch,4,width,requires_grad=True),torch.randn(batch,32,width,requires_grad=True),
        torch.ones(batch,12,dtype=torch.bool),torch.ones(batch,4,dtype=torch.bool),
        torch.ones(batch,32,dtype=torch.bool))


def test_readouts_current_only_gradient_contract():
    m=memory()
    dino=DINOReadout(memory_dim=16,width=16,layers=1,heads=4,channels=8)
    predicted=dino(m)
    assert predicted.shape==(2,3,9,16,8)
    target=torch.randn_like(predicted,requires_grad=True)
    valid=torch.ones(predicted.shape[:-1],dtype=torch.bool)
    valid_scene_mse(predicted,target,valid).backward()
    assert m.spatial.grad is not None and m.spatial.grad.abs().sum()>0
    assert m.global_memory.grad is None and m.predictive.grad is None and target.grad is None
    m=memory()
    future=FutureReadout(memory_dim=16,width=16,layers=1,heads=4,channels=8)
    query=sample_future_queries(2,24,generator=torch.Generator().manual_seed(1))
    predicted=future(m,query)
    assert predicted.shape==(2,24,8)
    predicted.square().mean().backward()
    assert all(value.grad is not None and value.grad.abs().sum()>0 for value in
               [m.spatial,m.global_memory,m.predictive])


def test_scene_mse_masked_nan_and_microbatch_equivalence():
    prediction=torch.tensor([[[1.],[3.]],[[2.],[9.]],[[7.],[8.]]],requires_grad=True)
    target=torch.tensor([[[0.],[float('nan')]],[[0.],[0.]],[[float('nan')],[float('nan')]]],requires_grad=True)
    valid=torch.tensor([[True,False],[True,True],[False,False]])
    full=valid_scene_mse(prediction,target,valid)
    assert full.item()==pytest.approx((1+(4+81)/2)/2)
    expected=torch.autograd.grad(full,prediction,retain_graph=True)[0]
    split=sum(valid_scene_mse(prediction[i:i+1],target[i:i+1],valid[i:i+1],scene_denominator=2) for i in range(3))
    got=torch.autograd.grad(split,prediction)[0]
    assert torch.equal(expected,got)
    empty=valid_scene_mse(prediction,target,torch.zeros_like(valid))
    assert empty.item()==0 and empty.requires_grad
    with pytest.raises(ValueError):valid_scene_mse(prediction,target,valid,scene_denominator=0)
    with pytest.raises(FloatingPointError):valid_scene_mse(prediction,target,torch.ones_like(valid))


def test_stratified_sampler_and_target_join():
    a=sample_future_queries(2,128,generator=torch.Generator().manual_seed(42))
    b=sample_future_queries(2,128,generator=torch.Generator().manual_seed(42))
    assert torch.equal(a,b)
    for row in a:
        assert len(torch.unique(row,dim=0))==128
        counts=torch.bincount(row[:,0]*4+row[:,1],minlength=12)
        assert counts.min()==10 and counts.max()==11
    targets=torch.arange(2*3*4*8*16*2).reshape(2,3,4,8,16,2)
    valid=torch.ones(targets.shape[:-1],dtype=torch.bool)
    gathered,mask=gather_future_targets(targets,valid,a)
    assert gathered.shape==(2,128,2) and mask.all()
    c,t,y,x=a[0,0].tolist()
    assert torch.equal(gathered[0,0],targets[0,c,t,y,x])
    with pytest.raises(ValueError):sample_future_queries(1,4,generator=torch.Generator())


def object_target(batch=2):
    boxes=torch.zeros(batch,3,8,requires_grad=True)
    with torch.no_grad():boxes[:,0]=torch.tensor([4.,2.,0.,4.,2.,1.5,0.,1.])
    return dict(boxes=boxes,classes=torch.zeros(batch,3,dtype=torch.long),
                valid=torch.tensor([[True,False,False],[False,False,False]])[:batch],
                coverage_valid=torch.zeros(batch,dtype=torch.bool))


def test_object_matching_no_invisible_negatives_and_valid_gradients():
    m=memory()
    readout=CurrentObjectReadout(memory_dim=16,width=16,layers=1,heads=4,num_queries=4)
    pred=readout(m)
    target=object_target()
    loss,terms=current_object_loss(pred,target,return_terms=True)
    assert terms['matched']==1 and torch.isfinite(loss)
    loss.backward()
    assert readout.boxes.weight.grad.abs().sum()>0 and readout.classes.weight.grad.abs().sum()>0
    assert m.spatial.grad.abs().sum()>0 and m.global_memory.grad.abs().sum()>0
    assert target['boxes'].grad is None and m.predictive.grad is None
    # A scene with no visible objects and unobserved background contributes no
    # artificial all-background supervision, while remaining graph-connected.
    empty={key:value[1:] for key,value in target.items()}
    loss=current_object_loss({key:value[1:] for key,value in pred.items()},empty)
    assert loss.item()==0 and loss.requires_grad


def test_object_full_update_denominators_match_microbatches():
    torch.manual_seed(5)
    pred=dict(existence_logits=torch.randn(2,4,requires_grad=True),
        class_logits=torch.randn(2,4,7,requires_grad=True),boxes=torch.randn(2,4,8,requires_grad=True))
    target=object_target()
    full=current_object_loss(pred,target)
    split=sum(current_object_loss({k:v[i:i+1] for k,v in pred.items()},
        {k:v[i:i+1] for k,v in target.items()},scene_denominators={
            'existence':1,'classification':1,'box':1}) for i in range(2))
    assert torch.allclose(full,split)
    actual=torch.autograd.grad(full,tuple(pred.values()),retain_graph=True)
    expected=torch.autograd.grad(split,tuple(pred.values()))
    assert all(torch.equal(a,b) for a,b in zip(actual,expected))


def test_physical_current_depth_and_visibility():
    points=torch.tensor([[[1.,1.,10.],[-1.,1.,-10.],[100.,1.,10.]]])
    transform=torch.eye(4)[None,None]
    intrinsics=torch.tensor([[100.,0.,50.],[0.,100.,50.],[0.,0.,1.]])[None,None]
    depths=current_depth_from_ego(points,transform)
    assert torch.equal(depths,torch.tensor([[[10.,-10.,10.]]]))
    valid=visible_current_centers(points,intrinsics,transform,(100,100))
    assert valid.tolist()==[[[True,False,False]]]


def test_actual_timestamp_tubelet_binding():
    times=(torch.arange(12)*500000+123456)[None]
    tubelets=validate_clip_timestamps(times)
    assert torch.equal(tubelets,times[:,4:].reshape(1,4,2))
    bad=times.clone();bad[0,6]+=250000
    with pytest.raises(ValueError):validate_clip_timestamps(bad)
    with pytest.raises(ValueError):validate_clip_timestamps(times.float())


def test_missing_future_is_masked_without_fabricated_timestamps():
    times=(torch.arange(12)*500000+123456)[None]
    times[0,7]=-1
    assert not clip_timing_valid(times).any()
    features=torch.zeros(1,3,4,8,16,512,dtype=torch.bfloat16)
    valid=torch.zeros(1,3,4,8,16,dtype=torch.bool)
    target=WAFutureTargetBatch(features,valid,times,times[:,4:].reshape(1,4,2),{})
    assert target.validate() is target
    valid[0,0]=True
    with pytest.raises(ValueError):target.validate()
    times[0,7]=-2
    with pytest.raises(ValueError):validate_clip_timestamps(times,allow_incomplete=True)


def payload(value=1.):
    return dict(dino=torch.tensor([[value,2.]],dtype=torch.float32),
                future=torch.tensor([[3.,4.]],dtype=torch.bfloat16),
                valid=torch.tensor([True,False]),timestamps=torch.arange(12,dtype=torch.int64))


def cache_identity():return dict(cohort='TRAIN',layout=tensor_layout(payload()),teachers='pinned-test-identity')


def test_lossless_cache_partial_roundtrip_and_role_hash_checks(tmp_path):
    writer=BinaryTeacherCacheWriter(tmp_path,cache_identity(),shard_records=1)
    metadata=dict(cohort='TRAIN',source_record_hash='observation-a')
    assert writer.add('a',payload(),metadata)
    assert not writer.add('a',payload(),metadata)
    with pytest.raises(FileNotFoundError):TeacherCacheReader(tmp_path,expected_cohort='TRAIN')
    with pytest.raises(ValueError):finalize_teacher_cache(tmp_path,['a','missing'])
    finalize_teacher_cache(tmp_path,['a'],verify_full_shards=True)
    reader=TeacherCacheReader(tmp_path,expected_cohort='TRAIN',max_open_shards=1)
    record=reader.get('a',source_record_hash='observation-a')
    assert all(torch.equal(record[k].view(torch.uint8),v.view(torch.uint8)) for k,v in payload().items())
    with pytest.raises(ValueError):reader.get('a',source_record_hash='wrong-observation')
    with pytest.raises(ValueError):TeacherCacheReader(tmp_path,expected_cohort='DEV')
    with pytest.raises(ValueError):TeacherCacheReader(tmp_path,expected_cohort='TRAIN',expected_identity='wrong')
    with pytest.raises(ValueError):writer.add('a',payload(9.),metadata)
    reader.close()


def test_cache_bounded_fd_lru_and_corruption(tmp_path):
    writer=BinaryTeacherCacheWriter(tmp_path,cache_identity(),shard_records=1)
    for token in ['a','b','c']:
        writer.add(token,payload(),dict(cohort='TRAIN',source_record_hash=token))
    writer.close();finalize_teacher_cache(tmp_path,['a','b','c'])
    reader=TeacherCacheReader(tmp_path,expected_cohort='TRAIN',max_open_shards=1)
    first=reader.get('a')['dino']
    reader.get('b');reader.get('c')
    assert len(reader._fds)==1 and torch.equal(first,payload()['dino'])
    row=reader.records['b'];path=tmp_path/row['file']
    # Even preserving stat metadata cannot bypass record content validation.
    original=path.stat()
    with path.open('r+b') as stream:stream.write(b'corrupt!')
    import os
    os.utime(path,ns=(original.st_atime_ns,original.st_mtime_ns))
    with pytest.raises(ValueError):reader.get('b')
    reader.close()


def test_producer_temporal_partition_preserves_exact_scene_coverage():
    from tools.qwen_world_postcot.teacher_cache import partition_producer_records
    rows=[dict(token=str(i),log='l'+str(i//10),timestamp=i*500000) for i in reversed(range(23))]
    before=list(rows)
    parts=[partition_producer_records(rows,rank,4,['1','21']) for rank in range(4)]
    assert sorted(row['token'] for part in parts for row in part)==sorted(row['token'] for row in rows)
    assert len({row['token'] for part in parts for row in part})==23 and rows==before
    for part in parts:
        first=[row['token'] for row in part if row['token'] in ['1','21']]
        assert [row['token'] for row in part[:len(first)]]==first
    with pytest.raises(ValueError):partition_producer_records(rows,0,4,['unknown'])


def test_canonical_clip_resolver_uses_real_logs_and_current_qwen_paths(tmp_path):
    import pickle
    from tools.qwen_world_postcot.teacher_cache import CanonicalClipResolver
    from starVLA.model.modules.qwen_world_postcot.teacher_targets import CAMERAS
    logs=tmp_path/'logs';logs.mkdir();sensor=tmp_path/'sensor';sensor.mkdir()
    frames=[]
    for i in range(12):
        cameras={}
        for camera in CAMERAS:
            relative=f'log/{camera}/{i}.jpg';path=sensor/relative
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'path-only-unit-contract')
            cameras[camera.upper()]=dict(data_path=relative)
        frames.append(dict(token=str(i),timestamp=1000000+i*500000,cams=cameras))
    with (logs/'log.pkl').open('wb') as f:pickle.dump(frames,f)
    row=dict(token='3',log='log',timestamp=frames[3]['timestamp'],
        images={camera:str(sensor/frames[3]['cams'][camera.upper()]['data_path']) for camera in CAMERAS},
        teacher_frames=[dict(offset=i-3,timestamp=frame['timestamp'],images={c:None for c in CAMERAS})
                        for i,frame in enumerate(frames)])
    resolver=CanonicalClipResolver(logs,sensor)
    result=resolver.resolve(row)
    assert result[3]['images']==row['images']
    assert all(all(frame['images'].values()) for frame in result)
    assert resolver.last_receipt['sha256']
    row['teacher_frames'][6]['timestamp']+=1
    with pytest.raises(ValueError):resolver.resolve(row)


def test_future_image_preprocessing_uses_official_cv2_without_unused_pil(tmp_path,monkeypatch):
    from PIL import Image
    from tools.qwen_world_postcot.teacher_cache import BoundedImageDecoder
    path=tmp_path/'frame.jpg';Image.new('RGB',(64,32),(21,43,65)).save(path)
    original=Image.open
    def unexpected(*args,**kwargs):raise AssertionError('Future WA preprocessing must not call PIL')
    monkeypatch.setattr(Image,'open',unexpected)
    decoder=BoundedImageDecoder(2)
    dino,wa,receipt=decoder.get(path,dino_required=False)
    assert dino is None and wa.shape==(3,256,512) and decoder.dino_decodes==0
    monkeypatch.setattr(Image,'open',original)
    current,wa_again,current_receipt=decoder.get(path,dino_required=True)
    assert current.shape==(3,288,512) and torch.isfinite(current).all()
    assert torch.equal(wa,wa_again) and receipt==current_receipt
    assert decoder.dino_decodes==1 and decoder.hits==1


def test_rgb_checkpoint_never_overwrites_weights_receipt_or_partial(tmp_path):
    from tools.qwen_world_postcot.train_rgb_decoder import save_checkpoint
    state=dict(step=1,identity='rgb-test',model={'weight':torch.tensor([1.])})
    path=tmp_path/'step_000001.pt';receipt=path.with_suffix('.receipt.json')
    save_checkpoint(path,state)
    original=(path.read_bytes(),receipt.read_bytes())
    with pytest.raises(FileExistsError):
        save_checkpoint(path,dict(state,model={'weight':torch.tensor([99.])}))
    assert (path.read_bytes(),receipt.read_bytes())==original
    orphan=tmp_path/'step_000002.pt'
    orphan.with_suffix('.receipt.json').write_text('diagnostic receipt')
    with pytest.raises(FileExistsError):save_checkpoint(orphan,state)
    assert not orphan.exists()
    interrupted=tmp_path/'step_000003.pt'
    partial=interrupted.with_name(interrupted.name+'.partial');partial.write_bytes(b'interrupted')
    with pytest.raises(FileExistsError):save_checkpoint(interrupted,state)
    assert partial.read_bytes()==b'interrupted' and not interrupted.exists()


def test_rgb_output_lock_excludes_another_process_and_releases(tmp_path):
    import subprocess
    import sys
    from tools.qwen_world_postcot.train_rgb_decoder import output_run_lock
    child="""import fcntl,sys
with open(sys.argv[1],'a') as stream:
    try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit(75)
"""
    lock=tmp_path/'.train_rgb_decoder.lock'
    with output_run_lock(tmp_path):
        assert subprocess.run([sys.executable,'-c',child,str(lock)],check=False).returncode==75
        with pytest.raises(RuntimeError,match='owns this output'):
            with output_run_lock(tmp_path):pass
    assert subprocess.run([sys.executable,'-c',child,str(lock)],check=False).returncode==0


@pytest.mark.parametrize('saved_step',[1473,2000,2001])
def test_rgb_terminal_resume_is_read_only_and_over_budget_rejected(tmp_path,monkeypatch,capsys,saved_step):
    # A CPU-only lifecycle fixture: no teacher/model forward is claimed here.
    import json
    import sys
    from tools.qwen_world_postcot import train_rgb_decoder as module
    root=tmp_path/'run';root.mkdir()
    population=tmp_path/'population.json';population.write_text('{}')
    checkpoint=root/f'step_{saved_step:06d}.pt'
    module.save_checkpoint(checkpoint,dict(step=saved_step,identity='bound-run'))
    if saved_step==1473:
        module.save_checkpoint(root/'step_002000.pt',dict(step=2000,identity='bound-run'))
    (root/'TRAINING_TOKENS.json').write_text('["scene"]')
    (root/'resolved.json').write_text('{"original_config":true}')
    (root/'RUN_STATUS.json').write_text('{"status":"COMPLETE","last_l1":0.057}')
    (root/'.train_rgb_decoder.lock').touch()
    before={path.name:(path.read_bytes(),path.stat().st_mtime_ns) for path in root.iterdir()}
    reader=SimpleNamespace(records={'scene':{}},identity={'sha256':'teacher'},closed=False)
    def close():reader.closed=True
    reader.close=close
    monkeypatch.setattr(module,'load_population',lambda _:({'cohort':'TRAIN'},[{'token':'scene'}]))
    monkeypatch.setattr(module,'TeacherCacheReader',lambda *args,**kwargs:reader)
    monkeypatch.setattr(module,'identity_hash',lambda _:'bound-run')
    def forbidden(*args,**kwargs):raise AssertionError('Terminal resume must not initialize/train a model or RNG')
    monkeypatch.setattr(module,'DINORGBDecoder',forbidden)
    monkeypatch.setattr(module.random,'seed',forbidden)
    monkeypatch.setattr(sys,'argv',['train_rgb_decoder','--population',str(population),
        '--teacher-cache',str(tmp_path/'cache'),'--output',str(root),'--device','cpu',
        '--updates','2000','--resume',str(checkpoint)])
    rng=torch.get_rng_state().clone()
    if saved_step==2000:
        module.main()
        result=json.loads(capsys.readouterr().out)
        assert result['already_complete'] and result['optimizer_updates_performed']==0
        assert result['artifacts_rewritten'] is False
    elif saved_step>2000:
        with pytest.raises(ValueError,match='exceeds.*budget'):module.main()
    else:
        with pytest.raises(ValueError,match='Obsolete.*later sealed'):module.main()
    assert reader.closed and torch.equal(torch.get_rng_state(),rng)
    assert {path.name:(path.read_bytes(),path.stat().st_mtime_ns) for path in root.iterdir()}==before


def test_frozen_teacher_outer_train_keeps_eval():
    teacher=FrozenTargetModule();teacher.child=torch.nn.Dropout(.9)
    teacher.train(True)
    assert not teacher.training and not teacher.child.training


def test_rgb_decoder_deterministic_dimensions_and_gradients():
    torch.manual_seed(1)
    decoder=DINORGBDecoder(feature_dim=8,channels=(8,8,8,8,8,8))
    features=torch.randn(1,9,16,8,requires_grad=True)
    a=decoder(features);b=decoder(features)
    assert a.shape==(1,3,288,512) and torch.equal(a,b)
    loss=rgb_reconstruction_loss(a,torch.rand_like(a,requires_grad=True),torch.tensor([True]))
    loss.backward()
    assert features.grad.abs().sum()>0 and decoder.input[0].weight.grad.abs().sum()>0
    with pytest.raises(ValueError):decoder(torch.randn(1,6,8,8))
