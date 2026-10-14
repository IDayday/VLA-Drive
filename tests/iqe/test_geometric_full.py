from types import SimpleNamespace
from pathlib import Path
import json
import pytest
import torch
from iqe.io import atomic_json, digest, file_hash
from iqe.geometric_readout import calibration_bank as cb
from iqe.geometric_readout.full import Config, calibrated_fetch, collective_run_lock


def fixture(tmp_path, monkeypatch):
    source=SimpleNamespace(root=tmp_path/'source',identity='locked-test-source')
    atomic_json(source.root/'BASE_BINDING.json',{'test_fixture':True})
    roles={r:[] for r in cb.ROLES}
    for i,role in enumerate(cb.ROLES):
        images=[]
        for j in range(3):
            p=tmp_path/f'image{i}_{j}.bin';p.write_bytes(bytes([i,j]));images.append(str(p))
        path=tmp_path/f'observation{i}.json'
        record=dict(token=f's{i}',log=f'log{i}',image_paths=images,current_calibration=dict(
            intrinsics=torch.eye(3).expand(3,3,3).tolist(),
            extrinsics=torch.eye(4).expand(3,4,4).tolist(),distortion=[[0.]*5]*3))
        atomic_json(path,record)
        scene=SimpleNamespace(scene_id=record['token'],source_log_id=record['log'],observation_ref=str(path),
            observation_hash=digest({'record':record,'images':{p:file_hash(p) for p in images}}))
        roles[role]=[scene]
    monkeypatch.setattr(cb,'scenes_by_role',lambda _:roles)
    return source,roles


def test_calibration_complete_resume_content_integrity_and_no_image_decode(tmp_path,monkeypatch):
    source,roles=fixture(tmp_path,monkeypatch);root=tmp_path/'cache'
    a=cb.prepare(source,root,2);checksum=file_hash(root/'MANIFEST.json')
    assert a['records']==4 and a['status']=='COMPLETE'
    # A repeated builder reuses complete log shards and performs no fresh reader computation.
    monkeypatch.setattr(cb.CalibrationReader,'read',lambda *_:pytest.fail('recomputed committed shard'))
    assert cb.prepare(source,root,2)==a and file_hash(root/'MANIFEST.json')==checksum
    reader=cb.PreparedCalibrationReader(source,root);s=roles['scorer_fit'][0]
    x=reader.batch([s,s],'cpu');assert x['camera_intrinsics'].shape==(2,3,3,3)
    from PIL import Image
    monkeypatch.setattr(Image,'open',lambda *_:pytest.fail('decoded input for calibration hit'))
    assert torch.equal(x['camera_to_ego'],reader.batch([s,s],'cpu')['camera_to_ego'])
    path=Path(next(iter(reader.records[s.scene_id]['images'])))
    path.write_bytes(b'changed current image')
    with pytest.raises(ValueError,match='image changed'):reader.read(s)


def test_partial_calibration_shards_resume_and_wrong_observation_rejected(tmp_path,monkeypatch):
    source,roles=fixture(tmp_path,monkeypatch);root=tmp_path/'cache'
    original=cb.CalibrationReader.read
    def fail(self,scene):
        if scene.scene_id=='s2':raise RuntimeError('injected interruption')
        return original(self,scene)
    monkeypatch.setattr(cb.CalibrationReader,'read',fail)
    with pytest.raises(RuntimeError):cb.prepare(source,root,1)
    assert not (root/'MANIFEST.json').exists() and list((root/'logs').glob('*.json'))
    monkeypatch.setattr(cb.CalibrationReader,'read',original);cb.prepare(source,root,1)
    r=cb.PreparedCalibrationReader(source,root);s=roles['scorer_fit'][0]
    wrong=SimpleNamespace(**(vars(s)|{'observation_hash':'wrong'}))
    with pytest.raises(ValueError,match='observation mismatch'):r.read(wrong)
    record=json.loads(Path(s.observation_ref).read_text());record['current_calibration']['intrinsics'][0][0][0]=99.
    atomic_json(s.observation_ref,record)
    with pytest.raises(ValueError,match='content changed'):r.read(s)


def test_calibrated_fetch_preserves_order_and_padding():
    scenes=[SimpleNamespace(scene_id=x) for x in ('a','b')]
    observed=[]
    class Reader:
        def batch(self,rows,device):
            observed.extend(s.scene_id for s in rows);return {'camera_intrinsics':torch.ones(len(rows),3,3,3)}
    marker=torch.tensor([True,False])
    def ordinary(entries,device):return {'context':{'world':torch.ones(2,1,3)},'natural':marker}
    f=calibrated_fetch(ordinary,scenes,Reader())
    value=f([{'scene_id':'b'},{'scene_id':'a','_padding':True}],'cpu')
    assert observed==['b','a'] and value['natural'] is marker
    assert set(value['context'])=={'world','camera_intrinsics'}


def test_run_lock_fails_fast_and_config_rejects_unregistered_fields(tmp_path):
    with collective_run_lock(tmp_path/'run.lock',0):
        with pytest.raises(ValueError,match='Resource temporarily unavailable'):
            with collective_run_lock(tmp_path/'run.lock',0):pytest.fail('duplicate admitted')
    kwargs=dict(experiment_id='test',source_campaign='/source',output_root='/output',
                calibration_bank='/calibration',geometry_audit='/audit',diagnostic_root='/diagnostic')
    assert Config(**kwargs).root==Path('/output')
    with pytest.raises(TypeError):Config(**kwargs,load_external_weights=True)
    with pytest.raises(ValueError):Config(**(kwargs|{'output_root':'/source'}))


def test_online_camera_side_channel_never_enters_vla_and_hooks_cleanup(monkeypatch):
    from iqe.geometric_readout.bundle import GeometricResearchSystem,CAMERAS
    from iqe.specialist_study.evaluation import ResearchSystem
    seen=[]
    class Head(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.ones(()))
        def forward(self,context,trajectory):
            assert set(context)==CAMERAS|{'world'}
            return context
    head=Head();system=GeometricResearchSystem(None,[torch.nn.Identity()],head)
    obs=[{'image':'unchanged reference'}]
    def base(self,observations,**kwargs):
        assert observations is obs and observations==[{'image':'unchanged reference'}]
        seen.append(1);return self.scorer({'world':torch.ones(1,2,3)},torch.zeros(1,2,8,3))
    monkeypatch.setattr(ResearchSystem,'shared_forward',base)
    camera={k:torch.ones(1,3,3,3,requires_grad=True) for k in CAMERAS}
    result=system.shared_forward(obs,calibration=camera)
    assert len(seen)==1 and all(not result[k].requires_grad for k in CAMERAS)
    assert not head._forward_pre_hooks
    with pytest.raises(ValueError):system.shared_forward(obs,calibration=camera|{'GT':torch.zeros(1)})
    assert not head._forward_pre_hooks
