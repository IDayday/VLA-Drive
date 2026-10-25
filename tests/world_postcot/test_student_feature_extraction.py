"""Narrow offline extraction contracts; these are not real IL quality results."""
import copy
import json
from types import SimpleNamespace

from PIL import Image
import pytest
import torch
from torch import nn

from starVLA.rl.qwen_world_postcot.contracts import file_sha, fingerprint
from tools.qwen_world_postcot import extract_student_features as extract


class CurrentData:
    def __init__(self, root):
        image=root/'current.png';Image.new('RGB',(8,8),(10,20,30)).save(image)
        self.tokens=['scene'];self.manifest=dict(cohort='TRAIN')
        self.records={'scene':dict(images={camera:str(image) for camera in extract.CAMERAS},
            source_record_hash='current-observation',timestamp=1234,proprio=[0.]*8,navigation='left',
            requested_GT='MUST_NOT_BE_READ',future='MUST_NOT_BE_READ')}
    def observation(self, token):
        row=self.records[token]
        return dict(images={camera:Image.open(path).convert('RGB') for camera,path in row['images'].items()},
                    proprio=row['proprio'],navigation=row['navigation'])
    def target(self, token):
        raise AssertionError('Extraction must not access GT')


class CurrentReadout(nn.Module):
    def __init__(self):
        super().__init__();self.projection=nn.Linear(8,1024);self.calls=0
    def forward(self, ego):
        self.calls+=1
        return self.projection(ego)[:,None,None,None,:].expand(-1,3,9,16,-1)


class CurrentModel(nn.Module):
    def __init__(self):
        super().__init__();self.dino_readout=CurrentReadout();self.encodes=0
    def encode_condition(self, batch, ego):
        self.encodes+=1;assert set(batch)=={'current_only'}
        return ego
    def forward(self,*args,**kwargs):
        raise AssertionError('Do not run the full training or actor forward')


def test_bounded_current_only_features_and_immutable_visualizer_artifact(tmp_path):
    data=CurrentData(tmp_path);model=CurrentModel().eval()
    def processor(observations):
        assert len(observations)==1 and set(observations[0])=={'images','proprio','navigation'}
        return dict(current_only=True)
    result=extract.extract_current(model,data,processor,['scene'],torch.device('cpu'))
    assert model.encodes==1 and model.dino_readout.calls==1
    feature=result['features']['scene']
    assert feature.shape==extract.FEATURE_SHAPE and feature.dtype==torch.float32
    assert result['readout_compute_dtypes']['scene']=='torch.bfloat16'
    assert not feature.requires_grad and result['valid']['scene'].all()
    data.records['scene']['requested_GT']='CHANGED';data.records['scene']['future']='CHANGED'
    again=extract.extract_current(model,data,processor,['scene'],torch.device('cpu'))
    assert torch.equal(feature,again['features']['scene'])
    for tokens,cohort in [([], 'TRAIN'),(['scene']*2,'TRAIN'),([str(i) for i in range(17)],'TRAIN'),
                           (['unknown'],'TRAIN'),(['scene'],'NAVTEST')]:
        with pytest.raises(ValueError):extract.validate_tokens(tokens,data,cohort)
    artifact=dict(teacher_identity='sealed-teacher',checkpoint=dict(model_sha256='trained-model'),
                  metadata=dict(schema=extract.SCHEMA,current_only=True),**result)
    output=tmp_path/'student.pt';receipt=extract.publish(output,artifact)
    restored=torch.load(output,weights_only=True)
    assert restored['teacher_identity']=='sealed-teacher'
    assert torch.equal(restored['features']['scene'],feature)
    assert receipt['status']=='COMPLETE' and receipt['sha256']==file_sha(output)
    saved=json.loads(output.with_suffix('.receipt.json').read_text());seal=saved.pop('receipt_sha256')
    assert fingerprint(saved)==seal
    with pytest.raises(FileExistsError):extract.publish(output,artifact)


def test_native_il_loader_is_strict_fp32_and_preserves_bound_identity(monkeypatch):
    class Full(nn.Module):
        def __init__(self):
            super().__init__();self.qwen=nn.Linear(2,2);self.dino_readout=nn.Linear(2,2)
    info=dict(path='/unit-checkpoint',seal=dict(model_file='native/model.pt',updates=1000),
        identity=dict(schema='world_postcot_full_IL_v1',uses_rl=False,uses_metric_supervision=True,
                      train_heads=True,model_config={'unit':True},qwen_config={'attn_implementation':'sdpa'}))
    before=copy.deepcopy(info)
    monkeypatch.setattr(extract,'inspect_checkpoint',lambda *args,**kwargs:info)
    weights=Full().state_dict()
    monkeypatch.setattr(extract.torch,'load',lambda *args,**kwargs:dict(module=weights))
    def constructor(path,*,config,checkpoint_config,train_heads):
        assert train_heads is True
        config.pop('unit');checkpoint_config.pop('attn_implementation')
        return Full()
    monkeypatch.setattr(extract,'QwenWorldPostCoT',constructor)
    model,returned=extract.load_il_with_readout('/unit-checkpoint','cpu')
    assert returned==before and all(not p.requires_grad for p in model.parameters())
    assert not model.training
    weights.pop('dino_readout.weight')
    with pytest.raises(RuntimeError,match='Missing key'):extract.load_il_with_readout('/unit-checkpoint','cpu')
    weights=Full().state_dict();weights['dino_readout.weight']=weights['dino_readout.weight'].bfloat16()
    with pytest.raises(ValueError,match='FP32'):extract.load_il_with_readout('/unit-checkpoint','cpu')
    info['identity']['uses_rl']=True
    with pytest.raises(ValueError,match='IL checkpoint'):extract.load_il_with_readout('/unit-checkpoint','cpu')


def test_processor_and_teacher_metadata_bindings_without_teacher_tensor_access(tmp_path,monkeypatch):
    processor=tmp_path/'processor';processor.mkdir()
    files={}
    for name in extract.PROCESSOR_FILES:
        (processor/name).write_text('{}');files[name]=file_sha(processor/name)
    info=dict(identity=dict(initialization=dict(qwen=dict(receipt=dict(path=str(processor),files=files,
        repository='registered-Qwen',revision='fixed-revision')))))
    assert extract.processor_binding(info)[1]['files']==files
    (processor/extract.PROCESSOR_FILES[0]).write_text('changed')
    with pytest.raises(ValueError,match='Processor'):extract.processor_binding(info)

    trained=tmp_path/'TRAIN';trained.mkdir();dev=tmp_path/'DEV';dev.mkdir()
    dino=dict(width=1024,grid=[9,16],weight_sha256='real-receipt-identity')
    def write_identity(root,cohort,population):
        value=dict(cohort=cohort,dino=dino,camera_names=list(extract.CAMERAS),population_sha256=population,
                   records_sha256='records',layout=dict(fields=dict(dino=dict(shape=list(extract.FEATURE_SHAPE)))))
        value['sha256']=fingerprint(value);(root/'IDENTITY.json').write_text(json.dumps(value));return value
    trained_identity=write_identity(trained,'TRAIN','train-data');target=write_identity(dev,'DEV','dev-data')
    (trained/'manifest.json').write_text(json.dumps(dict(role_partition_sha256='same-partition')))
    (dev/'MANIFEST.json').write_text('metadata-only-test')
    info['identity'].update(config=dict(teacher_cache=str(trained),training_data=str(trained)),
        teacher_identity_sha256=file_sha(trained/'IDENTITY.json'),data_sha256=file_sha(trained/'manifest.json'))
    data=SimpleNamespace(sha='dev-data',manifest=dict(cohort='DEV',records_sha256='records',role_partition_sha256='same-partition'),
        records={'scene':dict(source_record_hash='current')})
    class MetadataReader:
        def __init__(self,*args,**kwargs):
            self.identity=target;self.records={'scene':dict(metadata=dict(source_record_hash='current'))}
        def get(self,*args,**kwargs):raise AssertionError('No teacher tensor access is needed')
        def close(self):pass
    monkeypatch.setattr(extract,'TeacherCacheReader',MetadataReader)
    bound=extract.teacher_binding(info,data,dev,['scene'])
    assert bound['teacher_identity']==target['sha256'] and bound['training_teacher_identity']==trained_identity['sha256']
    target['population_sha256']='another-data'
    with pytest.raises(ValueError,match='spatial contract'):extract.teacher_binding(info,data,dev,['scene'])
