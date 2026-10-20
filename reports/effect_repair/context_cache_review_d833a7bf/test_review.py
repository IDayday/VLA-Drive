"""Independent CPU receipt/config tests. Payloads below are explicit fixtures."""
from copy import deepcopy
from pathlib import Path
import sys,json
import pytest,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from tools.action_effect import common as c
from tools.action_effect import collect_base_trajectories as col


def sealed(value):
    value=deepcopy(value);value.pop('identity',None);value['identity']=c.digest(value);return value


@pytest.mark.parametrize('change',[
 lambda x:x['g0'].update(workers=-1),
 lambda x:x['training'].update(imaginary_loss=99),
 lambda x:x['resources'].update(minimum_free_bytes=-1),
 lambda x:x.update(seed='wrong'),
 lambda x:x['training'].update(learning_rate=float('nan')),
 lambda x:x['g0'].update(candidate_count=64),
 lambda x:x['g0'].update(navsim_root='relative'),
 lambda x:x['training'].update(training_seeds=[42,42]),
 lambda x:x['training'].update(hidden=255),
])
def test_nested_config_negative_cases_rejected(tmp_path,change):
    cfg=yaml.safe_load(Path('configs/action_effect/main.yaml').read_text());change(cfg)
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):c.load_config(path)


def fixture(tmp_path):
    entries=[{'scene_id':'synthetic_a','source_log':'log_A','effect_role':'fit','observation_hash':'1'*64},
             {'scene_id':'synthetic_b','source_log':'log_B','effect_role':'validation','observation_hash':'2'*64}]
    cohort=tmp_path/'COHORT.json';c.write_json(cohort,sealed({'entries':entries}))
    binding={'identity':'a'*64,'cohort':c.artifact(cohort),'base_checkpoint_binding_hash':'b'*64}
    tag=c.digest([binding['identity'],[x['scene_id'] for x in entries]])
    shard=tmp_path/(tag+'.pt');shard.write_bytes(b'Explicit synthetic payload; no real tensor or pretrained weights')
    artifact=c.artifact(shard)
    rows=[{'scene_id':e['scene_id'],'source_log':e['source_log'],'role':e['effect_role'],
           'observation_hash':e['observation_hash'],'shard_offset':i,'shard':artifact,
           'raw_hash':'c'*64,'physical_hash':'d'*64} for i,e in enumerate(entries)]
    receipt=sealed({'schema':'action_effect_context_shard_v1','binding_identity':binding['identity'],
             'base_binding_hash':binding['base_checkpoint_binding_hash'],'single_encoder_per_scene':True,
             'entries':rows,'shard':artifact})
    c.write_json(shard.with_suffix('.COMPLETE.json'),receipt)
    head=tmp_path/'head.pt';head.write_bytes(b'Explicit synthetic head artifact; no tensors')
    head_receipt=tmp_path/'head.COMPLETE.json'
    c.write_json(head_receipt,{'schema':'action_effect_original_action_head_v1','artifact':c.artifact(head),
             'base_binding_hash':binding['base_checkpoint_binding_hash'],'strict_original_master':True})
    manifest=sealed({'schema':'action_effect_context_manifest_v1','status':'COMPLETE',
             'binding_identity':binding['identity'],'dtype':'float32','max_samples':2,
             'single_encoder_per_scene':True,'cohort':binding['cohort'],'entries':rows,
             'shards':[artifact],'head_receipt':c.artifact(head_receipt)})
    return binding,entries,receipt,manifest


def test_current_main_yaml_valid():
    assert c.load_config('configs/action_effect/main.yaml')['pilot_scenes']==64


def test_exact_complete_fixture_reuse_and_no_gpu_required(tmp_path,monkeypatch):
    binding,entries,receipt,manifest=fixture(tmp_path)
    assert col.validate_context_manifest(manifest,binding,entries)==manifest
    out=tmp_path/'contexts';out.mkdir();c.write_json(out/'MANIFEST_000002.json',manifest)
    monkeypatch.setattr(col,'prepare',lambda cfg:binding)
    result=col.collect({'artifact_root':str(tmp_path)},max_samples=2,device='cpu')
    assert result['reused'] and result['scenes']==2


@pytest.mark.parametrize('change',[
 lambda x:x.update(status='PARTIAL'),
 lambda x:x.update(entries=[]),
 lambda x:x.update(shards=[]),
 lambda x:x.update(dtype='bfloat16'),
 lambda x:x.update(max_samples=1),
 lambda x:x['entries'].reverse(),
 lambda x:x['entries'][0].update(scene_id='foreign'),
 lambda x:x['head_receipt'].update(sha256='0'*64),
])
def test_incomplete_or_mismatched_manifest_rejected_even_if_resealed(tmp_path,change):
    binding,entries,receipt,m=fixture(tmp_path);change(m);m=sealed(m)
    with pytest.raises((ValueError,KeyError)):col.validate_context_manifest(m,binding,entries)


@pytest.mark.parametrize('change',[
 lambda x:x.update(base_binding_hash='0'*64),
 lambda x:x.update(binding_identity='0'*64),
 lambda x:x.update(single_encoder_per_scene=False),
 lambda x:x['entries'][0].update(role='validation'),
 lambda x:x['entries'][0].update(shard_offset=1),
 lambda x:x['entries'][0].update(observation_hash='0'*64),
 lambda x:x['entries'][0].update(source_log='other_log'),
 lambda x:x['entries'].reverse(),
])
def test_shard_identity_order_role_and_join_rejected(tmp_path,change):
    binding,entries,receipt,m=fixture(tmp_path);change(receipt);receipt=sealed(receipt)
    with pytest.raises(ValueError):col.validate_shard(receipt,binding,entries)


def test_changed_shard_bytes_and_head_bytes_rejected(tmp_path):
    binding,entries,receipt,m=fixture(tmp_path)
    shard=Path(receipt['shard']['path']);shard.write_bytes(b'changed')
    with pytest.raises(ValueError):col.validate_context_manifest(m,binding,entries)
    # Independent new fixture: head-body checksum is not inferred from path.
    other=tmp_path/'other';other.mkdir();binding,entries,receipt,m=fixture(other)
    head=other/'head.pt';head.write_bytes(b'changed')
    with pytest.raises(ValueError):col.validate_context_manifest(m,binding,entries)


def test_historical_empty_manifest_counterexample_now_rejected(tmp_path,monkeypatch):
    binding,entries,receipt,m=fixture(tmp_path)
    out=tmp_path/'contexts';out.mkdir()
    c.write_json(out/'MANIFEST_000002.json',{'binding_identity':'a'*64,'shards':[]})
    monkeypatch.setattr(col,'prepare',lambda cfg:binding)
    with pytest.raises(ValueError):col.collect({'artifact_root':str(tmp_path)},max_samples=2,device='cpu')


def test_wrong_receipt_identity_rejected_before_dependency_loading():
    with pytest.raises(ValueError):c.verify_identity({'identity':'wrong','cohort':{'path':'missing'}})
