import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / 'tests/st_world'))
from test_history import real_fixture
from resume_records import CommittedHistoryRecords, resume_original_writer
from tools.st_world_policy import prepare_future_metadata as bounded, prepare_history as original
from tools.st_world_policy import prepare_causal_history as writer
from tools.st_world_policy.history_shards import record_bytes
from starVLA.dataloader import st_world_history as history, st_world_schema as schema


def package(tmp_path):
    images = tmp_path / 'images'
    images.mkdir()
    frames, current = real_fixture(images)
    record = history.build_history_record(frames, 't2', 'log', current, [images], raw_log_sha256='a'*64)
    root = tmp_path / 'metadata'
    (root / 'history_shards').mkdir(parents=True)
    name = '000000_' + hashlib.sha256(b'log').hexdigest()[:16] + '.jsonseq'
    raw = record_bytes(record)
    (root / 'history_shards' / name).write_bytes(raw)
    (root / 'SELECTED_SCENES.json').write_text(json.dumps({'scenes':[
        {'token':'t2','log':'log'}, {'token':'t3','log':'log'}]}))
    census = dict(schema='st_partial_history_metadata_census_v1',
        status='CENSUS_COMPLETE_NOT_INPUT_QUALIFICATION', root=str(root),
        source_images_logs_models_read=False, shards=1, scenes=1, expected_scenes=2,
        rows=[dict(name=name, bytes=len(raw), scenes=1, sha256=hashlib.sha256(raw).hexdigest())])
    limits=dict(max_seconds=60, max_rss_bytes=2<<30, max_metadata_bytes=16<<20,
        max_source_read_bytes=16<<20, read_mib_per_second=32)
    return root, images, frames, current, record, census, limits


def cache(p):
    root, images, frames, current, record, census, limits = p
    budget = bounded.PreparationBudget(root, limits)
    return CommittedHistoryRecords(root, budget, census, bounded=bounded,
        history=history, schema=schema, record_bytes=record_bytes)


def get(c, p, **kwargs):
    root, images, frames, current, record, census, limits = p
    args=dict(offsets=(-1.,-.5), tolerance=.05, sensor_roots=[images])
    args.update(kwargs)
    return c.get('t2','log',current,'a'*64,**args)


def test_committed_record_roundtrip_rehash_once_without_decode(tmp_path):
    p=package(tmp_path);c=cache(p)
    with patch('PIL.Image.open',side_effect=AssertionError('committed image decode forbidden')):
        assert get(c,p)==p[4]
        after=c.rehashed_image_bytes
        assert after>0 and get(c,p)==p[4]
        assert c.rehashed_image_bytes==after
        assert c.summary()['rehashed_unique_image_count']==6


def test_same_path_same_size_restored_mtime_wrong_bytes_rejected(tmp_path):
    p=package(tmp_path);c=cache(p);image=Path(p[4]['frames'][0]['views'][0]['image']['path'])
    before=image.stat();raw=image.read_bytes();image.write_bytes(bytes([raw[0]^1])+raw[1:])
    os.utime(image,ns=(before.st_atime_ns,before.st_mtime_ns))
    with pytest.raises(ValueError,match='SHA/version'):get(c,p)


def test_after_first_hash_same_content_replacement_rejected_by_inode(tmp_path):
    p=package(tmp_path);c=cache(p);get(c,p)
    image=Path(p[4]['frames'][0]['views'][0]['image']['path']);before=image.stat()
    replacement=image.with_suffix('.replacement');replacement.write_bytes(image.read_bytes())
    os.utime(replacement,ns=(before.st_atime_ns,before.st_mtime_ns));replacement.replace(image)
    with pytest.raises(ValueError,match='after this invocation'):get(c,p)


@pytest.mark.parametrize('field',['raw','current','tolerance','offset'])
def test_source_and_causal_contract_mismatch_rejected(tmp_path,field):
    p=package(tmp_path);c=cache(p)
    current=copy.deepcopy(p[3]);raw='a'*64;offsets=(-1.,-.5);tolerance=.05
    if field=='raw':raw='b'*64
    if field=='current':current['timestamp']+=1
    if field=='tolerance':tolerance=.04
    if field=='offset':offsets=(-1.,)
    with pytest.raises(ValueError,match='source/current/causal'):
        c.get('t2','log',current,raw,offsets=offsets,tolerance=tolerance,sensor_roots=[p[1]])


@pytest.mark.parametrize('kind',['checksum','missing','extra','selfhash','prefix'])
def test_bad_committed_metadata_rejected(tmp_path,kind):
    p=list(package(tmp_path));root,_,_,_,record,census,_=p
    shard=root/'history_shards'/census['rows'][0]['name']
    if kind=='checksum':shard.write_bytes(shard.read_bytes().replace(b't2',b'x2',1))
    if kind=='missing':shard.unlink()
    if kind=='extra':(shard.parent/'orphan.partial').write_bytes(b'x')
    if kind=='selfhash':
        record['current_timestamp_us']+=1;raw=record_bytes(record);shard.write_bytes(raw)
        census['rows'][0].update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    if kind=='prefix':
        (root/'SELECTED_SCENES.json').write_text(json.dumps({'scenes':[{'token':'t3','log':'log'},{'token':'t2','log':'log'}]}))
    with pytest.raises(ValueError):cache(p)


def test_only_covered_scene_reused_and_native_new_scene_unchanged(tmp_path):
    p=package(tmp_path);root,images,frames,current,record,census,limits=p
    command=['build','--student-root',str(tmp_path),'--output',str(root),'--split','train',
        '--max-scenes','2','--record-layout','packed_log_v1']
    current3=dict(current,token='t3',timestamp=1500000,
        image_paths=[str(images/f'3_{v}.png') for v in history.VIEWS])
    expected_new=original.build_history_record(frames,'t3','log',current3,[images],raw_log_sha256='a'*64)
    original_budget=bounded.PreparationBudget;original_build=original.build_history_record
    with resume_original_writer(writer=writer,original=original,bounded=bounded,history=history,
            schema=schema,record_bytes=record_bytes,census=census,expected_command_args=command) as state:
        assert writer.parser().parse_args(command).resume is True
        with pytest.raises(ValueError,match='registered builder'):writer.parser().parse_args(command+['--resume'])
        bounded.PreparationBudget(root,limits)
        assert original.build_history_record(frames,'t2','log',current,[images],raw_log_sha256='a'*64)==record
        assert original.build_history_record(frames,'t3','log',current3,[images],raw_log_sha256='a'*64)==expected_new
        assert state['records'].reused==1 and state['records'].new==1
    assert bounded.PreparationBudget is original_budget and original.build_history_record is original_build


def test_failure_envelope_retains_actual_cause_and_original_controller(tmp_path):
    import run_resume_once as bridge
    observed={'returncode':2,'failure':'storage','cleanup':{'live_owned_remaining':0},
        'child_accounting':{'error':{'type':'StorageBudgetError','message':'disk'},'counters':{'rchar':23}},
        'read_accounting':{'charged':42}}
    assert bridge.failure_summary(observed)['child_error']['type']=='StorageBudgetError'
    m=bridge.load_controller()
    source=m.bound_source(m.source_bytes(m.ORIGINAL,m.inherited_binding().PINS[str(m.ORIGINAL.relative_to(m.RUNTIME))]))
    assert 'BOUND_RESUME_CALL(m,guard,reg)' in source and 'BOUND_PRESERVE_START(start)' in source
    assert 'core.monitor(' in source and 'CONSERVATIVE_FULL_ALLOWANCE_ON_EXCEPTION' in source
    assert "'preserved_child_failure':BOUND_FAILURE_SUMMARY(observed)" in source
