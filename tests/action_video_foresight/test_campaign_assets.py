import json
from tools.action_video_foresight.run_formal_campaign import complete_targets
import pytest
from pathlib import Path
from starVLA.model.modules.vehicle_joint.initialization import file_sha256
from tools.action_video_foresight.stage_native_targets import copy_chunk
from tools.action_video_foresight.evaluate_milestone import checked_ego_summary


def test_formal_controller_waits_for_complete_population_and_rejects_changed_identity(tmp_path):
    (tmp_path/'identity.json').write_text(json.dumps({'identity':'fixed'}))
    assert not complete_targets(tmp_path,'fixed',101592)
    (tmp_path/'COMPLETE.json').write_text(json.dumps({'identity':'fixed','scenes':64,'chunks':1}))
    with pytest.raises(ValueError,match='different population'):
        complete_targets(tmp_path,'fixed',101592)
    (tmp_path/'COMPLETE.json').write_text(json.dumps({'identity':'fixed','scenes':101592,'chunks':1}))
    assert not complete_targets(tmp_path,'fixed',101592)
    (tmp_path/'chunk_000000.json').write_text('{}')
    (tmp_path/'chunk_000000.safetensors').write_bytes(b'fixture; never a training asset')
    assert complete_targets(tmp_path,'fixed',101592)
    with pytest.raises(ValueError,match='identity changed'):
        complete_targets(tmp_path,'other',101592)


def test_native_replica_rejects_foreign_or_corrupt_data(tmp_path):
    source, output = tmp_path/'source', tmp_path/'local'
    source.mkdir(); output.mkdir()
    entity = source/'chunk_000000.safetensors'
    entity.write_bytes(b'copy-integrity fixture; not a training target')
    meta = {'identity':'native-video', 'sha256':file_sha256(entity), 'bytes':entity.stat().st_size}
    (source/'chunk_000000.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='identity changed'):
        copy_chunk(source, output, 0, 'dino-sequence')
    copy_chunk(source, output, 0, 'native-video')
    assert file_sha256(output/entity.name) == meta['sha256']
    (output/entity.name).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='corrupt'):
        copy_chunk(source, output, 0, 'native-video')


def test_ego_evaluation_cannot_resume_past_failed_or_incomplete_summary(tmp_path):
    with pytest.raises(FileNotFoundError):
        checked_ego_summary(tmp_path,1696)
    record={'valid':False,'failed':1,'scenes':1696,'groups':{'all':{'ADE':None,'FDE':None,'yaw_MAE_rad':None}}}
    (tmp_path/'summary.json').write_text(json.dumps(record))
    with pytest.raises(RuntimeError,match='Incomplete'):
        checked_ego_summary(tmp_path,1696)
    record.update(valid=True,failed=0);record['groups']['all'].update(ADE=.5,FDE=1.,yaw_MAE_rad=.1)
    (tmp_path/'summary.json').write_text(json.dumps(record))
    assert checked_ego_summary(tmp_path,1696)['valid']
