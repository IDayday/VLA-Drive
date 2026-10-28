import json
from types import SimpleNamespace
import numpy as np
import pytest
from starVLA.rl.qwen_world_postcot.contracts import file_sha,fingerprint
from starVLA.rl.qwen_world_postcot.optimized_targets import OptimizedTargets


def prepared(tmp_path):
    tokens=['scene_b','scene_a'];array=np.arange(48,dtype=np.float32).reshape(2,8,3)
    np.save(tmp_path/'targets.npy',array);(tmp_path/'tokens.json').write_text(json.dumps(tokens))
    m=dict(schema='world_optimized_planning_targets_v1',current_observation_manifest_sha256='original_observation_sha',
        DEV_or_Test_labels_used=False,token_sha256=fingerprint(tokens),files={n:file_sha(tmp_path/n) for n in ['targets.npy','tokens.json']})
    m['sha256']=fingerprint(m);(tmp_path/'MANIFEST.json').write_text(json.dumps(m))
    (tmp_path/'COMPLETE.json').write_text(json.dumps(dict(manifest_sha256=file_sha(tmp_path/'MANIFEST.json'))))
    return SimpleNamespace(tokens=tokens,sha='original_observation_sha'),array


def test_token_alignment_and_readonly_source(tmp_path):
    data,array=prepared(tmp_path);reader=OptimizedTargets(tmp_path,data)
    assert np.array_equal(reader.target('scene_a').numpy(),array[1])
    changed=reader.target('scene_a');changed.zero_()
    assert np.array_equal(reader.target('scene_a').numpy(),array[1])
    assert data.sha=='original_observation_sha'


def test_reject_other_observation_population(tmp_path):
    data,_=prepared(tmp_path);data.sha='another_dataset'
    with pytest.raises(ValueError):OptimizedTargets(tmp_path,data)


def test_reject_target_file_mutation(tmp_path):
    data,_=prepared(tmp_path)
    with (tmp_path/'targets.npy').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError):OptimizedTargets(tmp_path,data)
