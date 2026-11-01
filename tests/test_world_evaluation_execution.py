"""The auxiliary restore fix cannot silently replace training or inference."""
import json
from pathlib import Path
import pytest
from starVLA.rl.qwen_world_safety_exploration.contracts import file_sha, fingerprint
from tools.qwen_world_safety_exploration.evaluation_execution import evaluation_source


def test_explicit_precision_only_eval_source_and_changed_predictor_rejection(tmp_path):
    old, new = tmp_path/'old', tmp_path/'new'
    files = ['tools/qwen_world_safety_exploration/evaluate.py',
             'starVLA/model/modules/qwen_world_postcot/actor.py',
             'starVLA/model/modules/qwen_world_postcot/model.py',
             'starVLA/model/modules/qwen_world_postcot/flow.py',
             'starVLA/rl/qwen_world_postcot/deployment.py']
    for root in (old, new):
        for name in files:
            p = root/name;p.parent.mkdir(parents=True, exist_ok=True);p.write_text('unchanged')
        (root/'starVLA/rl/qwen_world_postcot/snapshot_navtest.py').write_text(root.name)
    evidence = tmp_path/'passed.json';evidence.write_text(json.dumps(dict(status='PASS')))
    record = dict(schema='world_FP32_native_snapshot_only_eval_execution_v1', status='PASS',
                  scientific_source=str(old), evaluation_source=str(new), training_source_unchanged=True,
                  unchanged_prediction_files={p:file_sha(old/p) for p in files},
                  old_snapshot_sha256=file_sha(old/'starVLA/rl/qwen_world_postcot/snapshot_navtest.py'),
                  new_snapshot_sha256=file_sha(new/'starVLA/rl/qwen_world_postcot/snapshot_navtest.py'),
                  CPU_evidence=dict(path=str(evidence), sha256=file_sha(evidence)))
    receipt = tmp_path/'receipt.json'
    def write():receipt.write_text(json.dumps(dict(record, sha256=fingerprint(record))))
    write()
    run = dict(id='IMPROVEMENT_FP32_OUTPUT_G8', source=str(old), evaluation_source=str(new),
               evaluation_source_receipt=str(receipt))
    assert evaluation_source(run) == str(new)
    assert evaluation_source(dict(id='old', source=str(old))) == str(old)
    with pytest.raises(ValueError):evaluation_source(dict(run, id='IMPROVEMENT_MIXED_NOISE_G8'))
    record['unchanged_prediction_files'][files[0]] = 'wrong';write()
    with pytest.raises(ValueError):evaluation_source(run)
