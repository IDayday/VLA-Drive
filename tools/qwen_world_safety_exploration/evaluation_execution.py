"""Explicit auxiliary native-snapshot repair, independent of training source."""
from functools import lru_cache
import json
from pathlib import Path

from starVLA.rl.qwen_world_safety_exploration.contracts import file_sha, fingerprint


@lru_cache(maxsize=8)
def _verified(original, execution, receipt_path, receipt_sha):
    if file_sha(receipt_path) != receipt_sha:raise ValueError('Evaluation repair receipt changed')
    record = json.loads(Path(receipt_path).read_text())
    body = dict(record)
    saved = body.pop('sha256', None)
    if (saved != fingerprint(body) or record.get('status') != 'PASS'
            or record.get('schema') != 'world_FP32_native_snapshot_only_eval_execution_v1'
            or Path(record['scientific_source']).resolve() != Path(original).resolve()
            or Path(record['evaluation_source']).resolve() != Path(execution).resolve()
            or record.get('training_source_unchanged') is not True):
        raise ValueError('Unreviewed evaluation execution source')
    required = {'tools/qwen_world_safety_exploration/evaluate.py',
                'starVLA/model/modules/qwen_world_postcot/actor.py',
                'starVLA/model/modules/qwen_world_postcot/model.py',
                'starVLA/model/modules/qwen_world_postcot/flow.py',
                'starVLA/rl/qwen_world_postcot/deployment.py'}
    if not required.issubset(record['unchanged_prediction_files']):
        raise ValueError('Prediction equivalence inventory is incomplete')
    for relative, sha in record['unchanged_prediction_files'].items():
        for root in (Path(original), Path(execution)):
            p = (root/relative).resolve()
            if not p.is_relative_to(root.resolve()) or file_sha(p) != sha:
                raise ValueError('Evaluation repair changed predictor/scoring code: '+relative)
    for key, root in [('old_snapshot_sha256', original), ('new_snapshot_sha256', execution)]:
        if file_sha(Path(root)/'starVLA/rl/qwen_world_postcot/snapshot_navtest.py') != record[key]:
            raise ValueError('Native snapshot execution repair changed')
    evidence = record['CPU_evidence']
    if (file_sha(evidence['path']) != evidence['sha256']
            or json.loads(Path(evidence['path']).read_text()).get('status') != 'PASS'):
        raise ValueError('Native snapshot repair lacks passed equivalence evidence')
    return execution


def evaluation_source(run):
    source = run.get('evaluation_source')
    if source is None:return run['source']
    if run['id'] != 'IMPROVEMENT_FP32_OUTPUT_G8':
        raise ValueError('Evaluation source override is restricted to the declared FP32 diagnostic repair')
    receipt = run['evaluation_source_receipt']
    return _verified(run['source'], source, receipt, file_sha(receipt))
