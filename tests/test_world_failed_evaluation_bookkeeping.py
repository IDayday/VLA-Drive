"""Completed siblings of a failed shard must release scheduler capacity."""
import json

from tools.qwen_world_safety_exploration.eval_owner import EvaluationOwner


def test_failed_evaluation_harvests_siblings_without_retrying_failed_shard(tmp_path):
    (tmp_path/'rank1.json').write_text(json.dumps(dict(status='COMPLETE', world=2)))
    owner = EvaluationOwner.__new__(EvaluationOwner)
    entry = dict(status='FAILED_REVIEW_REQUIRED', world=2, evaluation=str(tmp_path),
        ranks={'0': dict(status='FAILED_REVIEW_REQUIRED', job='interrupted'),
               '1': dict(status='RUNNING', job='completed')})
    owner.registry = dict(runs=[])
    owner.state = dict(runs={}, evaluations={'failed_model': entry})
    owner.helper_active = lambda: False
    inspected = []
    def finished(node):
        inspected.append(node['job'])
        return True
    owner.finished = finished
    owner.advance([])
    assert inspected == ['completed']
    assert entry['status'] == 'FAILED_REVIEW_REQUIRED'
    assert entry['ranks']['0'] == dict(status='FAILED_REVIEW_REQUIRED', job='interrupted')
    assert entry['ranks']['1']['status'] == 'COMPLETE'
    assert 'summarizer' not in entry
    assert not any(n.get('job') and n.get('status') not in ('COMPLETE', 'FAILED_REVIEW_REQUIRED')
        for n in entry['ranks'].values())
