import json
from pathlib import Path
from tools.d44_native_rl import qualification_queue as queue


def test_queue_counts_only_real_zero_exit_and_preserves_math_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(queue,'allowed_hosts',lambda **kwargs:[])
    tasks=[]
    for i,code in enumerate((0,1,75)):
        job=tmp_path/str(i);job.mkdir();(job/'RESULT.json').write_text(json.dumps(dict(returncode=code,end_unix=10**12)))
        tasks.append(dict(status='RUNNING_OR_ADMITTING',hosts=[],attempts=[dict(job=str(job),host='host',gpus=[0])]))
    value=queue.tick(dict(worker_source=str(tmp_path),root=str(tmp_path)),dict(tasks=tasks))
    assert [t['status']for t in value['tasks']]==['COMPLETE','FAILED_REQUIRES_REVIEW','RESOURCE_BACKOFF']
    assert not value['complete']
