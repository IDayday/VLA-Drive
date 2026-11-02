import json
from pathlib import Path
from tools.d44_native_rl import qualification_queue as queue


def test_queue_counts_only_real_zero_exit_and_preserves_math_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(queue,'allowed_hosts',lambda **kwargs:[])
    monkeypatch.setattr(queue,'qualification_done',lambda task:True)
    tasks=[]
    for i,code in enumerate((0,1,75)):
        job=tmp_path/str(i);job.mkdir();(job/'RESULT.json').write_text(json.dumps(dict(returncode=code,end_unix=10**12)))
        tasks.append(dict(status='RUNNING_OR_ADMITTING',hosts=[],attempts=[dict(job=str(job),host='host',gpus=[0])]))
    value=queue.tick(dict(worker_source=str(tmp_path),root=str(tmp_path)),dict(tasks=tasks))
    assert [t['status']for t in value['tasks']]==['COMPLETE','FAILED_REQUIRES_REVIEW','RESOURCE_BACKOFF']
    assert not value['complete']


def test_exit0_failed_math_is_not_pass_and_guarded_short_run_resumes(tmp_path):
    import pytest
    proof=tmp_path/'proof.json';proof.write_text(json.dumps(dict(status='FAIL')))
    with pytest.raises(ValueError):queue.qualification_done(dict(command=['python','-m','tools.d44_native_rl.qualify_packing','--output',str(proof)]))
    output=tmp_path/'run';output.mkdir();cp=output/'CP';cp.mkdir()
    (cp/'files.json').write_text('{}')
    (cp/'COMPLETE').write_text(json.dumps(dict(updates=5,phase='INNER_COMMITTED',files_sha256=queue.file_sha(cp/'files.json'))))
    (output/'latest.json').write_text(json.dumps(dict(checkpoint=str(cp))))
    for rank in range(2):(output/f'result_rank{rank}.json').write_text(json.dumps(dict(status='QUALIFICATION_STOPPED',updates=5,frozen_before='hash',frozen_after='hash')))
    task=dict(command=['python','-m','torch.distributed.run','--nproc_per_node=2','-m','tools.d44_native_rl.train','--qualification','--output',str(output),'--stop-after-update','10'])
    assert not queue.qualification_done(task)
    assert task['command'][-2:]==['--resume',str(cp)]
