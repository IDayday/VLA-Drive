"""Bounded CPU audit-helper tests. Fixtures are metadata, never GPU evidence."""
import copy,json,random
from pathlib import Path
import numpy as np
import pytest,torch
from tools.d46_world16.audit_fpo_gpu import exact,job_review,fresh_review


def test_full_RNG_tree_checks_every_domain_without_mutation():
    rng=dict(python=random.Random(42).getstate(),numpy=np.random.RandomState(42).get_state(),torch=torch.Generator().manual_seed(42).get_state(),cuda=torch.arange(20,dtype=torch.uint8),rollout_generator=torch.Generator().manual_seed(420000).get_state())
    exact(rng,copy.deepcopy(rng))
    for key in rng:
        bad=copy.deepcopy(rng)
        if key=='python':bad[key]=random.Random(43).getstate()
        elif key=='numpy':bad[key][1][0]+=1
        else:bad[key][0]^=1
        with pytest.raises(ValueError):exact(rng,bad)
    fresh=dict(fpo_joint_logratio_min=0.,fpo_joint_logratio_max=0.,fpo_ratio_mean=1.,fpo_ratio_max=1.,fresh_ratio_error=0.)
    assert fresh_review(fresh)==fresh
    with pytest.raises(ValueError,match='energy logratio'):fresh_review(dict(fresh,fpo_joint_logratio_max=.002))
    with pytest.raises(ValueError,match='Missing'):fresh_review({})


def test_job_receipts_bind_exact_two_nodes_source_resume_and_actual_exit(tmp_path):
    q=tmp_path/'q';science=tmp_path/'s';resume=q/'checkpoints/update000003_inner1'
    for n in range(2):
        job=tmp_path/f'node{n}';job.mkdir()
        cmd=['python','-m','torch.distributed.run','--nnodes=2','--nproc_per_node=8',f'--node_rank={n}','-m','tools.d46_world16.train','--qualification','--output',str(q),'--resume',str(resume),'--stop-after-update','4']
        (job/'LAUNCH.json').write_text(json.dumps(dict(command=cmd,source=str(science),source_sha='frozen')))
        (job/'RESULT.json').write_text(json.dumps(dict(command=cmd,cwd=str(science),host=f'node{n}',returncode=0,child_pid=100+n,gpus=list(range(8)),gpu_uuids=[f'UUID-{n}-{i}'for i in range(8)],start_unix=1.,end_unix=2.)))
    rows=job_review(tmp_path,science,'frozen',q,resume,4)
    assert len(rows)==2 and all(r['returncode']==0 for r in rows)
    path=tmp_path/'node1/RESULT.json';bad=json.loads(path.read_text());bad['returncode']=75;path.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='actually exit0'):job_review(tmp_path,science,'frozen',q,resume,4)
    bad['returncode']=0;bad['command']=bad['command']+['--resume','other'];path.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='command/cwd'):job_review(tmp_path,science,'frozen',q,resume,4)
