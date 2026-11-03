"""Metadata-only owner adoption/failure/command contracts; never dispatch GPUs."""
import copy
import json
from pathlib import Path
import pytest
import yaml
from starVLA.rl.qwen_world_compare.contracts import file_sha
from tools.d46_augustus_ops import owner as o


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return value


def receipt(route,phase,node,root):
    job=root/'jobs/AUGUSTUS_20261103'/route['experiment']/phase/f'node{node}'
    return dict(host=route['hosts'][node],job=str(job),command=o.command(route,node,phase),source=route['source'],source_sha=o.SCI_SHA,
        worker_source=route['worker_source'],worker_source_sha=o.WORKER_SHA,gpus=list(range(8)),expected_gpu_uuids=[f'GPU{node}_{k}'for k in range(8)])


@pytest.fixture
def setup(tmp_path,monkeypatch):
    directory=tmp_path/'operations/AUGUSTUS_CAMPAIGN_20261103';directory.mkdir(parents=True)
    routes=[];launches=[]
    for exp,nodes in [('AUG1_FULL',4),('AUG2_LORA',2)]:
        cfg=tmp_path/'registered_configs'/(exp+'.yaml');cfg.parent.mkdir(exist_ok=True)
        cfg.write_text(yaml.safe_dump(dict(experiment_id=exp,inner_updates=1,target_epochs=6.,group=8)))
        r=dict(experiment=exp,world=8*nodes,hosts=[f'{exp}_node{n}'for n in range(nodes)],master_addr='172.16.1.1',port=29861,
            source=str(tmp_path/'SCI'),source_sha=o.SCI_SHA[:7],worker_source=str(tmp_path/'OPS'),config=str(cfg),config_sha256=file_sha(cfg),
            output=str(tmp_path/'qualification'/(exp+'_q')),qualification=True,target_update=1,fresh_IL46=True,init_Adam_update=0)
        r['formal_output']=str(tmp_path/'runs'/(exp+'_v1'));routes.append(r)
        launches.extend(receipt(r,'qualification1',n,tmp_path)for n in range(nodes))
    plan=directory/'QUALIFICATION_PLAN.json';lp=directory/'QUALIFICATION_LAUNCHES.json'
    put(plan,dict(maximum_simultaneous_routes=2,routes=routes));put(lp,dict(receipts=launches))
    for item in launches:put(Path(item['job'])/'LAUNCH.json',item)
    monkeypatch.setattr(o,'head',lambda path:o.SCI_SHA if str(path).endswith('SCI')else o.WORKER_SHA)
    state=o.load(plan,lp,tmp_path);own=o.Owner(state,directory)
    return tmp_path,directory,routes,state,own,plan,lp


def complete_jobs(row,phase,code=0):
    for node in row['stages'][phase]['nodes']:
        put(Path(node['job'])/'RESULT.json',dict(returncode=code,command=node['command'],cwd=row['route']['source'],host=node['host']))


def test_max2_and_formal_commands_never_resume_qualification_weights(setup):
    root,directory,routes,state,owner,plan,lp=setup
    r=routes[1];q=o.command(r,0,'qualification_resume');formal=o.command(r,0,'formal')
    assert q[q.index('--resume')+1]==str(Path(r['output'])/'checkpoints/update000001_inner0')
    assert '--qualification'in q and q[q.index('--stop-after-update')+1]=='2'
    assert '--resume'not in formal and'--qualification'not in formal and'--stop-after-update'not in formal
    assert formal[formal.index('--output')+1]==r['formal_output']
    value=json.loads(plan.read_text());value['routes'].append(copy.deepcopy(value['routes'][0]));put(plan,value)
    with pytest.raises(ValueError,match='MAX2'):o.load(plan,lp,root)


def test_prechild_failure_retained_and_only_own_siblings_get_cooperative_stop(setup):
    root,directory,routes,state,owner,plan,lp=setup;row=state['routes']['AUG1_FULL'];node=row['stages']['qualification1']['nodes'][0]
    put(Path(node['job'])/'RESULT.json',dict(returncode=75,child_not_started=True,command=node['command'],cwd=row['route']['source'],host=node['host']))
    with pytest.raises(RuntimeError,match='failure/resource refusal'):owner.tick(row)
    owner.fail(row,'actual75 retained, no auto retry')
    assert row['status']=='REVIEW_REQUIRED'and len(row['stages'])==1
    for sibling in row['stages']['qualification1']['nodes'][1:]:assert(Path(sibling['job'])/'STOP_REQUESTED').exists()
    for other in state['routes']['AUG2_LORA']['stages']['qualification1']['nodes']:assert not(Path(other['job'])/'STOP_REQUESTED').exists()


def test_adopts_manual_resume_and_fresh_formal_without_double_launch(setup,monkeypatch):
    root,directory,routes,state,owner,plan,lp=setup;row=state['routes']['AUG2_LORA'];r=row['route'];complete_jobs(row,'qualification1')
    resume=[receipt(r,'qualification_resume',n,root)for n in range(2)];formal=[receipt(r,'formal',n,root)for n in range(2)]
    put(directory/'QUALIFICATION_RESUME_LAUNCHES.json',dict(receipts=resume));put(directory/'FORMAL_LAUNCHES.json',formal)
    calls=[]
    monkeypatch.setattr(o,'review_native',lambda route,out,u,q:dict(status='PASS',updates=u,qualification=q,optimizer_empty=u==0,scheduler_updates=u))
    owner.tick(row)
    assert row['phase']=='qualification_resume'and row['proofs']['q1']['q1_result_files_overwritten_by_explicit_own_resume']
    complete_jobs(row,'qualification_resume')
    # Q1 result files have been overwritten by Q2 exactly as the actual trainer.
    for rank in range(r['world']):put(Path(r['output'])/f'result_rank{rank}.json',dict(rank=rank,updates=2,status='QUALIFICATION_STOPPED',qualification=True,
        frozen_before='actual_frozen_hash',frozen_after='actual_frozen_hash',phase='READY',next_inner=0))
    owner.tick(row)
    assert row['phase']=='formal'and row['status']=='FORMAL_RUNNING'
    assert all(n['receipt']for n in row['stages']['formal']['nodes'])and calls==[]


def test_metadata_or_audit_failure_marks_review_without_stopping_healthy_children(setup):
    root,directory,routes,state,owner,plan,lp=setup
    row=state['routes']['AUG1_FULL'];owner.fail(row,'CPU metadata/audit mismatch, not an observed child failure')
    assert row['status']=='REVIEW_REQUIRED'and 'actual_child_failure'not in row
    for route in state['routes'].values():
        for node in route['stages']['qualification1']['nodes']:assert not(Path(node['job'])/'STOP_REQUESTED').exists()
