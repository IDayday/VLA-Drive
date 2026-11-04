import json
from pathlib import Path
import pytest
from tools.d46_aug_migration import owner


def test_failed_owned_gang_requests_only_its_peer_stop(tmp_path):
    left=tmp_path/'a';right=tmp_path/'b';left.mkdir();right.mkdir()
    (left/'RESULT.json').write_text(json.dumps({'returncode':1}))
    with pytest.raises(RuntimeError,match='child failed'):
        owner.finished([{'job':str(left)},{'job':str(right)}])
    assert (right/'STOP_REQUESTED').exists()
    assert not(left/'STOP_REQUESTED').exists()


def test_formal_launch_never_adopts_qualification_weights(tmp_path,monkeypatch):
    monkeypatch.setattr(owner,'ROOT',tmp_path)
    captured=[]
    def launch(host,source,job,command,gpus,**kw):
        captured.append(command);return {'host':host,'job':str(job),'command':command,'source':source}
    monkeypatch.setattr(owner,'native_launch',launch)
    route=dict(hosts=['training-vlawm-tsinghua-zt5','training-vlawm-tsinghua-zt'],
        source='/frozen/source',formal_source='/frozen/formal_metadata_fix',config='/frozen/config',master_addr='127.0.0.1',
        qualification_output=str(tmp_path/'qualification'),formal_output=str(tmp_path/'formal'))
    receipts=owner.launch_stage(route,'formal')
    assert len(receipts)==2
    assert all(r['source']==route['formal_source']for r in receipts)
    assert all('--resume'not in cmd and '--qualification'not in cmd for cmd in captured)
    assert all(cmd[cmd.index('--output')+1]==route['formal_output']for cmd in captured)


def test_learner_plus_pending_eval_reserves_both_slots(monkeypatch):
    from tools.d46_augustus_ops import evaluations as e
    host='training-vlawm-zt';monkeypatch.setattr(e,'SPARE_GPUS',{host:(0,)})
    row=dict(index=0,free_MiB=50000,used_MiB=30000,owners=[dict(known=True,kind='training',used_MiB=30000)])
    assert e.available_slots(host,[row],{})==[(host,0)]
    assert e.available_slots(host,[row],{(host,0):1})==[]


def test_secondary_four_rank_fresh_formal_preserves_scene_configuration(tmp_path,monkeypatch):
    monkeypatch.setattr(owner,'ROOT',tmp_path);calls=[]
    def launch(host,source,job,command,gpus,**kw):
        calls.append((command,gpus));return dict(job=str(job),command=command)
    monkeypatch.setattr(owner,'native_launch',launch)
    route=dict(hosts=['training-vlawm-tsinghua-zt5'],gpus=[0,1,2,3],world=4,source='/science',
        config='/B64_G16_M2.yaml',master_addr='127.0.0.1',master_port=29648,
        qualification_output=str(tmp_path/'qualification'),formal_output=str(tmp_path/'formal'),
        proof_directory=str(tmp_path/'proof'),job_namespace='secondary')
    owner.launch_stage(route,'formal');cmd,gpus=calls[0]
    assert gpus==[0,1,2,3]and '--nnodes=1'in cmd and '--nproc_per_node=4'in cmd
    assert '--resume'not in cmd and '--qualification'not in cmd and '--master_port=29649'in cmd
    assert cmd[cmd.index('--config')+1]=='/B64_G16_M2.yaml'
