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
        captured.append(command);return {'host':host,'job':str(job),'command':command}
    monkeypatch.setattr(owner,'native_launch',launch)
    route=dict(hosts=['training-vlawm-tsinghua-zt5','training-vlawm-tsinghua-zt'],
        source='/frozen/source',config='/frozen/config',master_addr='127.0.0.1',
        qualification_output=str(tmp_path/'qualification'),formal_output=str(tmp_path/'formal'))
    receipts=owner.launch_stage(route,'formal')
    assert len(receipts)==2
    assert all('--resume'not in cmd and '--qualification'not in cmd for cmd in captured)
    assert all(cmd[cmd.index('--output')+1]==route['formal_output']for cmd in captured)
