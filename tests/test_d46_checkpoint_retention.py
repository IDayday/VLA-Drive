import json
from pathlib import Path
from tools.d46_fpo_campaign.retention import rotate


def create(root,u):
    p=root/'checkpoints'/f'update{u:06}_inner0';p.mkdir(parents=True)
    (p/'optimizer.pt').write_bytes(b'native')
    for i in range(8):(p/f'rank{i}.pt').write_bytes(b'rng')
    (p/'COMPLETE').write_text(json.dumps(dict(phase='READY',world=8,updates=u,next_inner=0)))
    return p


def test_recent_ten_and_old_best_in_use_pins(tmp_path):
    points=[create(tmp_path,u)for u in range(100,1400,100)]
    (tmp_path/'latest.json').write_text(json.dumps({'checkpoint':str(points[-1])}))
    result=rotate(tmp_path,keep=10,pinned=[points[0],points[1]])
    assert len(result['removed'])==1 and not points[2].exists()
    assert all(p.exists()for p in points if p!=points[2])


def test_incomplete_save_is_never_collected(tmp_path):
    point=create(tmp_path,100);(tmp_path/'latest.json').write_text(json.dumps({'checkpoint':str(point)}))
    partial=tmp_path/'checkpoints/update000200_inner0.incomplete.x';partial.mkdir()
    rotate(tmp_path,keep=1);assert partial.exists()and point.exists()
def test_initial_publication_without_latest_never_prunes(tmp_path):
    directory=tmp_path/'checkpoints';directory.mkdir()
    pending=directory/'update000000_inner0';pending.mkdir()
    (pending/'optimizer.pt').write_bytes(b'in-flight initial save')
    result=rotate(tmp_path,keep=1)
    assert result['status']=='WAITING_INITIAL_LATEST'
    assert result['removed']==[] and (pending/'optimizer.pt').read_bytes()==b'in-flight initial save'

