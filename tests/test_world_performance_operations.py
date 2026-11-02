import copy,hashlib,json
from pathlib import Path
import pytest,yaml
from tools.qwen_world_performance import launch as L


def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_real_seven_recipe_exposure_gates_m2_and_s5_keep_distinct_update_clocks():
    from tools.qwen_world_performance.configure import recipes
    root='/mnt/project/qwen-world-performance-rl-20261103/data/'
    points={}
    for name,cfg in recipes(root+'NAVTRAIN_SCENES.json',root+'AUDIT.json'):
        mapping=L.exposure_gates(cfg);points[name]=[L.point(mapping,e)['updates']for e in(1,3,6,20)]
        assert L.point(mapping,1)['valid_scene_exposures']==103288
        assert L.point(mapping,6)['valid_scene_exposures']==103288*6
    assert points['s1_nominal_rescue']==[3228,9684,19368,64560]
    assert points['s2_conservative_awfm']==[1614,4842,9684,32280]
    assert points['s3_large_fpo']==points['s2_conservative_awfm']
    assert points['s5_performance_max']==[6456,12912,22596,67788]
    assert L.ORDER[:2]==['S1_NOMINAL_RESCUE','S5_PERFORMANCE_MAX']


def test_guarded_m2_pending_checkpoint_can_requeue_but_not_code1_or_foreign_inventory(tmp_path,monkeypatch):
    monkeypatch.setattr(L,'ROOT',tmp_path)
    root=tmp_path/'runs/own';cp=root/'checkpoints/update000005_inner1';cp.mkdir(parents=True)
    cfg=tmp_path/'own.yaml';cfg.write_text('same_science: true\n')
    identity=dict(schema='world_postcot_cleanx_scope_RL_v1',purpose='formal',world=8,recipe=yaml.safe_load(cfg.read_text()))
    identity['sha256']=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    dump(root/'identity.json',identity);dump(cp/'identity.json',identity)
    ranks=[dict(rank=r,updates=5,next_inner=1,phase='INNER_COMMITTED',run=identity['sha256'],valid_scene_exposures=128)for r in range(8)]
    dump(cp/'transaction.json',dict(ranks=ranks))
    for n in ['optimizer.pt']+[f'rank{r}.pt'for r in range(8)]:(cp/n).write_bytes(b'metadata-test-not-a-real-tensor')
    files={p.name:digest(p)for p in cp.iterdir()};dump(cp/'files.json',files);dump(cp/'COMPLETE',dict(schema=2,updates=5,next_inner=1,phase='INNER_COMMITTED',world=8,files_sha256=digest(cp/'files.json')))
    run=dict(id='own',root=str(root),config=str(cfg),inner_updates=2,source='same-frozen-source')
    node=dict(job=str(tmp_path/'jobs/own/attempt001'),host='expired-temp')
    result=dict(returncode=0,host='expired-temp',cwd='same-frozen-source')
    for rank in range(8):dump(root/f'result_rank{rank}.json',dict(updates=5,checkpoint=str(cp),rank=rank,host='expired-temp'))
    kwargs=dict(authorized_hosts=[],grant=dict(temporary_hosts=['expired-temp'],expires_unix=100,checkpoint_guard_seconds=10),now=110)
    returned=L.resource_return(run,node,result,dict(checkpoint=str(cp)),10,**kwargs)
    assert returned['checkpoint']['next_inner']==1 and returned['checkpoint']['updates']==5
    assert L.resource_return(run,node,dict(result,returncode=1),dict(checkpoint=str(cp)),10,**kwargs)is None
    assert L.resource_return(run,node,result,dict(checkpoint=str(cp)),10,authorized_hosts=['expired-temp'],grant=dict(temporary_hosts=[]),now=50)is None
    bad=dict(files);bad['../escape']='x';dump(cp/'files.json',bad)
    with pytest.raises(ValueError,match='inventory'):L.inspect_native(run,cp)
