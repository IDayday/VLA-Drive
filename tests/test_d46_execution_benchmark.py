"""Pure CPU tool checks; synthetic artifacts are never GPU qualification."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest,torch
from tools.d46_world16 import benchmark_execution as b
from starVLA.rl.d44_native_rl.recovery import material_digest


def test_prepare_pins_theta_and_distinct_real_scene_material(monkeypatch,tmp_path):
    cp=tmp_path/'cp';cp.mkdir()
    from starVLA.rl.qwen_world_compare.contracts import fingerprint
    identity=dict(world=16,recipe=dict(arm='D3_FPO'));identity['sha256']=fingerprint(identity)
    (cp/'identity.json').write_text(json.dumps(identity));(cp/'optimizer.pt').write_bytes(b'fixed')
    files={n:b.sha(cp/n)for n in('identity.json','optimizer.pt')};(cp/'files.json').write_text(json.dumps(files))
    (cp/'COMPLETE').write_text(json.dumps(dict(world=16,phase='READY',next_inner=0,updates=1500,files_sha256=b.sha(cp/'files.json'))))
    scenes=[dict(token=f'TRAIN{i}',behavior_version=dict(run=identity['sha256'],update=1500))for i in range(4)]
    batch=dict(scenes=scenes,behavior_update=1500);batch['digest']=material_digest(batch);path=tmp_path/'behavior.pt';torch.save(batch,path)
    monkeypatch.setattr(b,'head',lambda p:'frozen');monkeypatch.setattr(b.subprocess,'check_output',lambda *a,**kw:'')
    args=SimpleNamespace(checkpoint=str(cp),behavior=str(path),baseline_source=str(tmp_path),candidate_source=str(tmp_path),result_dir=str(tmp_path/'results'),output=str(tmp_path/'PLAN.json'),warmup=2,iterations=5)
    b.prepare(args);plan=b.read(args.output)
    assert plan['updates']==1500 and len(set(plan['tokens']))==4 and plan['no_Adam']
    batch['behavior_update']=1502;torch.save(batch,path);args.output=str(tmp_path/'bad.json')
    with pytest.raises(ValueError,match='next-collection'):b.prepare(args)


def test_compare_keeps_exact_and_failed_tolerance_separate(tmp_path):
    for mode,delta in [('baseline',0.),('prefix',0.),('fusion',.01)]:
        grad=tmp_path/(mode+'.pt');torch.save(dict(losses=torch.tensor([1.]),energies=torch.ones(4,4,16),gradients={'lora':torch.ones(16)+delta}),grad)
        row=dict(source={'commit':mode},plan_sha256='same',native_files_sha256='native',behavior_sha256='behavior',gradient_payload=str(grad),gradient_payload_sha256=b.sha(grad),median_seconds=2. if mode=='baseline'else 1.,peak_GPU_allocated_bytes=10,profiler_scalar_sync_events={})
        (tmp_path/(mode+'.json')).write_text(json.dumps(row))
    out=tmp_path/'COMPARE.json';b.compare(SimpleNamespace(result_dir=str(tmp_path),modes=['prefix','fusion'],atol=1e-5,rtol=1e-3,output=str(out)))
    rows=b.read(out)['comparisons'];assert rows[0]['bitexact']and rows[0]['all_gradients_elementwise_close']
    assert not rows[1]['bitexact']and not rows[1]['all_gradients_elementwise_close']
    assert rows[1]['gradient_max_abs']>.009 and b.read(out)['status']=='MEASURED_NOT_FORMAL_ADMISSION'
