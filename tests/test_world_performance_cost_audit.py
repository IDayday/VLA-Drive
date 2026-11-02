import json
from tools.qwen_world_performance.cost_audit import collect, merged_seconds


def test_overlap_union_and_incomplete_receipts_do_not_create_compute(tmp_path):
    assert merged_seconds([(0, 10), (3, 8), (10, 15), (20, 21)]) == 16
    (tmp_path/'RUNS.json').write_text(json.dumps({'runs':[{'id':'own','root':str(tmp_path/'run')}]}))
    base=tmp_path/'jobs'/'own'
    for name, start, end, child in [('a',0,3600,1),('b',1800,5400,2),('reject',0,99999,None)]:
        path=base/name;path.mkdir(parents=True)
        (path/'RESULT.json').write_text(json.dumps(dict(start_unix=start,end_unix=end,child_pid=child,
            host='host',gpu_uuids=['GPU-one'],returncode=0)))
    value=collect(tmp_path)
    assert value['completed_admitted_invocation_GPU_slot_hours']==2
    assert value['completed_physical_device_interval_union_hours']==1.5
    assert len(value['unadmitted_receipts'])==1
    assert value['CPU_core_hours'] is None
    assert value['published_training_snapshots'][0]['latest_published_fresh_PDM_queries'] is None
    cpu=tmp_path/'jobs'/'evaluation'/'own_u1'/'export_attempt001';cpu.mkdir(parents=True)
    (cpu/'RESULT.json').write_text(json.dumps(dict(start_unix=0,end_unix=60,child_pid=3,
        host='host',gpu_uuids=[],gpus=[],returncode=0)))
    shard=tmp_path/'jobs'/'evaluation'/'own_u1'/'NAVTEST'/'rank0_attempt001';shard.mkdir(parents=True)
    (shard/'RESULT.json').write_text(json.dumps(dict(start_unix=0,end_unix=60,child_pid=4,
        host='other_host',gpu_uuids=['GPU-two'],returncode=0)))
    value=collect(tmp_path)
    assert len(value['CPU_only_receipts'])==1
    assert not value['rejected_receipts']
    assert abs(value['completed_admitted_invocation_GPU_slot_hours']-2-1/60)<1e-10
