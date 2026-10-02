import json
import pytest
from tools.action_video_foresight.summarize_live_campaign import union_seconds,resource_cost,read_steps


def test_physical_gpu_union_does_not_double_count_shared_jobs(tmp_path):
    for name,start,end in [('trainer',0,3600),('probe',1800,5400)]:
        root=tmp_path/'runs'/name;root.mkdir(parents=True)
        (root/'status.json').write_text(json.dumps({'gpu_count':1,'host':'same_host',
            'cuda_visible_devices':'0','start_unix':start,'end_unix':end,'kind':name}))
    cost=resource_cost(tmp_path,5400)
    assert cost['job_allocated_GPUh']==2.
    assert cost['physical_device_interval_union_GPUh']==1.5
    assert cost['unassigned_allocation_GPUh']==0.
    with pytest.raises(ValueError,match='Negative'):union_seconds([(2,1)])


def test_inflight_jsonl_tail_is_not_a_failed_completed_record(tmp_path):
    path=tmp_path/'steps.jsonl';path.write_text('{"update":1}\n{"update":')
    assert read_steps(path)==[{'update':1}]
    path.write_text('{"update":1}\nBAD\n')
    with pytest.raises(json.JSONDecodeError):read_steps(path)
