import json
import shutil

from iqe.io import atomic_json
from iqe.specialist_study.costs import training_cost


def test_resume_cost_counts_saved_segments_once(tmp_path):
    folder=tmp_path/'experts/E00';folder.mkdir(parents=True)
    archive=tmp_path/'runtime/training_segments/E00';archive.mkdir(parents=True)
    atomic_json(archive/'result_2.json', {'optimizer_steps':2,'new_optimizer_steps':2,
        'status':'STOPPED_AT_BOUNDARY','elapsed_seconds':10.,'GPU_seconds':10.})
    shutil.copy2(archive/'result_2.json',archive/'result_2_duplicate.json')
    atomic_json(folder/'result.json', {'optimizer_steps':4,'new_optimizer_steps':2,
        'status':'COMPLETE','elapsed_seconds':6.,'GPU_seconds':6.})
    rows=[{'optimizer_step':s,'global_batch_actual':128,'compute_update_seconds':1.}
          for s in (1,2,2,3,4)]
    (folder/'steps.jsonl').write_text('\n'.join(map(json.dumps,rows)))
    cost=training_cost(tmp_path,'experts','E00')
    assert cost['sample_presentations']==512
    assert cost['optimizer_update_loop_seconds']==4.
    assert cost['completed_invocation_allocated_GPU_seconds']==16.
    assert cost['invocation_coverage_complete']
    assert len(cost['invocation_receipts'])==2


def test_missing_source_cost_remains_explicit(tmp_path):
    folder=tmp_path/'scorers/S00';folder.mkdir(parents=True)
    atomic_json(folder/'result.json', {'optimizer_steps':4,'new_optimizer_steps':2,
        'status':'COMPLETE','elapsed_seconds':6.,'GPU_seconds':6.})
    (folder/'steps.jsonl').write_text(json.dumps({'optimizer_step':4,'global_batch_actual':128,'compute_update_seconds':1.}))
    cost=training_cost(tmp_path,'scorers','S00')
    assert cost['completed_invocation_allocated_GPU_seconds']==6.
    assert not cost['invocation_coverage_complete']
