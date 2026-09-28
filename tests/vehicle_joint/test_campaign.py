import copy
import json
import csv
import pytest
from tools.ddpolicy_vehicle.campaign import validate_allocations, charged_gpu_hours
from tools.ddpolicy_vehicle.analyse_campaign import common_vehicle_comparison
from tools.ddpolicy_vehicle.scratch_cache import manage_cache


def test_allocations_reject_overlap_unapproved_host_and_foreign_run():
    plan={'models':[{'run_id':'formal_A_seed42_001','seed':42,'training_slot':{'gpus':[0,1]}},
                    {'run_id':'formal_B_seed42_001','seed':42,'training_slot':{'gpus':[2,3]}}],
          'evaluation_slots':[{'gpus':[0]},{'host':'training-vla-zt2','gpus':[1]}]}
    validate_allocations(plan)
    bad=copy.deepcopy(plan);bad['models'][1]['training_slot']['gpus']=[1,2]
    with pytest.raises(ValueError,match='Overlapping'):validate_allocations(bad)
    bad=copy.deepcopy(plan);bad['evaluation_slots'][0]['host']='unallocated-host'
    with pytest.raises(ValueError,match='authorized'):validate_allocations(bad)
    bad=copy.deepcopy(plan);bad['models'][0]['run_id']='../historic'
    with pytest.raises(ValueError,match='new named'):validate_allocations(bad)


def test_budget_counts_each_attempt_once_including_failed_load(tmp_path):
    run=tmp_path/'training'/'model';run.mkdir(parents=True)
    first={'status':'PAUSED','start_unix':1000,'end_unix':4600,'gpu_count':2}
    second={'status':'COMPLETE','start_unix':5000,'end_unix':8600,'gpu_count':2}
    (run/'attempt_1.json').write_text(json.dumps(first));(run/'attempt_2.json').write_text(json.dumps(second))
    # The latest status repeats attempt2 and must not be charged twice.
    (run/'status.json').write_text(json.dumps(second))
    failed=tmp_path/'runs'/'failed';failed.mkdir(parents=True)
    (failed/'status.json').write_text(json.dumps({'status':'FAILED','start_unix':0,'end_unix':1800,'gpu_count':1}))
    assert charged_gpu_hours(tmp_path)==4.5


def test_common_vehicle_errors_do_not_hide_full_population_misses(tmp_path):
    paths=[]
    for side in ('left','right'):
        path=tmp_path/(side+'.csv')
        rows=[{'token':'s','track_id':str(i),'log':'log1','motion_group':'moving',
               'export_failure':'','detected':'True' if i==0 else 'False','selected':'True' if i==0 else 'False',
               'joint_ADE':(1 if side=='left' else 2) if i==0 else ''} for i in range(2)]
        with path.open('w') as f:
            w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
        paths.append(path)
    common_vehicle_comparison([paths[0]],[paths[1]],tmp_path/'comparison')
    value=json.loads((tmp_path/'comparison/summary.json').read_text())
    assert value['full_supervised_vehicle_population']==2
    assert value['metrics']['joint_ADE']['all']['common_targets_all_inference_seeds']==1
    assert value['metrics']['joint_ADE']['all']['right_minus_left']['mean']==1


def test_only_owned_completed_prediction_copies_can_be_released(tmp_path):
    cache=tmp_path/'scratch';bank=tmp_path/'predictions';bank.mkdir()
    manage_cache(cache,bank)
    (cache/'copy.pt').write_text('exact temporary copy')
    (bank/'shard_0.json').write_text(json.dumps({'status':'paused','failed':0,'completed':1,'requested':2}))
    with pytest.raises(ValueError,match='Retain'):manage_cache(cache,bank,True)
    assert (cache/'copy.pt').exists()
    with pytest.raises(ValueError,match='ownership'):manage_cache(cache,tmp_path/'foreign',True)
    (bank/'shard_0.json').write_text(json.dumps({'status':'complete','failed':0,'completed':2,'requested':2}))
    manage_cache(cache,bank,True)
    assert not cache.exists()
