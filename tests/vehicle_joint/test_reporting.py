import csv
import json
import pytest
from tools.ddpolicy_vehicle.report_campaign import benchmark_table


def test_reporting_averages_scores_and_rejects_dropped_or_failed_scenes(tmp_path):
    index=tmp_path/'index.json';index.write_text(json.dumps([{'token':'a','log':'l1'},{'token':'b','log':'l2'}]))
    fields=['token','log','status','score','no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','ego_progress','comfort']
    for seed,value in [(42,.8),(43,.6)]:
        folder=tmp_path/f'formal_evaluation/run/checkpoint/navtest/scores_seed{seed}';folder.mkdir(parents=True)
        (folder/'summary.json').write_text(json.dumps({'valid':True,'failed':0}))
        rows=[dict(zip(fields,[t,l,'ok',value,1,1,1,value,1])) for t,l in [('a','l1'),('b','l2')]]
        with (folder/'scenes.csv').open('w') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    model={'run_id':'run','arm':'C','seed':42}
    result=benchmark_table(tmp_path,model,'checkpoint','navtest',[42,43],index)
    assert result['mean']['PDMS']==70 and result['scenes']==2
    path=tmp_path/'formal_evaluation/run/checkpoint/navtest/scores_seed43/scenes.csv'
    path.write_text('\n'.join(path.read_text().splitlines()[:2])+'\n')
    with pytest.raises(ValueError,match='population'):benchmark_table(tmp_path,model,'checkpoint','navtest',[42,43],index)
    rows[0]['status']='failed'
    with path.open('w') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    with pytest.raises(ValueError,match='Failed rows'):benchmark_table(tmp_path,model,'checkpoint','navtest',[42,43],index)
