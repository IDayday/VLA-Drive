import numpy as np
import json
import sys
from tools.local_interaction_mask_v2.analyze_joint_plans import path_encounters,main


def test_passage_proxy_distinguishes_arrival_times_and_ignores_stationary_order():
    ego=np.stack([np.arange(8)-2,np.zeros(8)],axis=-1)
    crossing=np.stack([np.zeros(8),np.arange(8)-4],axis=-1)
    events=path_encounters(ego,np.stack([crossing,np.zeros_like(crossing)]))
    assert set(events)=={0}
    assert events[0]['distance_m']==0 and events[0]['arrival_gap_s']==1.
    assert path_encounters(np.zeros_like(ego),crossing[None])=={}


def test_exported_actual_and_internal_ego_order_remain_distinct(tmp_path,monkeypatch):
    actual=np.stack([np.arange(8)-2,np.zeros(8),np.zeros(8)],axis=-1)
    internal=np.stack([np.arange(8)-5,np.zeros(8)],axis=-1)
    crossing=np.stack([np.zeros(8),np.arange(8)-4],axis=-1)
    np.savez(tmp_path/'scene.npz',trajectory=actual,graph_joint_xy=np.stack([internal,crossing,np.full((8,2),np.nan)]),
             graph_active_actor_mask=np.array([True,True,False]),graph_source_slot_ids=np.array([-1,5,-2]))
    (tmp_path/'tokens.json').write_text('["scene"]')
    monkeypatch.setattr(sys,'argv',['analyze','--tokens',str(tmp_path/'tokens.json'),'--predictions',str(tmp_path),'--output',str(tmp_path/'report')])
    main();report=json.loads((tmp_path/'report/summary.json').read_text())
    assert report['failed']==0 and report['passage_order_disagreement_pairs_total']==1
    assert report['actual_DiT_ego_first_pairs_total']==1
    assert report['internal_graph_neighbor_first_pairs_total']==1
