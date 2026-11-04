import json
from tools.d46_aug_migration.tracking import snapshot


def write(tmp_path,rank,*,bad=False):
    rows=[]
    for u in range(1,7):
        rows.append(dict(updates=u,run='own',invocation='attempt',at=100+u*10,inner_completed=1 if u%2 else 2,
            gradient_norm=2.,global_advantage_std=.2 if not bad else .3,
            losses=dict(fresh_ratio_error=0.,reference_KL=.02,ratio_mean=1. if u%2 else .99,clip_fraction=0.),
            role_gradient_norms=dict(action_io=.8,text_lora=.6),scene_exposures=((u+1)//2)*64,scene_epochs=.002,
            exploration=dict(effective_PG_group_fraction=.5,feasible_candidate_fraction=.9,all_zero_groups=2,all_one_groups=4)))
    (tmp_path/f'metrics_rank{rank}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n{"partial":')


def test_allrank_health_uses_end_to_end_clock_and_ignores_partial_line(tmp_path):
    write(tmp_path,0);write(tmp_path,1)
    s=snapshot(tmp_path,2,now=161)
    assert s['alerts']==[] and s['common_update']==6
    assert s['end_to_end']['seconds_per_update']==10
    assert abs(s['role_gradient_squared_norm_fraction']['action_io']-.64)<1e-10
    assert s['exploration']['all_zero_groups']==2


def test_missing_rank_is_not_healthy_and_different_global_std_is_reported(tmp_path):
    write(tmp_path,0)
    assert snapshot(tmp_path,2)['status']=='WAITING_ALL_RANKS'
    write(tmp_path,1,bad=True)
    assert '全局优势std不一致' in snapshot(tmp_path,2,now=161)['alerts']
