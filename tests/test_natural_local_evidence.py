import numpy as np
import shapely
from shapely.geometry import box
import pytest
from starVLA.dataloader.minimal_repair_scene import AnnotatedScene
from starVLA.model.modules.decision_local.local_evidence import extract_local_evidence,objects_at_time,UnknownReason
from starVLA.model.modules.decision_local.local_constraint_checker import ConstraintConfig,LocalConstraintChecker
from starVLA.model.modules.decision_local.minimal_repair_targets import solve_minimal_repair
from tools.decision_local.audit_base_training_sources import split_roles,recording


def obj(x=0., y=0., reliable=True, velocity=(0.,0.)):
    return dict(corners=np.array([[x-1,y-1],[x+1,y-1],[x+1,y+1],[x-1,y+1]]),
                velocity=np.array(velocity),reliable=reliable,stationary=True)


def scene(frames=None, observed=True):
    return AnnotatedScene(road=box(-20,-2,80,2),
        body=dict(length=4.,width=2.,rear_axle_to_center=1.),
        frames=frames or [{} for _ in range(9)],observed=np.full((9,160,200),observed),
        road_height_known=True)


def test_early_road_positive_survives_late_unknown_and_strict_target_does_not():
    s=scene();s.road_height_known=False;s.local_road_height_known=np.array([1]*8+[0],bool)
    poses=np.array([[4.,2.,0.],[16.,0.,0.]])
    facts=extract_local_evidence(s,poses,[1.,4.])
    assert facts.positive_known[0,0] and not facts.label_valid[1,0]
    plan=np.stack((np.arange(1,9)*2,np.ones(8)*2,np.zeros(8)),-1)
    teacher=solve_minimal_repair(plan,[4.,0.],LocalConstraintChecker(s,ConstraintConfig(velocity_boundary='measured_vector_v2')))
    assert teacher['status_name']=='UNCERTAIN' and not teacher['target_valid'] and teacher['target'] is None


def test_a_only_codec_preserves_physical_targets_and_legal_current_state(monkeypatch):
    import torch
    from starVLA.dataloader import foresight_dataset as fd
    from starVLA.dataloader.a_only_codec import install_legacy_codec,recode_historical_targets
    raw=np.array([[0.,0.,0.],[3.,-.7,.2],[20.,2.,-.5]],np.float32)
    old=torch.from_numpy(fd.encode_ego(raw))
    codec=dict(schema='a_only_action_affine_v1',fit_role='A_train',mean_xy_m=[10.4,.3],std_xy_m=[8.9,2.3])
    for key in ('X_MEAN','Y_MEAN','X_STD','Y_STD'):monkeypatch.setattr(fd,key,getattr(fd,key))
    install_legacy_codec(codec)
    new=recode_historical_targets(old,codec)
    torch.testing.assert_close(fd.decode_ego(new),torch.from_numpy(raw),rtol=1e-6,atol=2e-6)
    assert torch.equal(new[:,2:],old[:,2:])
    current=np.array([[1.5,-.1,.02]],np.float32)
    torch.testing.assert_close(fd.decode_ego(torch.from_numpy(fd.encode_ego(current))),torch.from_numpy(current),rtol=1e-6,atol=2e-6)


def test_a_only_codec_rejects_statistics_from_correction_pool():
    from starVLA.dataloader.a_only_codec import validate_codec
    with pytest.raises(ValueError):
        validate_codec(dict(schema='a_only_action_affine_v1',fit_role='B_predictor',mean_xy_m=[1.,0.],std_xy_m=[1.,1.]))


def test_known_collision_survives_unknown_other_actor():
    frames=[dict(known=obj(4),unknown=obj(4,reliable=False)) for _ in range(9)]
    facts=extract_local_evidence(scene(frames,False),np.array([[4.,0.,0.]]),[1.])
    assert facts.positive_known[0,2] and facts.label_valid[0,1]
    assert facts.unknown_reason[0,2]&int(UnknownReason.CURRENT_OBJECT_SUPPORT)
    assert any(x['actor']=='known' for x in facts.events)


def test_no_box_is_not_negative_or_distance_truth():
    facts=extract_local_evidence(scene(observed=False),np.zeros((1,3)),[1.])
    assert not facts.label_valid[0,2] and not facts.negative_known[0,2]
    assert not facts.label_valid[0,1]


def test_complete_observed_body_allows_negative_not_just_corners():
    s=scene();s.observed[:,80,42]=False
    facts=extract_local_evidence(s,np.zeros((1,3)),[1.])
    assert not facts.negative_known[0,2]
    s.observed[:]=True
    facts=extract_local_evidence(s,np.zeros((1,3)),[1.])
    assert facts.negative_known[0,2]


def test_disappearing_actor_is_unknown_not_release():
    s=scene([dict(a=obj())]+[{} for _ in range(8)])
    facts=extract_local_evidence(s,np.zeros((1,3)),[1.])
    assert not facts.negative_known[0,2]
    assert facts.unknown_reason[0,2]&int(UnknownReason.FUTURE_TRACK_MISSING)
    assert not any(x['constraint_type']=='release' for x in facts.events)


@pytest.mark.parametrize('source',['CURRENT','CV','FUTURE'])
def test_per_actor_geometry_matches_existing_union(source):
    s=scene([dict(a=obj(i/2,velocity=(1.,0.)),b=obj(8,reliable=False)) for i in range(9)])
    times=np.array([0.,.5,.75,1.,4.]);known,unknown,valid=s.at_times(times,source)
    for t,k,u,v in zip(times,known,unknown,valid):
        actors,tubes,v2=objects_at_time(s,t,source)
        assert v==v2
        assert shapely.equals(shapely.union_all(list(actors.values())),k)
        assert shapely.equals(shapely.union_all([x[0] for x in tubes.values()]),u)


def test_enter_and_release_use_physical_time_and_same_actor_query():
    s=scene([dict(a=obj(10-i*2)) for i in range(9)])
    facts=extract_local_evidence(s,np.zeros((1,3)),[2.])
    assert any(e['constraint_type']=='enter' and e['time_window']==[0.,2.] for e in facts.events)


def test_recording_roles_keep_adjacent_fragments_together_and_exclude_dev():
    rows=[dict(log=f'2021.06.14.12.00.00_veh-{i:02d}_{j:05d}_{j+10:05d}',token=f'{i}/{j}')
          for i in range(60) for j in (0,20)]
    dev=[dict(log='2021.06.14.12.00.00_veh-01_00011_00019',token='dev')]
    roles,index=split_roles(rows,dev)
    assert len(index['excluded_dev_recording'])==2
    used={}
    for role,group in index.items():
        for row in group:
            r=recording(row['log']);assert used.setdefault(r,role)==role
    assert set(used)==set(roles)


def test_local_only_scene_updates_predictor_but_not_min_trajectory():
    import torch
    from starVLA.model.modules.decision_local.natural_error_contract import LocalTrainingTargets,local_prediction_loss,certified_repair_loss
    truth=torch.zeros(1,7,8,5);valid=torch.zeros_like(truth,dtype=torch.bool)
    valid[:,:,1,0]=True;truth[:,:,1,0]=-1
    targets=LocalTrainingTargets(truth,valid,torch.zeros_like(valid),torch.zeros_like(valid),
        torch.full((1,8,4),float('nan')),torch.zeros(1,8,dtype=torch.bool),torch.tensor([2]))
    raw=torch.zeros_like(truth,requires_grad=True)
    loss=local_prediction_loss(raw,targets)['loss'];loss.backward()
    assert raw.grad[:,:,1,0].abs().sum()>0 and raw.grad[:,:,2:,0].abs().sum()==0
    output=torch.zeros(1,8,4,requires_grad=True)
    min_loss,_=certified_repair_loss(output,targets);min_loss.backward()
    assert min_loss.item()==0 and output.grad.abs().sum()==0


def test_empty_local_labels_finite_and_full_data_cache_rejected():
    import torch
    from starVLA.model.modules.decision_local.natural_error_contract import LocalTrainingTargets,local_prediction_loss,verify_feature_base
    truth=torch.full((2,7,8,5),float('nan'));valid=torch.zeros_like(truth,dtype=torch.bool)
    targets=LocalTrainingTargets(truth,valid,valid,valid,torch.zeros(2,8,4),
        torch.zeros(2,8,dtype=torch.bool),torch.tensor([2,4]))
    raw=torch.zeros_like(truth,requires_grad=True);loss=local_prediction_loss(raw,targets,difference_weight=1)['loss']
    loss.backward();assert loss.item()==0 and raw.grad.abs().sum()==0
    with pytest.raises(ValueError):verify_feature_base({'anchor_identity':'old_S0','checkpoint_sha256':'old'},
        {'identity':'S0_A','training_role':'A_train','checkpoint':{'sha256':'new'}})


def test_withdrawn_host_rejected_without_connection(monkeypatch):
    from tools.decision_local import common
    monkeypatch.setattr(common.socket,'gethostname',lambda:'training-vla-zt2-worker-0')
    with pytest.raises(RuntimeError):common.check_host()


def test_new_base_cache_requires_explicit_population_role(tmp_path):
    import json
    from tools.decision_local.anchor_runtime import selected_index
    from tools.decision_local.common import digest
    rows=[dict(token='a',log='recording_a_00000_00010'),dict(token='b',log='recording_b_00000_00010')]
    (tmp_path/'index.json').write_text(json.dumps(rows))
    role=tmp_path/'B_predictor.json';role.write_text(json.dumps(rows[1:]))
    anchor=dict(training_role='A_train',data={'train':{'current_root':str(tmp_path)}},
        role_indices={'B_predictor':{'path':str(role),'sha256':digest(role)}})
    with pytest.raises(ValueError):selected_index(anchor,'train')
    assert selected_index(anchor,'train','B_predictor')==rows[1:]
    role.write_text(json.dumps(rows))
    with pytest.raises(ValueError):selected_index(anchor,'train','B_predictor')


def test_old_anchor_selection_default_is_unchanged(tmp_path):
    import json
    from tools.decision_local.anchor_runtime import selected_index
    rows=[dict(token='old',log='original_00000_00010')]
    (tmp_path/'index.json').write_text(json.dumps(rows))
    assert selected_index({'data':{'train':{'current_root':str(tmp_path)}}},'train')==rows


def test_relation_evaluation_does_not_turn_unknown_nan_into_metrics():
    import torch
    from tools.decision_local.evaluate_relations import scene_metrics
    from starVLA.model.modules.decision_local.consequence_contract import CHANNELS
    raw=torch.zeros(1,7,8,5);truth=torch.full_like(raw,float('nan'))
    valid=torch.zeros_like(raw,dtype=torch.bool);poses=torch.zeros(1,7,8,3)
    metrics=scene_metrics(raw,truth,valid,poses)
    assert all(np.isfinite(v).all() for v in metrics.values())
    assert metrics[CHANNELS[0]+'_valid_points'][0]==0
