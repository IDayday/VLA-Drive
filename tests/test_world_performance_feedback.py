from types import SimpleNamespace
import copy
from starVLA.rl.qwen_world_performance import feedback as F
from starVLA.rl.qwen_world_performance.data_protocol import load_feedback
from starVLA.rl.simwam_recipe import checkpoint


def leaf(token='train',dac=1):
    version=dict(run='own-run',update=0,adapter_sha256='actual-model-sha')
    metrics=dict(no_at_fault_collisions=1.,drivable_area_compliance=float(dac),time_to_collision_within_bound=1.,ego_progress=.8,comfort=1.,score=float(dac)*11/12)
    feedback=dict(status='OK',token=token,log='legal-train-log',protocol_sha256='p'*64,metrics=metrics,trajectory_hash='requested-sha',cache=dict(sha256='cache-sha'))
    return dict(token=token,log='legal-train-log',protocol_sha256='p'*64,valid_scene=True,records=[feedback],nominal_improvement=dict(enabled=False,nominal=dict(source_version=version,feedback=feedback)))


def runtime(tmp_path):
    offered=[];stream=SimpleNamespace(logs={'train':'legal-train-log'},offer_feedback=lambda p:offered.append(str(p)))
    return SimpleNamespace(spec=dict(population=4),training_identity=dict(sha256='own-run'),args=SimpleNamespace(output=str(tmp_path)),stream=stream),offered


def test_actual_observation_age_and_history_survive_snapshot_and_native_state(tmp_path):
    r,offered=runtime(tmp_path);state=dict(valid_scene_exposures=2,performance_auxiliary={})
    sample=leaf();window=dict(behavior_version=sample['nominal_improvement']['nominal']['source_version'],window_start_cursor=0,valid_global_scenes=2,samples=[sample,dict(sample,valid_scene=False)])
    F.accumulate_nominal_feedback(state,window);out=F.flush_feedback(r,window,state)
    first=load_feedback(out['snapshot']['path'],logs=r.stream.logs,run_identity='own-run')
    assert first['records']['train']['observed_valid_scene_clock']==0 and len(offered)==1
    # New publication must not pretend the older observation was just rescored.
    memory=state['performance_auxiliary']['priority_feedback'];assert memory['snapshot_material']==first
    restored=copy.deepcopy(state);assert F.feedback_state_validation(restored)==F.feedback_state_validation(state)
    state['valid_scene_exposures']=4;sample=leaf(dac=0);window.update(window_start_cursor=2,samples=[sample]);window['behavior_version']['update']=4;sample['nominal_improvement']['nominal']['source_version']['update']=4
    F.accumulate_nominal_feedback(state,window);out=F.flush_feedback(r,window,state)
    second=load_feedback(out['snapshot']['path'],logs=r.stream.logs,run_identity='own-run')
    assert second['records']['train']['score']==0 and second['records']['train']['best_own_score']> .9
    assert second['records']['train']['ever_strict_feasible'] and second['records']['train']['observed_valid_scene_clock']==2
    assert len(offered)==2 and memory['actual_local_B1_observations']==2


def test_rank_with_no_records_still_joins_half_epoch_collective(tmp_path,monkeypatch):
    r,offered=runtime(tmp_path);state=dict(valid_scene_exposures=2,performance_auxiliary={});sample=leaf();window=dict(behavior_version=sample['nominal_improvement']['nominal']['source_version'],window_start_cursor=0,valid_global_scenes=2,samples=[])
    F.accumulate_nominal_feedback(state,window)
    row=dict(log='legal-train-log',score=11/12,metrics=sample['nominal_improvement']['nominal']['feedback']['metrics'],observed_valid_scene_clock=0,source_behavior_update=0,source_model_sha256='actual-model-sha',feedback_material_sha256='f'*64)
    calls=[]
    def gather(value):
        calls.append(value)
        if isinstance(value,dict)and'clock'in value:return[value,dict(value,rows=1)]
        if isinstance(value,dict)and'cache'in value:return[value,dict(cache={'train':row},history={'train':dict(best_own_score=11/12,ever_strict_feasible=True,last_observed_valid_scene_clock=0)})]
        return[value,value]
    monkeypatch.setattr(checkpoint,'gather',gather)
    out=F.flush_feedback(r,window,state)
    assert out['flushed']and len(offered)==1
    assert any(isinstance(v,dict)and v.get('cache')=={}for v in calls)
