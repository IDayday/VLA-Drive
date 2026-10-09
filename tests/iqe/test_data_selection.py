from dataclasses import replace, asdict
import numpy as np
import pytest
import torch
from iqe.contracts import strict_record, SceneRecord, FeatureRecord, CandidateRecord, ScoreRecord
from iqe.data.splits import validate_isolation, split_training_logs
from iqe.data.manifests import freeze_targets, save_scenes, load_scenes
from iqe.data.sampler import mixture, sampling_plan, ConsumedSampler
from iqe.data.feature_cache import FeatureCache
from iqe.data.candidate_bank import CandidateBank, save_trajectory
from iqe.data.mining import diagnose_scene, MiningThresholds
from iqe.data.support_retrieval import retrieve
from iqe.model import CandidateBatch
from iqe.scorer import ScorePrediction
from iqe.selector import SelectionRule, RouterRule
from iqe.evaluation.calibration import calibrate
from iqe.evaluation.oracle import oracle_report
from iqe.evaluation.selection import selection_report
from iqe.io import digest, read_json, atomic_json
from conftest import scene

NC="no_at_fault_collisions"; DAC="drivable_area_compliance"


def score(value=.9, nc=1.,dac=1.,key="x",valid=True):
    return ScoreRecord(1,key,"p","ctx","ref",value if valid else None,{NC:nc,DAC:dac},{NC:valid,DAC:valid},"reference","v1",42,0,valid,None if valid else "failed")


def test_splits_log_parent_and_observation_leakage():
    records=[scene(i) for i in range(100)]
    a=split_training_logs(records);b=split_training_logs(list(reversed(records)))
    assert {s.scene_id:s.split_role for s in a}=={s.scene_id:s.split_role for s in b}
    for cases in ([scene(1),scene(2,"dev_report",group="g1",parent_group_id="g1")],
                  [scene(1),scene(2,"selector_cal",observation_hash="ob1")],
                  [scene(1,group="log"),scene(2,"final_test",group="log")]):
        with pytest.raises(ValueError,match="leakage"):validate_isolation(cases)
    with pytest.raises(ValueError):scene(1,"dev_report").eligible_input()


def test_strict_manifest_and_targets(tmp_path):
    rows=[scene(1),scene(2)]
    save_scenes(tmp_path/"a.json",rows)
    assert load_scenes(tmp_path/"a.json")==rows
    with pytest.raises(ValueError,match="unknown"):strict_record(SceneRecord,asdict(rows[0])|{"leak":1})
    with pytest.raises(ValueError,match="conflicting"):freeze_targets([scene(1),scene(2,observation_hash="ob1")])
    chosen,queue=freeze_targets([scene(1),scene(2,source_kind="synthetic",input_consistency_status="unverified"),scene(3,"dev_report")])
    assert chosen==[scene(1)] and len(queue)==2
    chosen,queue=freeze_targets([scene(1),scene(2,observation_hash="ob1",target_provenance="external_teacher")],external_targets=True)
    assert chosen[0].target_provenance=="gt"
    with pytest.raises(ValueError):scene(1,source_kind="synthetic").eligible_input()


def test_buckets_missing_and_cumulative_exposure():
    buckets={"hard_original":[scene(i) for i in range(3)],"new_labeled":[],"local_support":[],"global_anchor":[scene(10)]}
    ratios={"hard_original":.4,"new_labeled":.4,"local_support":.1,"global_anchor":.1}
    plan=sampling_plan(buckets,ratios,1000)
    assert plan["metadata"]["actual_proportions"]=={"hard_original":.9,"new_labeled":0.,"local_support":0.,"global_anchor":.1}
    assert plan["metadata"]["experiment_kind"]=="resampling_only"
    exposures=plan["metadata"]["buckets"]["hard_original"]["exposures_by_group"]
    assert max(exposures.values())-min(exposures.values())<=1
    assert sum(plan["metadata"]["exposures_by_scene"].values())==1000
    sampler=ConsumedSampler(plan);prefetched=sampler.peek(5);assert sampler.peek(5)==prefetched
    sampler.consume(5); state=sampler.state_dict();other=ConsumedSampler(plan);other.load_state_dict(state)
    assert other.peek(5)==sampler.peek(5)
    buckets["global_anchor"]=[]
    with pytest.raises(ValueError,match="global_anchor"):mixture(buckets,ratios)
    zero=ratios|{"hard_original":.5,"global_anchor":0.}
    with pytest.raises(ValueError,match="explicit"):mixture(buckets,zero)
    assert mixture(buckets,zero,explicit_anchor_zero=True)[0]["hard_original"]==1.


def test_feature_cache_hash_checksum_augmentation(tmp_path,features):
    features = features.subset([0])
    record=FeatureRecord(1,"a","ob","ctx","s0","shared","tok","prompt","transform","norm","camera","float32","fp32","none",0,"",{},True,None,"")
    cache=FeatureCache(tmp_path)
    cache.put(record,features)
    restored=cache.get(record)
    assert torch.equal(restored.scene,features.scene)
    for changed in (replace(record,normalizer_hash="wrong"),replace(record,augmentation_seed=1)):
        with pytest.raises(FileNotFoundError):cache.get(changed)
    receipt=read_json(tmp_path/record.cache_key/"COMPLETE.json");receipt["checksum"]="wrong";atomic_json(tmp_path/record.cache_key/"COMPLETE.json",receipt)
    with pytest.raises(ValueError,match="checksum"):cache.get(record)


def test_candidate_bank_join_stale_score_reject(tmp_path):
    ref=tmp_path/"t.npz";th=save_trajectory(ref,np.zeros((4,3)))
    c=CandidateRecord(1,"s1","expert_0","cp","feature",th,str(ref),"xy","ego",4,.5,True,"code","conf","now")
    bank=CandidateBank(tmp_path/"bank");bank.put_candidate(c);assert bank.candidate(c.key)==c
    sc=replace(score(key=c.key),metric_context_hash="ctx1");key=bank.put_score(sc)
    assert bank.score(key,c,"p","ctx1")==sc
    for mismatch in (replace(c,expert_checkpoint_hash="new"),replace(c,trajectory_hash="wrong")):
        with pytest.raises(ValueError,match="stale"):bank.score(key,mismatch,"p","ctx1")
    with pytest.raises(ValueError):ScoreRecord(1,c.key,"p","ctx1","ref",0.,{}, {},"r","v1",1,0,False,"error")
    with pytest.raises(ValueError):replace(sc,total_score_01=90.)


@pytest.mark.parametrize("old,target,selected,coverage,selection,target_gap,eligible",[
    ([score(.5)],score(.9),"expert_0",True,False,False,True),
    ([score(.5),score(.9)],score(.9),"expert_0",False,True,False,False),
    ([score(.5)],score(.5),"expert_0",True,False,True,False),
    ([score(.9,nc=0.)],score(.9),"expert_0",True,False,False,True),
    ([score(.9)],score(.9),"expert_0",False,False,False,False),
])
def test_gap_separation(old,target,selected,coverage,selection,target_gap,eligible):
    audit={"finite":True,"coordinates_time_yaw_legal":True,"input_context_consistent":True}
    result=diagnose_scene(scene(),{f"expert_{i}":s for i,s in enumerate(old)},target,selected,audit,MiningThresholds())
    assert result["coverage_gap"]==coverage and result["selection_gap"]==selection and result["target_gap"]==target_gap
    assert result["eligible_for_expert_training"]==eligible


def test_retrieval_excludes_groups_and_heldout():
    q=scene(1)
    candidates=[scene(2),scene(3,group="g1")]
    found,report=retrieve(q,np.array([0.,0.]),candidates,[np.array([1.,0.]),np.array([0.,0.])],2,"v1")
    assert [s.scene_id for s in found]==["s2"]
    with pytest.raises(ValueError):retrieve(q,np.array([0.,0.]),[scene(2,"dev_report")],[np.array([0.,0.])],2,"v1")


def candidates(values=(.5,.9,.9),ids=("expert_0","expert_2","expert_1")):
    tau=torch.arange(len(values)*12).float().reshape(1,len(values),4,3)
    c=CandidateBatch(tau,tau,ids,torch.ones(1,len(values),dtype=torch.bool),None)
    pred=ScorePrediction(torch.tensor([values]),{NC:torch.ones(1,len(values))*5,DAC:torch.ones(1,len(values))*5},c.valid.clone())
    return c,pred


def test_selector_tie_margin_safety_invalid_base():
    c,p=candidates()
    rule=SelectionRule(.1,{NC:.9,DAC:.9},{NC:.02,DAC:.02})
    assert rule(c,p)[1]==["expert_1"]
    p.component_logits[NC][0,2]=-5
    assert rule(c,p)[1]==["expert_2"]
    c.valid[0,1]=False
    assert rule(c,p)[1]==["expert_0"]
    c.valid[0,0]=False
    with pytest.raises(ValueError,match="Base invalid"):rule(c,p)
    c,p=candidates((.9,.9,.9));assert rule(c,p)[1]==["expert_0"]
    assert RouterRule()(torch.ones(1,3),c.expert_ids).item()==0


def test_calibration_role_and_no_fake_release():
    c,p=candidates((.5,.9), ("expert_0","expert_1"))
    config={"sampling":"natural_scene_distribution","protected_metrics":[NC,DAC],"safety_relative_tolerance":.02,"delta_grid":[0,.1],"eta_grid":[0,.9],
            "budgets":{"false_replacement_rate":0.,"high_to_zero_rate":0.,"new_hard_safety_violation_rate":0.}}
    kwargs=dict(true_scores=np.array([[.9,.1]]),components={NC:np.ones((1,2)),DAC:np.ones((1,2))},component_valid={NC:np.ones((1,2),bool),DAC:np.ones((1,2),bool)},previous_indices=np.zeros(1,int),config=config,dependency_hash="pool")
    with pytest.raises(ValueError):calibrate(c,p,role="dev_report",**kwargs)
    rule,report=calibrate(c,p,role="selector_cal",**kwargs)
    assert rule.always_base and not report["beneficial"]
    kwargs["true_scores"]=np.array([[.5,.9]])
    rule,report=calibrate(c,p,role="selector_cal",**kwargs)
    assert not rule.always_base and report["beneficial"]


def test_selection_capture_na_negative_and_regression():
    scores=np.array([[.8,.8],[.9,.9]])
    valid=np.ones_like(scores,bool);cm={NC:np.ones_like(scores),DAC:np.ones_like(scores)};mask={c:valid for c in cm}
    report=selection_report(scores,valid,np.array([0,1]),np.array([0,0]),cm,mask,["a","b"])
    assert report["oracle_capture"] is None
    scores=np.array([[.8,.9,.1],[.8,.9,.1]])
    valid=np.ones_like(scores,bool);cm={NC:np.ones_like(scores),DAC:np.ones_like(scores)};mask={c:valid for c in cm}
    report=selection_report(scores,valid,np.array([2,2]),np.array([1,1]),cm,mask,["a","b"])
    assert report["oracle_capture"]<0 and report["vs_previous"]["mean_gain_01"]<0
