from copy import deepcopy
import numpy as np
import pytest
from iqe.specialist_study.target_rules import eligibility
from iqe.specialist_study.data import failure_specialties,admissible_optimized_source
from iqe.specialist_study.bank import canonical_comparison_mask,trajectory_variants
from iqe.scoring.reference import COMPONENTS


def score(value=1.):return {'score_valid':True,'total_score_01':value,'named_components':dict.fromkeys(COMPONENTS,1.),'component_valid_masks':dict.fromkeys(COMPONENTS,True)}
def record():return {'gt':score(.95),'optimized':score(1.),'base':score(.4),'gt_geometry':{'valid':True},
    'optimized_geometry':{'valid':True,'historical_strict_pass':True},'optimized_provenance':{'source':'gt:quintic:official_refinement'}}


def test_sources_and_target_quality_differ_from_negative_bank_candidates():
    r=record();assert eligibility(r)['paired_eligible']
    r['optimized_provenance']['source']='diffusiondrivev2:quintic';assert not eligibility(r)['paired_eligible']
    assert eligibility(r)['gt_opportunity']
    for source in ['drivor','parent_construction:drivor','parent_proposal','previous_label','margin_current','zero_current']:
        assert not admissible_optimized_source(source)
    r=record();r['gt']['named_components']['drivable_area_compliance']=0
    assert not eligibility(r)['gt_eligible']
    r=record();r['gt']['component_valid_masks']['no_at_fault_collisions']=False
    assert not eligibility(r)['paired_eligible']


def test_ddc_zero_weight_and_failure_groups_are_explicit():
    r=score(.95);r['named_components']['driving_direction_compliance']=0
    assert failure_specialties(r)==['road_direction']
    r['named_components']['no_at_fault_collisions']=0
    assert failure_specialties(r)==['interaction','road_direction']
    r['component_valid_masks']['drivable_area_compliance']=False
    assert failure_specialties(r)==[]


def test_identical_candidate_does_not_create_additional_ranking_information():
    t=np.zeros((1,3,8,3),np.float32);t[:,2,:,0]=1
    v=np.ones((1,3),bool);scores=np.array([[.9,.9,.1]])
    assert canonical_comparison_mask(t,v,scores).tolist()==[[True,False,True]]
    with pytest.raises(Exception,match='labels changed'):canonical_comparison_mask(t,v,np.array([[.9,.8,.1]]))


def test_rich_variants_no_future_extrapolation_and_correct_time_geometry():
    p=np.zeros((8,3),np.float32);p[:,0]=np.arange(1,9)*2
    variants=trajectory_variants(p,.5)
    assert len(variants)==4
    assert all(x.shape==(8,3) and np.isfinite(x).all() for _,x in variants)
    assert all(x[-1,0]<=p[-1,0]+1e-6 for _,x in variants)
    assert len({x.tobytes() for _,x in variants})==4
