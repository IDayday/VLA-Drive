"""Native evidence boundaries, identity joins and scene transactions."""
import copy
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from starVLA.model.modules.qwen_world_compare.contracts import first_event_class
from starVLA.model.modules.qwen_world_postcot.candidate_bank import CandidateRecord, trajectory_hash, OFFICIAL_METRIC_NAMES, fingerprint
from starVLA.rl.qwen_world_compare.evidence import (
    EvidenceWriter, EvidenceBank, make_evidence_record, merge_evidence, validate_record,
)


def fake(scene='s1', offset=0., times=(None, None, None), nc=1.):
    trajectory = np.zeros((8, 3), dtype=np.float32);trajectory[:, 0] = np.arange(8)+offset
    road = np.zeros(41, dtype=bool)
    if times[1] is not None:
        road[round(times[1]*10):] = True
    metrics = dict(zip(OFFICIAL_METRIC_NAMES, [nc, float(times[1] is None), float(times[2] is None), .5, 1., 1.]))
    # OFFICIAL_METRIC_NAMES is the source of ordering; set named values explicitly.
    metrics.update(no_at_fault_collisions=nc, drivable_area_compliance=float(times[1] is None),
        time_to_collision_within_bound=float(times[2] is None), ego_progress=.5, comfort=1., driving_direction_compliance=1.)
    score = metrics['no_at_fault_collisions']*metrics['drivable_area_compliance']*(2.5+5*metrics['time_to_collision_within_bound']+2)/12
    c = CandidateRecord(scene_id=scene,source_log='log',split='TRAIN',timestamp_us=1000,
        observation_hash='obs'+scene,trajectory_fp32=trajectory.tolist(),trajectory_sha256=trajectory_hash(trajectory),
        raw_factor_labels=metrics,valid_masks={k:True for k in OFFICIAL_METRIC_NAMES},raw_total_score=score,
        score_protocol_sha256='proto',evaluator_revision={'native':'source'},metric_version='v1.1',context_sha256='cache',
        candidate_source='logged_gt',actor_checkpoint_sha256=None,sampling={}).validate()
    f=dict(token=scene,log='log',status='OK',protocol_sha256='proto',cache=dict(sha256='cache',path='mock'),
        metrics=dict(metrics,score=score),requested_trajectory=trajectory.copy(),simulated_states=np.zeros((41,11)),
        simulated_body_coordinates=np.zeros((41,5,2)),non_drivable_body_by_time=road,
        official_first_at_fault_collision_seconds=times[0],official_first_ttc_infraction_seconds=times[2],
        raw_progress_meters=8.,scoring_seconds=.2)
    evaluator={'native_protocol':'proto'}
    v=make_evidence_record(c,f,source_bank_hash='bank',evaluator_identity=evaluator)
    identity=dict(source_bank_hash='bank',score_protocol_sha256='proto',evaluator_sha256=fingerprint(evaluator))
    return c,f,v,identity


@pytest.mark.parametrize('seconds,expected',[(None,8),(0.,0),(.1,0),(.5,0),(.5000000000001,0),(.6,1),(1.,1),(3.5,6),(4.,7)])
def test_event_bins_are_physical_time_not_denoising_step(seconds,expected):
    assert first_event_class(seconds)==expected


def test_partial_NC_is_an_official_event_and_road_uses_time_mask():
    _,_,v,_=fake(times=(1.3,.5,.6),nc=.5)
    assert v['event_class']==[2,0,1] and v['event_first_seconds']==[1.3,.5,.6]
    assert v['factor_labels']['no_at_fault_collisions']==.5
    assert len(v['native_non_drivable_body_by_time'])==41


def test_no_event_is_known_none_not_missing_feedback():
    c,f,v,_=fake();assert v['event_class']==[8,8,8] and v['event_valid']==[True]*3
    del f['official_first_at_fault_collision_seconds']
    with pytest.raises(KeyError):make_evidence_record(c,f,source_bank_hash='bank',evaluator_identity={})


@pytest.mark.parametrize('field', ['trajectory','cache','factor','event','road'])
def test_wrong_or_inconsistent_native_evidence_is_refused(field):
    c,f,_,_=fake()
    if field=='trajectory':f['requested_trajectory'][0,0]+=.1
    if field=='cache':f['cache']['sha256']='different'
    if field=='factor':f['metrics']['comfort']=0.
    if field=='event':f['official_first_at_fault_collision_seconds']=1.
    if field=='road':f['non_drivable_body_by_time']=np.zeros(40,dtype=bool)
    with pytest.raises(ValueError):make_evidence_record(c,f,source_bank_hash='bank',evaluator_identity={})


def test_uncommitted_scene_cannot_publish_and_resume_preserves_commits(tmp_path):
    _,_,v,identity=fake();root=tmp_path/'part'
    with EvidenceWriter(root,identity,['s1','s2'])as w:
        key=w.begin_attempt('s1',1)
        w.finish_attempt(key,status='VALIDATED_SCORED',returned_candidates=1,wall_seconds=.3,simulator_batches=1)
        w.append_scene('s1',[v])
        w.begin_attempt('s2',1)
        with pytest.raises(ValueError):w.finish()
    assert not(root/'COMPLETE.json').exists()
    with EvidenceWriter(root,identity,['s1','s2'],resume=True)as w:
        assert w.has_scene('s1') and not w.has_scene('s2')
        assert w.accounting()['interrupted_partial_calls_unknown_cost']==1
        _,_,v2,_=fake('s2');w.append_scene('s2',[v2]);m=w.finish()
    assert m['counts']==dict(scenes=2,records=2)
    bank=EvidenceBank(root,source_bank_hash='bank')
    assert bank.get(v['record_id'])==v and bank.scene_records('s1')==[v]
    bank.close()


def test_scene_insert_failure_is_atomic(tmp_path):
    _,_,v,identity=fake();_,_,v2,_=fake(offset=2.)
    with EvidenceWriter(tmp_path/'part',identity,['s1'])as w:
        # Fault injection after the first insertion, inside the same transaction.
        w.db.execute(f"CREATE TRIGGER fail_second BEFORE INSERT ON records WHEN NEW.record_id='{v2['record_id']}' BEGIN SELECT RAISE(ABORT,'injected'); END")
        w.db.commit()
        with pytest.raises(sqlite3.IntegrityError):w.append_scene('s1',[v,v2])
        assert w.counts()==dict(scenes=0,records=0)
        w.db.execute('DROP TRIGGER fail_second');w.db.commit()
        w.append_scene('s1',[v,v2]);w.finish()


def test_part_merge_get_and_source_hash_are_sealed(tmp_path):
    _,_,a,identity=fake();_,_,b,_=fake('s2')
    parts=[tmp_path/'parts/rank00000',tmp_path/'parts/rank00001']
    for root,scene,v in zip(parts,['s1','s2'],[a,b]):
        with EvidenceWriter(root,identity,[scene])as w:w.append_scene(scene,[v]);w.finish()
    with pytest.raises(ValueError):merge_evidence(tmp_path,parts,expected_scenes=['s1','s2','missing'])
    assert not(tmp_path/'COMPLETE.json').exists()
    m=merge_evidence(tmp_path,parts,expected_scenes=['s1','s2']);assert m['counts']['records']==2
    bank=EvidenceBank(tmp_path,source_bank_hash='bank');assert bank.get(a['record_id'])==a
    assert bank.scene_records('s2')==[b]
    with pytest.raises(KeyError):bank.get('absent')
    bank.close()
    with pytest.raises(ValueError):EvidenceBank(tmp_path,source_bank_hash='other')


def test_altered_payload_does_not_pass_its_seal():
    _,_,v,identity=fake();v['event_class'][0]=0
    with pytest.raises(ValueError):validate_record(v,identity=identity)
