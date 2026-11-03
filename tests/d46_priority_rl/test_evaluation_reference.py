import json
from unittest.mock import patch
import pytest
from tools.d46_priority_rl import evaluation_reference as reference


def setup(tmp_path, *, dispatched=False):
    root=tmp_path/'operations';root.mkdir()
    proof={'parent':'same', 'trainable_payload_sha256':'same'}
    profile={'schema':'d46_explicit_equivalent_evaluation_v1',
             'pairs':[dict(alias='D1_half',reference='D0_half',alias_checkpoint='right',
                           reference_checkpoint='left',deployment_proof_sha256=reference.fingerprint(proof))],
             'priorities':{'D0_half':0}}
    (root/'EVALUATION_EQUIVALENCE.json').write_text(json.dumps(profile))
    cases={'D0_half':dict(request={'checkpoint':'left'},status='PENDING',jobs={}),
           'D1_half':dict(request={'checkpoint':'right'},status='PENDING',jobs={'0':[{}]}if dispatched else{})}
    return {'evaluations':cases},proof


def test_explicit_reference_is_unmeasured_and_never_complete(tmp_path):
    campaign,proof=setup(tmp_path)
    with patch.object(reference,'deployment_equivalence',return_value=proof):
        priorities=reference.prepare_references(campaign,tmp_path)
    row=campaign['evaluations']['D1_half']
    assert row['reference_only'] and not row['independently_evaluated']
    assert row['status']=='EQUIVALENT_REFERENCE_PENDING' and 'summary'not in row
    assert priorities=={'D0_half':0}


def test_complete_reference_remains_distinct_from_independent_complete(tmp_path):
    campaign,proof=setup(tmp_path)
    summary=tmp_path/'SUMMARY.json';summary.write_text(json.dumps(dict(qualification=False,scenes=12146,invalid_scenes=0)))
    campaign['evaluations']['D0_half'].update(status='COMPLETE',summary=str(summary))
    with patch.object(reference,'deployment_equivalence',return_value=proof):
        reference.prepare_references(campaign,tmp_path)
    alias=campaign['evaluations']['D1_half']
    assert alias['status']=='EQUIVALENT_REFERENCE_COMPLETE'
    assert 'summary'not in alias and alias['reference_summary']==str(summary)


def test_no_replacement_of_dispatched_work(tmp_path):
    campaign,_=setup(tmp_path,dispatched=True)
    with pytest.raises(ValueError,match='actually dispatched'):
        reference.prepare_references(campaign,tmp_path)


def test_changed_payload_proof_rejected(tmp_path):
    campaign,_=setup(tmp_path)
    with patch.object(reference,'deployment_equivalence',return_value={'changed':True}):
        with pytest.raises(ValueError,match='equivalence changed'):
            reference.prepare_references(campaign,tmp_path)


def test_no_profile_keeps_legacy_behavior(tmp_path):
    assert reference.prepare_references({},tmp_path)=={}
