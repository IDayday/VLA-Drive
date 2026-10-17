"""Independent CPU metadata fixtures; no live W payload, model or dispatch."""
import ast
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import subprocess

import pytest
from tests.st_world.test_reconciled_w_artifact import fixture, r, s, b, put, validate


@pytest.mark.parametrize('field,value',[('attempt_count_preserved',1),('additional_extraction_attempts',1),
    ('role_limit_bytes',512<<30),('remaining_role_read_bytes_upper_bound_before_small_metadata',256<<30)])
def test_receipt_cannot_reset_or_expand_attempt_budget(fixture,field,value):
    f=fixture;body=deepcopy(f['body']);body['upstream_accounting'][field]=value
    body['identity']=r.identity({k:v for k,v in body.items() if k!='identity'})
    bad=put(f['job']/'badbudget.json',body)
    with pytest.raises(ValueError):validate(f,bad)


@pytest.mark.parametrize('field,value',[('status','COMPLETE'),('first_monitor_error',{'type':'TimeoutError'}),
                                      ('cleanup',True),('limit_exceeded','RSS')])
def test_resigned_fixture_cannot_change_failed_attempt_semantics(fixture,field,value,monkeypatch):
    f=fixture;failed=deepcopy(f['failed']);review=deepcopy(f['review'])
    if field=='status':failed['status']=value
    elif field=='cleanup':failed['resources']['process_cleanup']['live_owned_remaining']=value
    elif field=='limit_exceeded':failed['resources']['counters'][field]=value
    else:failed['resources'][field]=value
    pin=put(Path(f['failed_pin']['path']),failed)
    review['compute_resources']=failed['resources'];review['input_pins_unchanged'][pin['path']]['sha256']=pin['sha256']
    rp=put(Path(f['review_pin']['path']),review)
    monkeypatch.setattr(r,'FAILED_SHA',pin['sha256']);monkeypatch.setattr(r,'REVIEW_SHA',rp['sha256'])
    with pytest.raises(ValueError):r.create_receipt(rp,pin,f['prep'],f['cache'],b.read)


def test_current_stat_detects_same_size_payload_replacement(fixture):
    f=fixture;p=f['cache']/'token0.safetensors';before=p.stat()
    replacement=p.with_suffix('.replacement');replacement.write_bytes(b'X'*before.st_size)
    os.utime(replacement,ns=(before.st_atime_ns,before.st_mtime_ns));replacement.replace(p)
    with pytest.raises(ValueError):validate(f)


def test_fixture_creation_gap_is_not_presented_as_content_revalidation(fixture):
    # No historical stat was recorded by the original full-payload review.
    # The adapter explicitly snapshots NOW and keeps later read_world mandatory.
    f=fixture;p=f['cache']/'token0.safetensors';p.write_bytes(b'changed before snapshot')
    candidate=r.create_receipt(f['review_pin'],f['failed_pin'],f['prep'],f['cache'],b.read)
    assert candidate['payload_content_revalidated_here'] is False
    assert candidate['later_fit_still_requires_original_read_world_hash_validation'] is True
    assert candidate['launch_approved'] is False and candidate['new_resource_authorization'] is False
    assert candidate['payload_snapshots'][0]['payload_sha256']==f['review']['records'][0]['payload_sha256']


def test_reconciliation_does_not_replace_resource_admission(fixture):
    f=fixture
    a={'previous_stage_result':f['failed_pin'],'previous_artifact_reconciliation':f['receipt_pin'],
       'preparation':f['prep']}
    # Prerequisite passes, but the actual supervisor resource gate still rejects.
    s.stage_inputs(a,'train_plan')
    with pytest.raises(ValueError,match='independent stage/resource admission'):
        s.validate(a,{},'train_plan',0)
    assert not (f['job']/'stages/train_plan').exists()


def test_precise_review_anchor_and_unmodified_launch_controls():
    root=Path(__file__).resolve().parents[5]
    report=root/'reports/st_world_policy/20261017/correctness/sp0_dev_artifact_reconciliation_v1/ARTIFACT_RECONCILIATION.json'
    assert hashlib.sha256(report.read_bytes()).hexdigest()==r.REVIEW_SHA
    rel='reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1/supervise_once.py'
    prior=ast.parse(subprocess.check_output(['git','show','2a7407ab^:'+rel],text=True))
    current=ast.parse((root/rel).read_text())
    def functions(tree):return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    a,c=functions(prior),functions(current)
    assert set(a)==set(c)
    assert {k for k in a if a[k]!=c[k]}=={'check_manual_fields','stage_inputs','validate'}
    # main's one-shot directory reservation, locking, Popen and final result
    # creation are unchanged; only prerequisite resolution and helper pin grew.
