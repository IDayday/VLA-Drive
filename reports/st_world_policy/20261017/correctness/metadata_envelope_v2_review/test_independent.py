"""Independent CPU temporary-fixture checks; no live ledger or assets."""
import copy
import hashlib
import json

import pytest

from tools.st_world_policy import cache_storage_budget as b
from tools.st_world_policy import metadata_storage as m
from tools.st_world_policy import spatial_w_storage as w
from tests.st_world.test_metadata_envelope_v2 import setup, expansion
from tests.st_world.test_metadata_storage_guard import write


def followup(tmp_path):
    f, (old, _, _, _, first) = setup(tmp_path)
    m.migrate_policy(**first)
    current = b.StorageBudget(first['new_policy'])
    before = current.state_path.read_bytes()
    doc = copy.deepcopy(current.policy)
    kw = copy.deepcopy(first)
    kw.update(old_policy=first['new_policy'],
              old_policy_sha256=hashlib.sha256(first['new_policy'].read_bytes()).hexdigest(),
              new_policy=tmp_path/'v2-following.json',
              expected_old_ledger_sha256=hashlib.sha256(before).hexdigest(),
              quiescence_receipt=tmp_path/'v2-following-quiescence.json')
    q = json.loads(first['quiescence_receipt'].read_text())
    return f, current, before, doc, q, kw


def seal(doc, q, kw):
    kw['new_policy_sha256'] = write(kw['new_policy'], doc)
    q['bindings'].update({k:kw[k] for k in ('old_policy_sha256','new_policy_sha256')})
    q['bindings']['old_ledger_sha256'] = kw['expected_old_ledger_sha256']
    kw['quiescence_sha256'] = write(kw['quiescence_receipt'], q)


def repin_parent(package):
    parent = {k:v for k,v in package.items() if k!='previous_package_sha256'}
    parent.update(schema=m.PACKAGE_SCHEMA,maximum_bytes=m.PACKAGE_MAX)
    package['previous_package_sha256'] = b._digest(parent)


@pytest.mark.parametrize('mutation', ['baseline_resealed','parent','cap','extra_resealed',
                                     'old_claim','old_legacy'])
def test_v2_followup_cannot_mutate_prior_identity(tmp_path, mutation):
    _, current, before, doc, q, kw = followup(tmp_path)
    p = doc['metadata_package']
    if mutation=='baseline_resealed':
        p['baseline_policy_sha256']='9'*64; repin_parent(p)
    if mutation=='parent': p['previous_package_sha256']='9'*64
    if mutation=='cap': p['maximum_bytes']=m.FULL_METADATA_PACKAGE_MAX+1
    if mutation=='extra_resealed': p['unreviewed_override']=True; repin_parent(p)
    if mutation=='old_claim': doc['claims'][0]['output_bytes']+=4096
    if mutation=='old_legacy': doc['legacy_reservations']=[]
    seal(doc,q,kw)
    with pytest.raises(b.StorageBudgetError): m.migrate_policy(**kw)
    assert current.state_path.read_bytes()==before


@pytest.mark.parametrize('kind',['wrong_hash','different_pinned_policy'])
def test_wrong_old_policy_even_with_new_receipt_rejected(tmp_path,kind):
    _, current, before, doc, q, kw = followup(tmp_path)
    if kind=='wrong_hash': kw['old_policy_sha256']='f'*64
    else:
        old_doc=copy.deepcopy(current.policy)
        old_doc['metadata_package']['baseline_policy_sha256']='f'*64
        repin_parent(old_doc['metadata_package'])
        kw['old_policy']=tmp_path/'foreign-old.json'
        kw['old_policy_sha256']=write(kw['old_policy'],old_doc)
        doc=copy.deepcopy(old_doc)
    seal(doc,q,kw)
    with pytest.raises(b.StorageBudgetError): m.migrate_policy(**kw)
    assert current.state_path.read_bytes()==before


def test_same_v2_additive_claim_preserves_envelope_and_old_claims(tmp_path):
    f,current,before,doc,q,kw=followup(tmp_path)
    reg=copy.deepcopy(f.reg); reg['output_root']=str(tmp_path/'cache'/'new-future-index')
    reg['writer_kind']='future_index'
    regpath=tmp_path/'new-index-registration.json'; regsha=write(regpath,reg)
    row=dict(claim_id='new-future-index',root=reg['output_root'],cache_identity=regsha,
             identity_kind=m.REG_SCHEMA,registration_path=str(regpath),registration_sha256=regsha,
             **reg['storage_requirements'])
    doc['claims'].append(row); seal(doc,q,kw)
    result=m.migrate_policy(**kw)
    assert result==m.migrate_policy(**kw)
    after=json.loads(current.state_path.read_bytes()); prior=json.loads(before)
    assert after['core']['metadata_package']==prior['core']['metadata_package']
    assert after['claims']==prior['claims']
    assert result['metadata_package_reserved_bytes']==7*(1<<20)
    # The real guard must reserve this new registration; no general writer bypass.
    with m.metadata_writing(policy_path=kw['new_policy'],policy_sha256=kw['new_policy_sha256'],
            registration_path=regpath,registration_sha256=regsha,
            checkpoint_receipt=kw['checkpoint_receipt'],checkpoint_sha256=kw['checkpoint_sha256']):
        state=json.loads(current.state_path.read_bytes())
        assert state['claims'][row['root']]['charged_bytes']==3*(1<<20)
    assert b.StorageBudget(kw['new_policy']).initialize()['guarded_claims']==2


def test_frozen_w_rejects_resealed_foreign_metadata_parent(tmp_path):
    from tests.st_world.test_spatial_w_storage import fixture
    f=fixture(tmp_path);m.migrate_policy(**f.kwargs)
    old,_,_,_,kw=expansion(tmp_path,f.new_path,f.kwargs['checkpoint_receipt'],f.kwargs['checkpoint_sha256'])
    m.migrate_policy(**kw)
    current=b.StorageBudget(kw['new_policy'])
    core=copy.deepcopy(current.core)
    core['metadata_package']['baseline_policy_sha256']='f'*64
    repin_parent(core['metadata_package'])
    with pytest.raises(b.StorageBudgetError,match='frozen W changed original metadata package'):
        w.validate_package(core,current.declarations)


def test_per_writer_limit_not_raised_by_package_v2(tmp_path):
    f, *_=followup(tmp_path)
    reg=copy.deepcopy(f.reg);reg['limits']['max_disk_bytes']=4*b.GIB+1
    with pytest.raises(b.StorageBudgetError,match='resource limit'):
        m._resources(reg)
