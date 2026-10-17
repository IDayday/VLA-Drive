"""Small CPU metadata fixtures; no real policy, claim, or metadata build mutation."""
import copy
import importlib.util
from pathlib import Path
import pytest

s=importlib.util.spec_from_file_location('future_controls',Path(__file__).with_name('prepare_controls.py'))
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def policy():
    return {'metadata_package':{'schema':'st_metadata_storage_package_v1','maximum_bytes':4*m.GiB,
        'baseline_policy_sha256':'a'*64,'baseline_ledger_sha256':'b'*64,
        'baseline_claims':{},'baseline_legacy_reservations':[]},
        'claims':[{'root':'/fixture/existing','opaque_identity':'retain'}],
        'legacy_reservations':[{'root':'/fixture/legacy','reserved_bytes':12}],
        'frozen_w_package':{'unchanged':'must remain byte equal'},'cache_cap_bytes':768*m.GiB}


def test_exact_v2_envelope_preserves_previous_and_all_old_claims():
    old=policy();before=copy.deepcopy(old);addition={'root':'/fixture/future','opaque_identity':'new'}
    new=m.extend_policy(old,[addition])
    assert old==before
    assert new['claims']==before['claims']+[addition]
    assert new['frozen_w_package']==old['frozen_w_package']
    assert new['legacy_reservations']==old['legacy_reservations']
    assert new['metadata_package']['previous_package_sha256']==m.digest(old['metadata_package'])
    assert new['metadata_package']['maximum_bytes']==8*m.GiB
    assert new['cache_cap_bytes']==old['cache_cap_bytes']
    assert m.extend_policy(new,[{'root':'/fixture/index'}])['metadata_package']==new['metadata_package']


@pytest.mark.parametrize('case',['claim_replace','duplicate_new','wrong_old_cap','unknown_package'])
def test_no_silent_old_identity_changes(case):
    old=policy();claims=[{'root':'/fixture/new'}]
    if case=='claim_replace':claims=[{'root':'/fixture/existing'}]
    elif case=='duplicate_new':claims*=2
    elif case=='wrong_old_cap':old['metadata_package']['maximum_bytes']=8*m.GiB
    else:old['metadata_package']['schema']='other'
    with pytest.raises(ValueError):m.extend_policy(old,claims)


def test_all_nine_existing_metadata_files_preserved(tmp_path):
    (tmp_path/'control').mkdir()
    for i in range(9):(tmp_path/'control'/f'{i}.json').write_text('{}')
    pins,dirs=m.existing_pins(tmp_path)
    assert len(pins)==9 and dirs==['control']
    assert len(list((tmp_path/'control').iterdir()))==9
    assert all(len(p['sha256'])==64 for p in pins)


def test_preexisting_count_over_bound_fails_without_deleting(tmp_path):
    for i in range(17):(tmp_path/f'{i}.json').write_text('{}')
    with pytest.raises(ValueError,match='16-pin'):m.existing_pins(tmp_path)
    assert len(list(tmp_path.iterdir()))==17


@pytest.mark.parametrize('kind',['tensor','symlink','directory_alias'])
def test_preexisting_inputs_cannot_hide_payloads(tmp_path,kind):
    if kind=='tensor':(tmp_path/'bad.pt').write_text('fixture')
    elif kind=='symlink':(tmp_path/'bad.json').symlink_to('/tmp/unknown')
    else:(tmp_path/'bad').symlink_to('/tmp',target_is_directory=True)
    with pytest.raises(ValueError):m.existing_pins(tmp_path)


def test_output_budget_rejects_before_file_creation(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'WRITE_BYTES',768*1024)
    with pytest.raises(ValueError,match='output budget'):m.write(tmp_path,'extra.json',{})
    assert not (tmp_path/'extra.json').exists()
