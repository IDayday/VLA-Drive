"""Small source/contract probes only; no proposed adapter is implemented here."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import types

import pytest
from tools.st_world_policy import cache_storage_budget as b

OLD='581399eb309677c20fe7464791b4f18685e947a6'
DONOR='619707b4524b211b7dc313a9723c70195f26aa6b'
PROPOSAL=Path('reports/st_world_policy/20261017/w_train_extract_preparation_v1/STORAGE_COMPATIBILITY_PROPOSAL.json')

def source(commit,path):
    return subprocess.check_output(['git','show',commit+':'+path])

def storage(commit):
    module=types.ModuleType('independent_metadata_probe')
    exec(compile(source(commit,'tools/st_world_policy/metadata_storage.py'),'<reviewed source>','exec'),module.__dict__)
    return module

def packages():
    old=dict(schema='st_metadata_storage_package_v1',maximum_bytes=4*b.GIB,
             baseline_claims={},baseline_legacy_reservations=[],
             baseline_policy_sha256='a'*64,baseline_ledger_sha256='b'*64)
    new={**old,'schema':'st_metadata_storage_package_v2','maximum_bytes':8*b.GIB,
         'previous_package_sha256':b._digest(old)}
    return old,new

def test_exact_proposed_donor_and_all_unchanged_dependency_pins():
    proposal=json.loads(PROPOSAL.read_text())
    for row in proposal['storage_checkout_to_freeze']['only_donor_files']:
        assert hashlib.sha256(source(DONOR,row['relative_path'])).hexdigest()==row['reviewed_donor_sha256']
        assert hashlib.sha256(source(OLD,row['relative_path'])).hexdigest()==row['compute_runtime_original']['sha256']
    for row in proposal['unchanged_computation_dependency_pins']:
        assert source(OLD,row['relative_path'])==source(DONOR,row['relative_path'])
        assert hashlib.sha256(source(OLD,row['relative_path'])).hexdigest()==row['compute_runtime_sha256']

def test_actual_old_v2_rejection_and_donor_baseline_preservation():
    old,new=packages();core={'metadata_package':new,'legacy_reservations':[]}
    with pytest.raises(b.StorageBudgetError,match='unknown metadata package'):
        storage(OLD).validate_package(core,{})
    donor=storage(DONOR)
    assert donor.original_package(new)==old
    assert donor.validate_package(core,{})==0
    assert donor.validate_package({'metadata_package':old,'legacy_reservations':[]},{})==0

@pytest.mark.parametrize('fault',['parent','baseline','maximum','schema'])
def test_donor_does_not_bypass_resource_package_identity(fault):
    _,new=packages()
    if fault=='parent':new['previous_package_sha256']='f'*64
    if fault=='baseline':new['baseline_policy_sha256']='f'*64
    if fault=='maximum':new['maximum_bytes']+=1
    if fault=='schema':new['schema']='unreviewed'
    with pytest.raises(b.StorageBudgetError):storage(DONOR).original_package(new)

@pytest.mark.parametrize('fault',[None,'dirty','commit','environment'])
def test_original_compute_source_runtime_still_rejects_mismatch(fault):
    tree=ast.parse(source(OLD,'tools/st_world_policy/spatial_probe.py'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='source_runtime')
    def check(command,**kwargs):
        if command[1]=='status':return b' M changed.py' if fault=='dirty' else b''
        return ('f'*40 if fault=='commit' else OLD)+'\n'
    expected={'base':{'cudnn':123,'torch':'fixture'}}
    actual={'base':{'torch':'wrong' if fault=='environment' else 'fixture'}}
    ns={'subprocess':types.SimpleNamespace(check_output=check),'REPO':'original compute root',
        'runtime_environment':lambda **_:actual,'re':re,
        'torch':types.SimpleNamespace(set_num_threads=lambda _:None)}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<original function>','exec'),ns)
    if fault:
        with pytest.raises(ValueError):ns['source_runtime'](OLD,expected,'cpu',configure_device=False)
    else:ns['source_runtime'](OLD,expected,'cpu',configure_device=False)
