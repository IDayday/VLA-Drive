import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import pytest

HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('compat',HERE/'compat.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
BASE=Path('/mnt/project/st-world-policy-20261016')
POLICY=BASE/'audits/full_future_index_preparation_v1/finalized_phase_v1/PROPOSED_POLICY.json'
POLICY_SHA='7a78476af60b191da42439dfdda0c63dae6334f7e911ee7bc3ed299e77486148'
SP1=BASE/'audits/spatial_probe_actual_SP1_5000_v1/stages/dev_plan/SIGNATURE.json'
SP1_SHA='4582329a1e78f01c7aceda8543ffef91a37f28964f5ba0e4a04afda6b315ade1'


def module(patched,proof):
    sys.path.insert(0,str(c.ROOTS[c.NEW]))
    source=c.raw(c.ROOTS[c.NEW]/'tools/st_world_policy/spatial_w_storage.py',c.FILES['tools/st_world_policy/spatial_w_storage.py']).decode()
    m=types.ModuleType('test_package_guard');m._verified_common_compute_source=lambda s:c.common_source(s,proof)
    exec(compile(c.patched_source(source) if patched else source,'<original-package-guard>','exec'),m.__dict__)
    return m


@pytest.fixture(scope='module')
def proof():return c.evidence()


@pytest.fixture
def bank(proof):
    core=json.loads(c.raw(POLICY,POLICY_SHA));sig1=json.loads(c.raw(SP1,SP1_SHA));m=module(True,proof)
    signatures={r['signature_path']:json.loads(c.raw(r['signature_path'],r['signature_sha256']))
        for r in core['claims'] if r.get('identity_kind')=='st_frozen_spatial_w_v1'}
    signatures[str(SP1)]=sig1
    claim=dict(claim_id='SP1-frozen-W-dev16-native5000-v1',root=str(BASE/'caches/learned_w_probe_SP1_5000_v1/dev'),
        cache_identity=sig1['identity'],identity_kind='st_frozen_spatial_w_v1',signature_path=str(SP1),signature_sha256=SP1_SHA,
        tensor_schema=m.tensor_schema(sig1['binding']),**m._requirements(16,144,2048))
    core['claims'].append(claim);core['frozen_w_package']['members'][claim['root']]=claim
    declarations={r['root']:r for r in core['claims']}
    def bind(mod):
        # Isolate the real package guard; original deep signature checks already
        # belong to native admission and are NOT re-run or claimed here.
        mod._pin=lambda ref:signatures[ref['path']]
        mod.validate_signature=lambda sig:(sig['binding'],mod._requirements(len(sig['rows']),sig['binding']['memory_tokens'],sig['binding']['hidden_dim']))
        return mod
    return core,declarations,signatures,bind


def test_real_small_counterexample_fixed_without_mutation(proof,bank):
    core,decl,sigs,bind=bank;before=copy.deepcopy(sigs)
    with pytest.raises(Exception,match='share original Base'):
        bind(module(False,proof)).validate_package(core,decl)
    assert bind(module(True,proof)).validate_package(core,decl)==138426288
    assert sigs==before


@pytest.mark.parametrize('fault',['third_runtime','base','environment','same_arm_source'])
def test_original_remaining_source_contracts_enforced(proof,bank,fault):
    core,decl,sigs,bind=bank
    selected=sigs[str(SP1)]
    if fault=='third_runtime':selected['binding']['source_sha']='a'*40
    if fault=='base':selected['checkpoint']['base_binding']='b'*64
    if fault=='environment':selected['binding']['environment']={'wrong':True}
    if fault=='same_arm_source':
        old=next(s for s in sigs.values() if s['binding']['arm']=='S-P0' and s['binding']['observations']['split']=='train')
        old['binding']['source_sha']=c.NEW
    with pytest.raises(Exception):bind(module(True,proof)).validate_package(core,decl)


def test_one_expression_only(proof):
    source=c.raw(c.ROOTS[c.NEW]/'tools/st_world_policy/spatial_w_storage.py').decode()
    updated=c.patched_source(source)
    assert updated.replace("_verified_common_compute_source(binding['source_sha'])","binding['source_sha']")==source
    assert "endpoint = (binding['checkpoint_identity'], binding['training_registration'], binding['source_sha']" in updated
    with pytest.raises(ValueError):c.patched_source(updated)


def test_receipt_exact_binding_and_adapter_hash(tmp_path,proof):
    p=tmp_path/'receipt.json';pin=c.build_receipt(p)
    assert c.verify(pin)==proof
    bad=copy.deepcopy(proof);bad['common_compute_blob_sha256']='0'*64;p.write_text(json.dumps(bad))
    changed={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    with pytest.raises(ValueError):c.verify(changed)
    with pytest.raises(ValueError):c.verify(pin)


def test_other_tree_changes_rejected(monkeypatch):
    original=c.git
    def different(root,*args):
        result=original(root,*args)
        if args[:2]==('ls-tree','-rz') and root==c.ROOTS[c.NEW]:
            return result+b'100644 blob '+b'a'*40+b'\tforeign_model.py\0'
        return result
    monkeypatch.setattr(c,'git',different)
    with pytest.raises(ValueError,match='non-storage'):c.evidence()


def test_install_before_payload_no_torch(proof):
    pin={'path':str(HERE/'COMPATIBILITY.json'),'sha256':hashlib.sha256((HERE/'COMPATIBILITY.json').read_bytes()).hexdigest()}
    result=c.install(pin)
    assert 'torch' not in sys.modules and result['actual_source_shas_preserved']==[c.OLD,c.NEW]
    assert sys.modules[c.TARGET]._verified_common_compute_source(c.OLD)==proof['common_compute_blob_sha256']
    with pytest.raises(ValueError):c.install(pin)


def test_wrapper_changes_only_fixed_observer_route():
    sys.modules['compat']=c
    spec=importlib.util.spec_from_file_location('compat_run',HERE/'run_once.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    original=r.original_control()
    assert original.COMPAT_OBSERVER==HERE.resolve()/'run_once.py'
    p,row,s=original.controls('SP1',supervisor=True)
    assert str(s.JOB)==row['proposed_job_root']
    with pytest.raises(ValueError):original.controls('SP4',supervisor=True)
    with pytest.raises(ValueError):s.supervise([sys.executable,'-c','print(1)'])


def test_old_proof_cannot_admit_third_source(proof):
    with pytest.raises(ValueError):c.common_source('f'*40,proof)
    assert c.common_source(c.OLD,proof)==c.common_source(c.NEW,proof)
