"""Bounded CPU source/control tests. No model/image/native tensor or live ledger."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

HERE=Path(__file__).absolute().parent
sys.path.insert(0,str(HERE))
import storage_bridge as bridge


def write(path,d):
    raw=(json.dumps(d,sort_keys=True,indent=2)+'\n').encode();path.write_bytes(raw)
    return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}


def git(path,*args):
    return subprocess.check_output(['git',*args],cwd=path,stderr=subprocess.DEVNULL)


@pytest.fixture(scope='module')
def bound(tmp_path_factory):
    root=tmp_path_factory.mktemp('storage-bridge');f=bridge.fixed(bridge.Reads());storage=root/'storage'
    subprocess.run(['git','clone','--shared','--no-checkout',f['compute']['path'],str(storage)],check=True,
                   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30)
    git(storage,'sparse-checkout','init','--no-cone')
    subprocess.run(['git','sparse-checkout','set','--no-cone','--stdin'],cwd=storage,check=True,
                   input=('\n'.join('/'+p for p in f['modules'])+'\n').encode(),stdout=subprocess.DEVNULL)
    git(storage,'checkout','--detach',f['compute']['source_sha'])
    for p in f['modules']:(storage/p).write_bytes(git(storage,'show',f['donor_commit']+':'+p))
    git(storage,'add',*f['modules']);git(storage,'-c','user.name=CPU fixture','-c','user.email=fixture@invalid',
        'commit','-m','Temporary exact two storage files')
    commit=git(storage,'rev-parse','HEAD').decode().strip()
    signature=bridge.Reads().document(f['signature'])
    binding=write(root/'binding.json',signature['binding'])
    prep={'runtime':{'path':f['compute']['path'],'source_sha':f['compute']['source_sha']},
          'caches':{'train':{'root':f['train_claim']['root'],'identity':f['cache_identity']}},
          'controls':{'train':{'binding':binding}}}
    prep_pin=write(root/'preparation.json',prep)
    policy_path=HERE.parent/'w_train_extract_preparation_v1/PROPOSED_POLICY.json'
    policy={'path':str(policy_path),'sha256':bridge.sha(policy_path.read_bytes())}
    doc={'schema':bridge.SCHEMA,'stage':'train_extract','compute_runtime':prep['runtime'],
         'storage_runtime':{'path':str(storage),'source_sha':commit,'base_commit':f['compute']['source_sha'],
                            'donor_commit':f['donor_commit']},
         'signature':f['signature'],'cache_identity':f['cache_identity'],'policy':policy,
         'helpers':{n:bridge.sha((HERE/n).read_bytes()) for n in bridge.HELPERS}}
    pin=write(root/'bridge.json',doc)
    admission={'stage':'train_extract','preparation':prep_pin,'runtime':prep['runtime'],
               'storage_policy':policy,'storage_compatibility':pin}
    admission_pin=write(root/'admission.json',admission)
    return SimpleNamespace(**locals())


def child(bound,code=None,*,admission_pin=None):
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':bound.f['compute']['path']}
    pin=admission_pin or bound.admission_pin
    argv=([sys.executable,str(HERE/'storage_bridge.py'),'--admission',pin['path'],'--admission-sha256',pin['sha256'],'--probe-install']
          if code is None else [sys.executable,'-c',code])
    return subprocess.run(argv,cwd=bound.f['compute']['path'],env=env,text=True,
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180)


def setup(bound):
    return f"""
import sys,json
from pathlib import Path
sys.path.insert(0,{str(HERE)!r})
import storage_bridge as b
ad=b.Reads().document({bound.admission_pin!r});p=b.Reads().document(ad['preparation'])
sys.path.insert(0,p['runtime']['path'])
"""


def test_actual_child_cli_exact_two_storage_modules_original_budget_no_torch(bound):
    result=child(bound);assert result.returncode==0,result.stderr
    d=json.loads(result.stdout);assert d['torch_imported'] is False
    assert d['compute_runtime']['source_sha']==bound.f['compute']['source_sha']
    assert d['signature']==bound.f['signature'] and d['cache_identity']==bound.f['cache_identity']
    assert d['metadata_bytes_read']<128<<20 and d['ledger_migration_performed'] is False
    for name in bridge.MODULES:assert d['canonical_modules'][name].startswith(str(bound.storage)+'/')
    assert d['canonical_modules']['tools.st_world_policy.cache_storage_budget'].startswith(bound.f['compute']['path']+'/')
    assert not (bound.root/'CLAIMS.json').exists()


@pytest.mark.parametrize('fault',['stage','source','compute_path','donor','base','commit','storage_path',
    'signature_sha','signature_path','cache','policy_pin','helper_sha','helper_inventory','schema'])
def test_wrong_pins_reject_before_import_or_reservation(bound,tmp_path,fault):
    d=copy.deepcopy(bound.doc);stage='train_extract'
    if fault=='stage':stage='dev_extract'
    if fault=='source':d['compute_runtime']['source_sha']='f'*40
    if fault=='compute_path':d['compute_runtime']['path']=str(tmp_path)
    if fault=='donor':d['storage_runtime']['donor_commit']='f'*40
    if fault=='base':d['storage_runtime']['base_commit']='f'*40
    if fault=='commit':d['storage_runtime']['source_sha']='f'*40
    if fault=='storage_path':d['storage_runtime']['path']=str(tmp_path)
    if fault=='signature_sha':d['signature']['sha256']='f'*64
    if fault=='signature_path':d['signature']['path']=str(tmp_path/'signature.json')
    if fault=='cache':d['cache_identity']='f'*64
    if fault=='policy_pin':d['policy']['sha256']='f'*64
    if fault=='helper_sha':d['helpers']['storage_bridge.py']='f'*64
    if fault=='helper_inventory':d['helpers']['extra.py']='f'*64
    if fault=='schema':d['schema']='unreviewed'
    pin=write(tmp_path/'bridge.json',d)
    with pytest.raises((ValueError,FileNotFoundError,subprocess.CalledProcessError)):
        bridge.validate(pin,bound.prep,stage,bound.policy)
    assert 'torch' not in sys.modules


@pytest.mark.parametrize('fault',['root','binding','claim','metadata_baseline','metadata_cap','W_cap','ledger'])
def test_wrong_role_policy_and_preparation_fail_closed(bound,tmp_path,fault):
    prep=copy.deepcopy(bound.prep);d=copy.deepcopy(bound.doc)
    if fault=='root':prep['caches']['train']['root']=str(tmp_path/'cache')
    elif fault=='binding':
        binding=copy.deepcopy(bound.signature['binding']);binding['observations']['split']='dev'
        prep['controls']['train']['binding']=write(tmp_path/'binding.json',binding)
    else:
        policy=bridge.Reads().document(bound.policy)
        if fault=='claim':policy['claims']=[r for r in policy['claims'] if r['root']!=bound.f['train_claim']['root']]
        if fault=='metadata_baseline':policy['metadata_package']['previous_package_sha256']='f'*64
        if fault=='metadata_cap':policy['metadata_package']['maximum_bytes']+=1
        if fault=='W_cap':policy['frozen_w_package']['maximum_bytes']+=1
        if fault=='ledger':policy['state_dir']=str(tmp_path/'ledger')
        d['policy']=write(tmp_path/'policy.json',policy)
    pin=write(tmp_path/'bridge.json',d)
    with pytest.raises(ValueError):bridge.validate(pin,prep,'train_extract',d['policy'])


@pytest.mark.parametrize('fault',['untracked','dirty_source','tracked_extra'])
def test_storage_checkout_cannot_be_dirty_or_include_extra_files(bound,tmp_path,fault):
    # Private sparse repo only. Copy git state to avoid mutating other test cases.
    storage=tmp_path/'storage'
    subprocess.run(['git','clone','--shared','--no-checkout',str(bound.storage),str(storage)],check=True,
                   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    git(storage,'sparse-checkout','init','--no-cone')
    subprocess.run(['git','sparse-checkout','set','--no-cone','--stdin'],cwd=storage,check=True,
        input=('\n'.join('/'+p for p in bound.f['modules'])+'\n').encode(),stdout=subprocess.DEVNULL)
    git(storage,'checkout','--detach',bound.commit)
    if fault=='dirty_source':
        path=storage/next(iter(bound.f['modules']));path.write_bytes(path.read_bytes()+b'\n# modified\n')
    else:(storage/'extra.txt').write_text('unreviewed')
    d=copy.deepcopy(bound.doc);d['storage_runtime']['path']=str(storage)
    if fault=='tracked_extra':
        git(storage,'add','--sparse','extra.txt');git(storage,'-c','user.name=fixture','-c','user.email=f@invalid','commit','-m','extra')
        d['storage_runtime']['source_sha']=git(storage,'rev-parse','HEAD').decode().strip()
    pin=write(tmp_path/'bridge.json',d)
    with pytest.raises(ValueError):bridge.validate(pin,bound.prep,'train_extract',bound.policy)


@pytest.mark.parametrize('fault',['old_module','stale_attribute','foreign_package','torch','stale_tools_parent'])
def test_child_rejects_preloaded_mixed_modules_before_install(bound,fault):
    actions={
      'old_module':'import tools.st_world_policy.metadata_storage',
      'stale_attribute':'import tools.st_world_policy as parent; parent.metadata_storage=object()',
      'foreign_package':"import tools.st_world_policy as parent; parent.__path__=['/foreign']",
      'torch':"import types; sys.modules['torch']=types.ModuleType('torch')",
      'stale_tools_parent':'import tools.st_world_policy; import tools; tools.st_world_policy=object()'}
    code=setup(bound)+actions[fault]+"\ntry:\n b.install(ad['storage_compatibility'],p,ad['stage'],ad['storage_policy'])\nexcept ValueError as e:\n print(str(e))\nelse:\n raise AssertionError('mixed import accepted')\n"
    r=child(bound,code);assert r.returncode==0,r.stderr


def test_executes_verified_bytes_not_a_second_loader_read(bound):
    code=setup(bound)+"""
checked=b.validate(ad['storage_compatibility'],p,ad['stage'],ad['storage_policy'])
b.validate=lambda *a,**kw:checked
import importlib.machinery
original_get_code=importlib.machinery.SourceFileLoader.get_code
def exact_get_code(loader,*args):
 if loader.name in b.MODULES: raise AssertionError('unchecked loader reopen')
 return original_get_code(loader,*args)
importlib.machinery.SourceFileLoader.get_code=exact_get_code
# Ensure only packages/budget use ordinary imports, before denying all loader get_code.
"""
    # Package/budget import must precede interception, storage modules remain absent.
    code=code.replace("import importlib.machinery\n", "b._packages(p['runtime']['path'])\nimport tools.st_world_policy.cache_storage_budget\nimport importlib.machinery\n")
    code+="b.install(ad['storage_compatibility'],p,ad['stage'],ad['storage_policy']); print('EXACT_BYTES')\n"
    r=child(bound,code);assert r.returncode==0,r.stderr;assert 'EXACT_BYTES' in r.stdout


def test_no_pin_is_noop():
    before=dict(sys.modules)
    assert bridge.install(None,{},'dev_plan',None) is None
    assert dict(sys.modules)==before


def test_real_read_replacement_is_rejected(tmp_path,monkeypatch):
    path=tmp_path/'x';path.write_bytes(b'original');real=os.fstat;calls=0
    def race(fd):
        nonlocal calls
        calls+=1
        if calls==2:
            replacement=tmp_path/'replacement';replacement.write_bytes(b'changed');os.replace(replacement,path)
        return real(fd)
    monkeypatch.setattr(bridge.os,'fstat',race)
    with pytest.raises(ValueError,match='changed'):bridge.Reads().raw(path)


def test_bounded_source_read_rejects_oversize_before_read(tmp_path):
    p=tmp_path/'large'
    with p.open('wb') as out:out.truncate(bridge.FILE_LIMIT+1)
    with pytest.raises(ValueError,match='bounded'):bridge.Reads().raw(p)


def test_observer_copy_is_only_documented_mechanical_seam():
    original=(HERE.parent/'spatial_probe_actual_preparation_v1/observed_cli.py').read_text()
    new=(HERE/'observed_cli.py').read_text()
    new=new.replace("from storage_bridge import upstream_module, install\n_control = upstream_module('prepare_bindings')\nread, write_once, write_control = _control.read, _control.write_once, _control.write_control",'from prepare_bindings import read, write_once, write_control')
    new=new.replace('def run(preparation, stage, attempt, policy, commitments, read_cap, *, compatibility=None, admission=None):','def run(preparation, stage, attempt, policy, commitments, read_cap):')
    new=new.replace("    provenance = install(compatibility, preparation, stage, policy, admission=admission)\n    if provenance is not None:\n        write_once(Path(attempt) / 'STORAGE_COMPATIBILITY.json', provenance)\n",'')
    new=new.replace("admission.get('checkpoint_commitments'), a.read_cap,\n               compatibility=admission.get('storage_compatibility'), admission=admission)","admission.get('checkpoint_commitments'), a.read_cap)")
    assert new==original


def test_changed_compute_checkout_still_rejects_before_install(bound,monkeypatch):
    original=bridge.Reads.git
    def dirty(self,root,*args):
        if root==bound.prep['runtime']['path'] and args[0]=='status':return b' M compute.py'
        return original(self,root,*args)
    monkeypatch.setattr(bridge.Reads,'git',dirty)
    with pytest.raises(ValueError,match='clean checkout'):
        bridge.validate(bound.pin,bound.prep,'train_extract',bound.policy)


def test_supervisor_wrapper_preserves_validation_and_only_redirects_exact_call(monkeypatch):
    import supervise_bridge as s
    calls=[]
    original=SimpleNamespace(PYTHON='/python',validate=lambda *a:calls.append(('validated',a)) or (960,64<<30),
        supervise=lambda argv,**kw:calls.append(('supervise',argv,kw)) or 'result')
    monkeypatch.setattr(s,'upstream_module',lambda name: original if name=='supervise_once' else object())
    monkeypatch.setattr(s,'fixed',lambda _: {'upstream':{'observed_cli.py':{'path':'/original/observed_cli.py'}}})
    monkeypatch.setattr(s,'validate',lambda *a,**kw:calls.append(('bridge',a)))
    parent=s.build_supervisor();argv=['/python','/original/observed_cli.py','--admission','/ad',
        '--admission-sha256','a'*64,'--attempt','/attempt','--read-cap','1024']
    with pytest.raises(ValueError,match='preflight'):parent.supervise(argv,cwd='/original')
    admission={'stage':'train_extract','storage_policy':{'path':'/policy','sha256':'a'*64},
               'storage_compatibility':{'path':'/bridge','sha256':'b'*64}}
    assert parent.validate(admission,{},'train_extract',1)==(960,64<<30)
    assert parent.supervise(argv,cwd='/original',popen='original explicit fixture')=='result'
    assert calls[-1][1][1]==str(HERE/'observed_cli.py')
    assert calls[-1][2]['popen']=='original explicit fixture'
    assert argv[1]=='/original/observed_cli.py'
    with pytest.raises(ValueError,match='Unexpected'):parent.supervise([*argv,'extra'])
    parent.validate({'stage':'dev_plan'},{},'dev_plan',1)
    parent.supervise(argv,cwd='/original')
    assert calls[-1][1]==argv  # absent compatibility pin retains original child


def test_installed_bridge_keeps_real_tiny_ledger_resource_checks(bound,tmp_path):
    # Execute the actual unchanged budget and actual donor resource validators.
    # Existing tests use private CPU controls only and deliberately absent tensors.
    code=setup(bound)+f"""
b.install(ad['storage_compatibility'],p,ad['stage'],ad['storage_policy'])
sys.path.insert(0,str(Path(p['runtime']['path'])/'tests/st_world'))
import test_spatial_w_storage as tests
import pytest
base=Path({str(tmp_path)!r})
tests.test_migration_is_typed_separate_and_old_claims_history_preserved(base/'success')
tests.test_declared_w_aggregate_exact_boundary_and_one_byte_over(base/'cap')
tests.test_single_campaign_cap_counts_all_declarations_exactly_once(base/'campaign')
for name in ['no_package','wrong_kind','shape','amount','overlap','metadata_baseline','w_baseline']:
 tests.test_wrong_admission_rejected_without_ledger_mutation(base/name,name)
for name in ['checkpoint','free']:
 with pytest.MonkeyPatch.context() as patch:
  tests.test_w_migration_preserves_resource_fail_closed_checks(base/name,patch,name)
assert 'torch' not in sys.modules
print('TINY_LEDGER_12_CHECKS_PASS')
"""
    result=child(bound,code);assert result.returncode==0,result.stderr
    assert 'TINY_LEDGER_12_CHECKS_PASS' in result.stdout


def test_real_versioned_observer_and_supervisor_help_are_torch_free(bound):
    for name in ['observed_cli.py','supervise_bridge.py']:
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1',
             'PYTHONPATH':bound.f['compute']['path']}
        r=subprocess.run([sys.executable,str(HERE/name),'--help'],cwd=bound.f['compute']['path'],
                         env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
        assert r.returncode==0,r.stderr
        assert '--admission-sha256' in r.stdout


def test_real_child_wrong_pin_fails_without_torch(bound):
    code=setup(bound)+"""
ad['storage_compatibility']['sha256']='f'*64
try:b.install(ad['storage_compatibility'],p,ad['stage'],ad['storage_policy'])
except ValueError:assert 'torch' not in sys.modules;print('REJECTED_BEFORE_TORCH')
else:raise AssertionError('wrong pin accepted')
"""
    r=child(bound,code);assert r.returncode==0,r.stderr;assert 'REJECTED_BEFORE_TORCH' in r.stdout


def test_real_temporary_v2_package_reserve_write_retains_W_baseline(bound,tmp_path):
    code=setup(bound)+f"""
b.install(ad['storage_compatibility'],p,ad['stage'],ad['storage_policy'])
sys.path.insert(0,str(Path(p['runtime']['path'])/'tests/st_world'))
import test_spatial_w_storage as t
import copy,hashlib,time
base=Path({str(tmp_path)!r});f=t.fixture(base)
t.m.migrate_policy(**f.kwargs)
old=t.b.StorageBudget(f.new_path);before=json.loads(old.state_path.read_text())
new=copy.deepcopy(f.new_doc);package=new['metadata_package']
package.update(previous_package_sha256=t.b._digest(package),schema=t.m.FULL_METADATA_PACKAGE_SCHEMA,maximum_bytes=8*t.b.GIB)
new_path=base/'expanded.json';new_sha=t.write(new_path,new)
q=copy.deepcopy(f.q);q['bindings']=dict(old_policy_sha256=hashlib.sha256(f.new_path.read_bytes()).hexdigest(),
 new_policy_sha256=new_sha,old_ledger_sha256=hashlib.sha256(old.state_path.read_bytes()).hexdigest(),
 pilot_inventory_sha256=f.inventory_sha,checkpoint_sha256=f.m0.checkpoint_sha)
q['checked_writer_roots']=sorted(set(old.declarations)|{{r['root'] for r in old.core['legacy_reservations']}})
q_pin=t.write(f.q_path,q)
t.m.migrate_policy(old_policy=f.new_path,old_policy_sha256=q['bindings']['old_policy_sha256'],
 new_policy=new_path,new_policy_sha256=new_sha,expected_old_ledger_sha256=q['bindings']['old_ledger_sha256'],
 quiescence_receipt=f.q_path,quiescence_sha256=q_pin,pilot_inventory=f.inventory_path,pilot_inventory_sha256=f.inventory_sha,
 checkpoint_receipt=f.m0.checkpoint_path,checkpoint_sha256=f.m0.checkpoint_sha)
budget=t.b.StorageBudget(new_path)
assert budget.core['frozen_w_package']==before['core']['frozen_w_package']
assert t.m.original_package(budget.core['metadata_package'])==before['core']['metadata_package']
for sig in f.signatures:
 handle=budget.reserve(sig.root,sig.signature['identity'],**{{k:sig.claim[k] for k in t.w.AMOUNTS}})
 with handle.writing(16384):
  sig.root.mkdir(parents=True);t.write(sig.root/'identity.json',sig.signature)
assert 'torch' not in sys.modules
print('TEMPORARY_V2_RESERVE_WRITE_PASS')
"""
    r=child(bound,code);assert r.returncode==0,r.stderr
    assert 'TEMPORARY_V2_RESERVE_WRITE_PASS' in r.stdout


def test_partial_install_poisoned_no_old_fallback(bound):
    code=setup(bound)+"""
checked=b.validate(ad['storage_compatibility'],p,ad['stage'],ad['storage_policy'])
checked['bytes']['tools/st_world_policy/spatial_w_storage.py']=b"raise RuntimeError('injected partial import')"
b.validate=lambda *a,**kw:checked
try:b.install(ad['storage_compatibility'],p,ad['stage'],ad['storage_policy'])
except RuntimeError:pass
else:raise AssertionError('injected import did not fail')
try:b.install(ad['storage_compatibility'],p,ad['stage'],ad['storage_policy'])
except ValueError:print('POISONED_NO_FALLBACK')
else:raise AssertionError('partial import reused')
assert 'torch' not in sys.modules
"""
    r=child(bound,code);assert r.returncode==0,r.stderr;assert 'POISONED_NO_FALLBACK' in r.stdout


@pytest.mark.parametrize('kind',['symlink','hardlink'])
def test_bounded_control_refuses_aliases(tmp_path,kind):
    source=tmp_path/'source';source.write_bytes(b'{}');alias=tmp_path/'alias'
    if kind=='symlink':alias.symlink_to(source)
    else:os.link(source,alias)
    with pytest.raises(ValueError):bridge.Reads().raw(alias)
