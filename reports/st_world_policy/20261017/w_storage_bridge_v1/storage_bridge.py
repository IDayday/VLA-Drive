"""One process-local SP0 train_extract storage bridge; no dispatch or migration.

Only two fixed, independently reviewed storage modules may be installed. All
compute modules still resolve from the original clean 581399eb checkout.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import types

HERE = Path(__file__).absolute().parent
FIXED_SHA = 'c4305f8d459694b2e89e546638c10e976074a2301b967eea32b4edc4f4bcb61f'
SCHEMA = 'st_SP0_train_W_storage_bridge_v1'
MODULES = ('tools.st_world_policy.metadata_storage', 'tools.st_world_policy.spatial_w_storage')
HELPERS = ('storage_bridge.py', 'observed_cli.py', 'supervise_bridge.py', 'FIXED_INPUTS.json')
READ_LIMIT = 128 << 20
FILE_LIMIT = 4 << 20
_INSTALLED = False


def sha(raw): return hashlib.sha256(raw).hexdigest()
def digest(value): return sha(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
def version(st): return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
def proc_rchar():
    with open('/proc/self/io', 'r') as stream:
        return int(next(line.split(':')[1] for line in stream if line.startswith('rchar:')))



def canonical(path):
    p = Path(path)
    if not p.is_absolute() or '..' in p.parts or str(p) != str(p.resolve()):
        raise ValueError('Noncanonical bridge path')
    for parent in p.parents:
        if not stat.S_ISDIR(parent.lstat().st_mode): raise ValueError('Unsafe bridge ancestor')
    return p


class Reads:
    def __init__(self): self.bytes = 0; self.start_rchar = proc_rchar()
    def check_io(self):
        if proc_rchar() - self.start_rchar > READ_LIMIT:
            raise ValueError('Bridge observed metadata read budget exceeded')
    def charge(self, size):
        self.bytes += size
        if self.bytes > READ_LIMIT: raise ValueError('Bridge metadata read budget exceeded')
        self.check_io()
    def raw(self, path, expected=None):
        p = canonical(path)
        fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as stream:
            before = os.fstat(stream.fileno())
            if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > FILE_LIMIT):
                raise ValueError('Bridge input must be bounded private regular metadata')
            self.charge(before.st_size + 1)
            raw = stream.read(FILE_LIMIT + 1); after = os.fstat(stream.fileno())
        if len(raw) > FILE_LIMIT or version(before) != version(after) or version(after) != version(p.stat()):
            raise ValueError('Bridge source/control changed while reading')
        self.check_io()
        if expected is not None and sha(raw) != expected: raise ValueError('Bridge SHA mismatch: ' + str(p))
        return raw
    def document(self, pin):
        if not isinstance(pin, dict) or set(pin) != {'path','sha256'} or not re.fullmatch('[0-9a-f]{64}', pin['sha256']):
            raise ValueError('Exact bridge pin required')
        def pairs(items):
            result = {}
            for key,value in items:
                if key in result: raise ValueError('Duplicate bridge JSON field')
                result[key] = value
            return result
        return json.loads(self.raw(pin['path'], pin['sha256']), object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite bridge control')))
    def git(self, root, *args):
        result = subprocess.run(['git', *args], cwd=root, check=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=15)
        self.charge(len(result.stdout)+len(result.stderr))
        return result.stdout


def fixed(reads): return reads.document({'path':str(HERE/'FIXED_INPUTS.json'),'sha256':FIXED_SHA})


def _clean_tree(reads, root, commit):
    canonical(root)
    if (not re.fullmatch('[0-9a-f]{40}', commit) or
            reads.git(root,'rev-parse','HEAD').decode().strip()!=commit or
            reads.git(root,'status','--porcelain','--untracked-files=all')):
        raise ValueError('Bridge requires exact committed clean checkout')


def validate(pin, preparation, stage, policy, *, admission=None):
    """Pure bounded preflight; no module install, torch, budget, locks or writes."""
    reads=Reads(); f=fixed(reads)
    if pin is None: return None
    d=reads.document(pin)
    fields={'schema','stage','compute_runtime','storage_runtime','signature','cache_identity','policy','helpers'}
    if set(d)!=fields or d['schema']!=SCHEMA or d['stage']!='train_extract' or stage!='train_extract':
        raise ValueError('Bridge is exclusively the fixed train_extract role')
    compute={'path':f['compute']['path'],'source_sha':f['compute']['source_sha']}
    if d['compute_runtime']!=compute or preparation.get('runtime')!=compute:
        raise ValueError('Original compute path/source changed')
    if (d['signature']!=f['signature'] or d['cache_identity']!=f['cache_identity'] or
            d['policy']!=policy or set(d['helpers'])!=set(HELPERS)):
        raise ValueError('Bridge signature/cache/policy/helper binding changed')
    for name in HELPERS:
        reads.raw(HERE/name,d['helpers'][name])
    if admission is not None:
        if (admission.get('storage_compatibility')!=pin or admission.get('runtime')!=compute or
                admission.get('storage_policy')!=policy or admission.get('stage')!=stage):
            raise ValueError('Supervisor and child bridge admission differ')
    _clean_tree(reads,compute['path'],compute['source_sha'])
    storage=d['storage_runtime']
    if (not isinstance(storage,dict) or set(storage)!={'path','source_sha','base_commit','donor_commit'} or
            storage['base_commit']!=compute['source_sha'] or storage['donor_commit']!=f['donor_commit'] or
            storage['path']==compute['path']):
        raise ValueError('Storage tree must be a distinct exact two-file derivative')
    _clean_tree(reads,storage['path'],storage['source_sha'])
    # Require a real descendant of the frozen source and an exact whole-tree delta.
    reads.git(storage['path'],'merge-base','--is-ancestor',compute['source_sha'],storage['source_sha'])
    delta=reads.git(storage['path'],'diff','--name-only',compute['source_sha'],storage['source_sha']).decode().splitlines()
    if sorted(delta)!=sorted(f['modules']): raise ValueError('Storage checkout changes non-storage files')
    module_bytes={}
    for relative,expected in f['modules'].items():
        raw=reads.raw(Path(storage['path'])/relative,expected)
        if raw!=reads.git(storage['path'],'show',storage['source_sha']+':'+relative):
            raise ValueError('Storage working bytes differ from commit')
        if raw!=reads.git(storage['path'],'show',f['donor_commit']+':'+relative):
            raise ValueError('Storage bytes differ from reviewed donor')
        module_bytes[relative]=raw
    for relative,expected in f['dependencies'].items():
        raw=reads.raw(Path(compute['path'])/relative,expected)
        if (raw!=reads.git(compute['path'],'show',compute['source_sha']+':'+relative) or
                raw!=reads.git(storage['path'],'show',storage['source_sha']+':'+relative)):
            raise ValueError('Unchanged computation dependency differs')
    signature=reads.document(d['signature'])
    if (signature.get('identity')!=d['cache_identity'] or
            digest({k:v for k,v in signature.items() if k!='identity'})!=d['cache_identity'] or
            signature['binding']['source_sha']!=compute['source_sha'] or
            signature['binding']['arm']!='S-P0' or signature['binding']['observations']['split']!='train' or
            len(signature['rows'])!=64 or signature['checkpoint']['sha256']!=f['checkpoint_identity']):
        raise ValueError('Actual immutable train signature differs')
    cache=preparation['caches']['train']
    if cache['identity']!=d['cache_identity'] or cache['root']!=f['train_claim']['root']:
        raise ValueError('Preparation cache identity/root changed')
    binding=reads.document(preparation['controls']['train']['binding'])
    if binding!=signature['binding']: raise ValueError('Original extraction binding differs from actual plan')
    core=reads.document(policy)
    package=core.get('metadata_package',{}); w=core.get('frozen_w_package',{})
    if (core.get('state_dir')!=f['ledger'] or core.get('cache_cap_bytes')!=768<<30 or core.get('free_reserve_bytes')!=1024<<30 or
            package.get('schema')!='st_metadata_storage_package_v2' or package.get('maximum_bytes')!=8<<30 or
            w.get('schema')!='st_frozen_spatial_w_storage_package_v1' or w.get('maximum_bytes')!=1<<30):
        raise ValueError('Bridge requires the exact metadata-v2 / typed-W resource envelope')
    original={k:v for k,v in package.items() if k!='previous_package_sha256'}
    original.update(schema='st_metadata_storage_package_v1', maximum_bytes=4<<30)
    if (digest(original)!=package.get('previous_package_sha256') or
            digest(original)!=w.get('metadata_package_sha256')):
        raise ValueError('Original metadata baseline was not preserved')
    matches=[r for r in core.get('claims',[]) if r.get('root')==cache['root']]
    if (len(matches)!=1 or matches[0]!=f['train_claim'] or matches[0].get('identity_kind')!='st_frozen_spatial_w_v1' or
            matches[0].get('cache_identity')!=d['cache_identity'] or
            matches[0].get('signature_path')!=d['signature']['path'] or
            matches[0].get('signature_sha256')!=d['signature']['sha256'] or
            w.get('members',{}).get(cache['root'])!=matches[0]):
        raise ValueError('Actual train member is absent or bound to another signature')
    return dict(document=d, fixed=f, bytes=module_bytes, metadata_bytes=reads.bytes,
                provenance=dict(schema='st_W_storage_bridge_provenance_v1',compatibility=pin,
                    compute_runtime=compute,storage_runtime=storage,donor_modules=f['modules'],
                    unchanged_dependencies=f['dependencies'],signature=d['signature'],
                    policy=policy,cache_identity=d['cache_identity'],metadata_bytes_read=reads.bytes,
                    proc_rchar_delta=proc_rchar()-reads.start_rchar,
                    mathematical_source_unchanged=True,ledger_migration_performed=False))


def _packages(compute):
    for name,relative in [('tools','tools'),('tools.st_world_policy','tools/st_world_policy')]:
        module=importlib.import_module(name)
        expected=Path(compute)/relative
        if set(map(lambda p: str(Path(p).resolve()),module.__path__))!={str(expected)}:
            raise ValueError('Foreign or mixed canonical package: '+name)
        if name.endswith('st_world_policy') and Path(module.__file__).resolve()!=expected/'__init__.py':
            raise ValueError('Foreign canonical package file')
    if getattr(sys.modules['tools'], 'st_world_policy', None) is not sys.modules['tools.st_world_policy']:
        raise ValueError('Stale canonical parent package attribute')
    return sys.modules['tools.st_world_policy']


def install(pin, preparation, stage, policy, *, admission=None):
    """Call only at the single-threaded seam before torch/compute import."""
    global _INSTALLED
    if pin is None: return None
    if _INSTALLED or 'torch' in sys.modules or 'tools.st_world_policy.spatial_probe' in sys.modules:
        raise ValueError('Bridge must be installed once before torch/compute import')
    checked=validate(pin,preparation,stage,policy,admission=admission)
    parent=_packages(checked['document']['compute_runtime']['path'])
    for name in MODULES:
        if name in sys.modules or name.rsplit('.',1)[1] in vars(parent):
            raise ValueError('Preloaded/stale storage module is forbidden')
    budget=importlib.import_module('tools.st_world_policy.cache_storage_budget')
    expected=Path(checked['document']['compute_runtime']['path'])/'tools/st_world_policy/cache_storage_budget.py'
    if Path(budget.__file__).resolve()!=expected or getattr(parent,'cache_storage_budget') is not budget:
        raise ValueError('Unchanged budget module has foreign provenance')
    _INSTALLED=True  # Any failure below poisons this child; never fall back to mixed modules.
    for name in MODULES:
        relative=name.replace('.','/')+'.py'; path=Path(checked['document']['storage_runtime']['path'])/relative
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module; setattr(parent,name.rsplit('.',1)[1],module)
        # Execute the SAME bytes verified above, not a loader which reopens the path.
        exec(compile(checked['bytes'][relative],str(path),'exec'),module.__dict__)
        if module.b is not budget: raise ValueError('Storage modules disagree on original budget class')
    return checked['provenance']


def upstream_module(name):
    """Load a fixed original supervisor/control helper without changing its __file__."""
    reads=Reads(); f=fixed(reads); ref=f['upstream'][name+'.py']
    raw=reads.raw(ref['path'],ref['sha256'])
    module=types.ModuleType(name);module.__file__=ref['path'];module.__package__=''
    if name in sys.modules: raise ValueError('Original helper already imported')
    sys.modules[name]=module;exec(compile(raw,ref['path'],'exec'),module.__dict__)
    return module


def main():
    p=argparse.ArgumentParser(description='Read-only exact SP0 storage preflight/install probe; never extracts')
    p.add_argument('--admission',required=True);p.add_argument('--admission-sha256',required=True)
    p.add_argument('--probe-install',action='store_true')
    a=p.parse_args();r=Reads();ad=r.document({'path':a.admission,'sha256':a.admission_sha256})
    prep=r.document(ad['preparation']);pin=ad.get('storage_compatibility')
    if pin is None: raise ValueError('This standalone preflight requires the explicit bridge pin')
    if a.probe_install:
        sys.path.insert(0,prep['runtime']['path'])
        result=install(pin,prep,ad['stage'],ad['storage_policy'],admission=ad)
        result['canonical_modules']={n:sys.modules[n].__file__ for n in (*MODULES,'tools.st_world_policy.cache_storage_budget')}
    else: result=validate(pin,prep,ad['stage'],ad['storage_policy'],admission=ad)['provenance']
    result['preflight_only']=True;result['torch_imported']='torch' in sys.modules
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__': main()
