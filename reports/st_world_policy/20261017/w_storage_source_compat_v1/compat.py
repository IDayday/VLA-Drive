"""Exact two-runtime storage-only compatibility; no policy mutation or model load."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import types

HERE=Path(__file__).resolve().parent
OLD='581399eb309677c20fe7464791b4f18685e947a6'
NEW='2a1e140a64d2817247ceb6214f7f2affa20d56c6'
ROOTS={OLD:Path('/mnt/project/VLA-Drive-st-spatial-probe-runtime-v2-20261017'),
       NEW:Path('/mnt/project/VLA-Drive-st-w-storage-only-581399eb-v1-20261017')}
FILES={'tools/st_world_policy/metadata_storage.py':'68ffeb4862c6ea0536e34e9c14f36ee189625e02ead806775cd397e179f7e5ca',
       'tools/st_world_policy/spatial_w_storage.py':'0d2c16be1e7ef98bf5b1c9ab6abc10f6b4fdd23ff722d8e0e41f0b280f957987'}
TARGET='tools.st_world_policy.spatial_w_storage'
SCHEMA='st_exact_storage_only_W_runtime_compatibility_v1'
READ_BYTES=0


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def raw(path,expected=None):
    global READ_BYTES
    path=Path(path)
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as f:
        a=os.fstat(f.fileno())
        if not stat.S_ISREG(a.st_mode) or a.st_size>2<<20:raise ValueError('Small regular compatibility source/control required')
        READ_BYTES+=a.st_size
        if READ_BYTES>64<<20:raise ValueError('Compatibility metadata budget exhausted')
        data=f.read((2<<20)+1);b=os.fstat(f.fileno())
    v=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if len(data)!=a.st_size or v(a)!=v(b) or v(b)!=v(path.stat()):raise ValueError('Compatibility file changed')
    sha=hashlib.sha256(data).hexdigest()
    if expected is not None and sha!=expected:raise ValueError('Compatibility source/control SHA mismatch')
    return data


def git(root,*args):
    global READ_BYTES
    data=subprocess.check_output(['git',*args],cwd=root,timeout=15)
    READ_BYTES+=len(data)
    if READ_BYTES>64<<20:raise ValueError('Compatibility git metadata budget exhausted')
    return data


def evidence():
    trees={};storage={}
    for commit,root in ROOTS.items():
        if git(root,'rev-parse','HEAD').decode().strip()!=commit or git(root,'status','--porcelain'):
            raise ValueError('Exact clean original runtime required')
        entries={}
        for item in git(root,'ls-tree','-rz',commit).split(b'\0'):
            if not item:continue
            header,path=item.split(b'\t',1);mode,kind,blob=header.decode().split();name=path.decode()
            if name not in FILES:entries[name]=[mode,kind,blob]
        trees[commit]=entries
        storage[commit]={name:hashlib.sha256(raw(root/name)).hexdigest() for name in FILES}
    if trees[OLD]!=trees[NEW]:raise ValueError('Computation or any non-storage tracked file differs')
    if storage[NEW]!=FILES:raise ValueError('Reviewed storage-only files differ')
    delta=git(ROOTS[NEW],'diff','--name-only',OLD,NEW).decode().splitlines()
    if sorted(delta)!=sorted(FILES):raise ValueError('Only the exact two storage files may differ')
    git(ROOTS[NEW],'merge-base','--is-ancestor',OLD,NEW)
    return {'schema':SCHEMA,'runtimes':{k:str(v) for k,v in ROOTS.items()},
        'common_compute_blob_sha256':digest(trees[OLD]),'common_tracked_entries':len(trees[OLD]),
        'excluded_files':sorted(FILES),'storage_sha256':storage,
        'adapter_sha256':hashlib.sha256(raw(HERE/'compat.py')).hexdigest(),
        'original_validator_sha256':FILES['tools/st_world_policy/spatial_w_storage.py'],
        'scope':'Only package common source comparison; original signature and same-arm endpoint identities preserved',
        'policy_schema_changed':False,'ledger_written':False}


def verify(pin):
    if set(pin)!={'path','sha256'}:raise ValueError('Exact compatibility receipt pin required')
    document=json.loads(raw(pin['path'],pin['sha256']))
    if document!=evidence():raise ValueError('Compatibility proof differs from exact clean runtimes/adapter')
    return document


def common_source(source,proof):
    if source not in (OLD,NEW) or proof.get('schema')!=SCHEMA or set(proof.get('runtimes',{}))!={OLD,NEW}:
        raise ValueError('Unapproved W extraction runtime')
    return proof['common_compute_blob_sha256']


def patched_source(source):
    old="common = (signature['checkpoint']['base_binding'], binding['source_sha'], binding['environment'])"
    new="common = (signature['checkpoint']['base_binding'], _verified_common_compute_source(binding['source_sha']), binding['environment'])"
    if source.count(old)!=1:raise ValueError('Original package common-source guard changed')
    return source.replace(old,new)


def install(pin):
    proof=verify(pin)
    if TARGET in sys.modules:raise ValueError('Install explicitly before any spatial_w_storage import')
    root=ROOTS[NEW];sys.path.insert(0,str(root))
    package=importlib.import_module('tools.st_world_policy')
    if Path(package.__file__).resolve().parent!=root/'tools/st_world_policy':raise ValueError('Foreign policy package')
    budget=importlib.import_module('tools.st_world_policy.cache_storage_budget')
    if Path(budget.__file__).resolve()!=root/'tools/st_world_policy/cache_storage_budget.py':raise ValueError('Foreign budget module')
    source=raw(root/'tools/st_world_policy/spatial_w_storage.py',FILES['tools/st_world_policy/spatial_w_storage.py']).decode()
    module=types.ModuleType(TARGET);module.__file__=str(root/'tools/st_world_policy/spatial_w_storage.py');module.__package__='tools.st_world_policy'
    module._verified_common_compute_source=lambda value:common_source(value,proof)
    exec(compile(patched_source(source),module.__file__,'exec'),module.__dict__)
    if module.b is not budget:raise ValueError('Budget module identity differs')
    sys.modules[TARGET]=module;setattr(package,'spatial_w_storage',module)
    return {'schema':'st_exact_W_source_compatibility_install_v1','receipt':pin,
            'adapter_sha256':proof['adapter_sha256'],'common_compute_blob_sha256':proof['common_compute_blob_sha256'],
            'actual_source_shas_preserved':[OLD,NEW],'original_validator_sha256':proof['original_validator_sha256'],
            'policy_or_ledger_written':False,'source_metadata_bytes':READ_BYTES}


def build_receipt(path):
    data=(json.dumps(evidence(),indent=2,sort_keys=True)+'\n').encode();path=Path(path)
    if path.exists():
        if raw(path)!=data:raise ValueError('Immutable compatibility receipt conflict')
    else:
        with path.open('xb') as f:f.write(data)
    return {'path':str(path),'sha256':hashlib.sha256(data).hexdigest()}
