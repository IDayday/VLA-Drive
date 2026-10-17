"""Fixed-source, fixed-recipe adapters; no generic command dispatch or monitor rewrite."""
import hashlib
import json
import os
import stat
from pathlib import Path
from types import ModuleType

HERE=Path(__file__).resolve().parent
BINDINGS_SHA='d89aa3690d7cc445c68d8f1f84d17cf23ce38ac763ddf3a340ef0843730ac59f'

def raw(path, sha=None):
    path=Path(path)
    fd=os.open(path,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as stream:
        info=os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size>128<<10:
            raise ValueError('Small regular source/control file required')
        value=stream.read((128<<10)+1)
    if len(value)>128<<10:raise ValueError('Source/control grew beyond bound')
    if sha is not None and hashlib.sha256(value).hexdigest()!=sha:raise ValueError('Pinned source/control changed')
    return value

def bindings():return json.loads(raw(HERE/'BINDINGS.json',BINDINGS_SHA))

def load(kind):
    b=bindings(); donor=b['donor_'+kind]; source=raw(donor['path'],donor['sha256']).decode()
    if kind=='controller':
        transform=json.loads(raw(b['transform']['path'],b['transform']['sha256']))
        if transform['donor']!=donor:raise ValueError('Wrong reviewed donor')
        for edit in transform['edits']:
            if source.count(edit['old'])!=edit['count']:raise ValueError('Mechanical adaptation no longer exact')
            source=source.replace(edit['old'],edit['new'])
        if hashlib.sha256(source.encode()).hexdigest()!=transform['transformed_sha256']:
            raise ValueError('Controller transformed source changed')
    elif kind!='prepare':raise ValueError('Only original factory and fixed controller supported')
    entry=HERE/('prepare_once.py' if kind=='prepare' else 'controller.py')
    module=ModuleType('fixed_topology240_'+kind);module.__file__=str(entry)
    exec(compile(source,str(entry),'exec'),module.__dict__)
    if kind=='prepare':
        module.ROOT=Path(b['factory_root']);module.CONTROL=module.ROOT/'controls'
        module.RUNTIME=Path(b['runtime']['path']);module.COMMIT=b['runtime']['sha']
        module.HERE=HERE;module.PLAN_SHA=b['plan']['sha256']
    return module
