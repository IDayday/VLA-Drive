"""CPU-only exact nested PLAN layout; no live registration or writer."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import pytest
from tools.st_world_policy import metadata_storage as m
from tools.st_world_policy import cache_storage_budget as b
from tests.st_world.test_metadata_storage_guard import fixture,update_registration

CONTROL_NAMES=('ADMITTED.json','CHILD.json','CHILD.log','CLI_RESULT.json',
               'CONTROL_WRITE.lock','COUNTERS.json','RESULT.json')


def prepared(tmp_path,count=9):
    f=fixture(tmp_path,nested=True,initial_plan=True)
    f.reg['directory_patterns']=['control','log_shards']
    names=['PLAN.lock',*('control/'+x for x in CONTROL_NAMES)]
    names.extend('control/extra_'+str(i)+'.json' for i in range(count-9))
    for name in names:
        p=f.root/name;p.parent.mkdir(exist_ok=True)
        p.write_bytes(b'' if name.endswith('.lock') else b'fixture record\n')
        f.reg['file_patterns'].append(re.escape(name))
        f.reg['preexisting_pins'].append({'relative_path':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    return f


@pytest.mark.parametrize('count',[9,16])
def test_exact_nested_plan_and_empty_locks_preserved(tmp_path,count):
    f=prepared(tmp_path,count);before={p['relative_path']:(f.root/p['relative_path']).read_bytes() for p in f.reg['preexisting_pins']}
    update_registration(f);m.migrate_policy(**f.kwargs)
    with m.metadata_writing(**f.guard) as ctx:
        ctx.check(before_commit=True)
    assert all((f.root/k).read_bytes()==v for k,v in before.items())
    assert len(before)==count and before['PLAN.lock']==before['control/CONTROL_WRITE.lock']==b''
    state=json.loads(f.old.state_path.read_text())
    assert state['claims'][str(f.root)]['charged_bytes']==3*(1<<20)
    with m.metadata_writing(**f.guard):pass  # exact resume without deleting controls


@pytest.mark.parametrize('failure',['seventeen','bad_hash','omitted_lock','duplicate','escape',
                                     'symlink','hardlink','unregistered_directory','cap'])
def test_realistic_layout_still_rejects_invalid_preexisting_files(tmp_path,failure):
    f=prepared(tmp_path,17 if failure=='seventeen' else 9)
    if failure=='bad_hash':f.reg['preexisting_pins'][-1]['sha256']='0'*64
    if failure=='omitted_lock':f.reg['preexisting_pins']=[p for p in f.reg['preexisting_pins'] if p['relative_path']!='control/CONTROL_WRITE.lock']
    if failure=='duplicate':f.reg['preexisting_pins'].append(dict(f.reg['preexisting_pins'][-1]))
    if failure=='escape':f.reg['preexisting_pins'][-1]['relative_path']='../outside.json'
    if failure in ('symlink','hardlink'):
        p=f.root/'control/CHILD.log';outside=tmp_path/'external';outside.write_bytes(p.read_bytes());p.unlink()
        if failure=='symlink':p.symlink_to(outside)
        else:os.link(outside,p)
    if failure=='unregistered_directory':f.reg['directory_patterns']=[]
    if failure=='cap':
        p=f.root/'control/CHILD.log';p.write_bytes(b'X'*(4<<20))
        for pin in f.reg['preexisting_pins']:
            if pin['relative_path']=='control/CHILD.log':pin['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    update_registration(f);m.migrate_policy(**f.kwargs)
    before=f.old.state_path.read_bytes()
    with pytest.raises(b.StorageBudgetError):
        with m.metadata_writing(**f.guard):pytest.fail('invalid layout entered write context')
    assert not (f.root/'METADATA_ADMISSION.json').exists()
    assert f.old.state_path.read_bytes()==before


def test_runtime_change_is_only_pin_count_and_comment():
    old=subprocess.check_output(['git','show','838dd21f^:tools/st_world_policy/metadata_storage.py'],text=True)
    new=Path(m.__file__).read_text()
    stripped=lambda text:'\n'.join(line for line in text.splitlines() if not line.lstrip().startswith('#'))
    assert stripped(new)==stripped(old).replace('len(pins) > 8','len(pins) > 16')
