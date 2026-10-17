"""CPU-only negative fixtures: never calls plan, imports runtime, or dispatches."""
import copy
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

spec=importlib.util.spec_from_file_location('future_plan_control',Path(__file__).with_name('plan_once.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture(monkeypatch, role='dev'):
    monkeypatch.setattr(m.os,'sched_getaffinity',lambda _: {0,1})
    scope={k:False for k in ('build_authorized','gpu','image_read','index_authorized',
        'migration_authorized','model_load','raw_log_read','teacher_tensor_read_or_write','training_authorized')}
    scope.update(role=role,exact_scenes=1696 if role=='dev' else 101592)
    out=str(m.BASE/'caches'/('full_future_metadata_'+role+'_v1'))
    a=dict(schema='st_full_future_plan_resource_admission_v1',status='PASS_BOUNDED_'+role.upper()+'_PLAN_ONLY',
        limits=copy.deepcopy(m.LIMITS),scope=scope,issued_at_unix=1,latest_start_unix=100,valid_until_unix=500,output_root=out)
    values={'--student-root':'/fixture/student','--partition':'/fixture/partition','--split':role,
        '--current-population':'/fixture/current','--raw-log-root':'/fixture/raw','--sensor-root':'/fixture/sensor',
        '--output':out,'--scope':'full_population','--max-scenes':str(scope['exact_scenes']),
        '--workers':'1','--read-mib-per-second':'64','--max-seconds':'1800' if role=='dev' else '14400',
        '--max-source-read-bytes':str((16 if role=='dev' else 512)<<30),
        '--max-metadata-bytes':str(128<<20 if role=='dev' else 2<<30),'--max-rss-bytes':str(4<<30),
        '--max-raw-log-bytes':str((128 if role=='dev' else 192)<<20)}
    argv=[m.PYTHON,'-m','tools.st_world_policy.prepare_future_metadata','plan']
    for k,v in values.items():argv.extend((k,v))
    return a,argv


@pytest.mark.parametrize('field,value',[('status','APPROVED'),('schema','unknown'),('latest_start_unix',0),
    ('valid_until_unix',110),('valid_until_unix',float('nan')),('issued_at_unix',True),
    ('output_root','/tmp/arbitrary')])
def test_reject_admission_mutations(monkeypatch,field,value):
    a,v=fixture(monkeypatch);a[field]=value
    with pytest.raises(ValueError):m.validate(a,v,10)


@pytest.mark.parametrize('field',['build_authorized','gpu','image_read','index_authorized','migration_authorized',
    'model_load','raw_log_read','teacher_tensor_read_or_write','training_authorized'])
def test_reject_extra_scope(monkeypatch,field):
    a,v=fixture(monkeypatch);a['scope'][field]=True
    with pytest.raises(ValueError):m.validate(a,v,10)


@pytest.mark.parametrize('change',['build','python','extra','duplicate','population','raw','budget','cpu'])
def test_reject_argv_budget_changes(monkeypatch,change):
    a,v=fixture(monkeypatch)
    if change=='build':v[3]='build'
    elif change=='python':v[0]='/root/miniconda3/envs/navsim/bin/python'
    elif change=='extra':v+=['--execute','yes']
    elif change=='duplicate':v+=['--workers','1']
    elif change=='population':v[v.index('--max-scenes')+1]='32'
    elif change=='raw':v[v.index('--max-raw-log-bytes')+1]=str(192<<20)
    elif change=='budget':a['limits']['wall_seconds']=1800
    else:monkeypatch.setattr(m.os,'sched_getaffinity',lambda _: {0})
    with pytest.raises(ValueError):m.validate(a,v,10)


def test_train_cannot_reuse_dev_raw_limit(monkeypatch):
    a,v=fixture(monkeypatch,'train');v[v.index('--max-raw-log-bytes')+1]=str(128<<20)
    with pytest.raises(ValueError):m.validate(a,v,10)


def test_train_cannot_reuse_dev_build_proposal_limits(monkeypatch):
    a,v=fixture(monkeypatch,'train');v[v.index('--max-seconds')+1]='1800'
    with pytest.raises(ValueError):m.validate(a,v,10)


def test_directory_and_allocated_blocks_count_toward_root_limit(tmp_path,monkeypatch):
    directory=tmp_path/'control';directory.mkdir();path=directory/'small';path.write_bytes(b'x')
    entries=[tmp_path,directory,path]
    expected=sum(max(p.stat().st_size,p.stat().st_blocks*512) for p in entries)
    assert m.root_bytes(tmp_path)==expected
    monkeypatch.setitem(m.LIMITS,'root_inclusive_bytes_including_control_temp_logs',expected-1)
    with pytest.raises(ValueError,match='Root-inclusive'):m.root_bytes(tmp_path)


@pytest.mark.parametrize('kind',['symlink','large','root','nonregular'])
def test_root_accounting_rejects_invalid_outputs(tmp_path,kind):
    if kind=='symlink':(tmp_path/'bad').symlink_to('/tmp')
    elif kind=='large':
        with (tmp_path/'bad').open('wb') as f:f.truncate((4<<20)+1)
    elif kind=='nonregular':m.os.mkfifo(tmp_path/'bad')
    else:
        with (tmp_path/'bad').open('wb') as f:f.truncate(4<<20)
    with pytest.raises(ValueError):m.root_bytes(tmp_path,reserve=32<<20 if kind=='root' else 0)


def test_changed_source_rejected(tmp_path):
    path=tmp_path/'source.py';path.write_text('old')
    pin={'path':str(path),'sha256':hashlib.sha256(b'old').hexdigest()};path.write_text('new')
    with pytest.raises(ValueError):m.pinned(pin,json_value=False)


def test_existing_root_never_loads_runtime_or_dispatches(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'load_controls',lambda _: ({},[],tmp_path,None,None,None))
    with pytest.raises(ValueError,match='Single-attempt'):m.run(None)


@pytest.mark.parametrize('changed_field',['st_atime_ns','st_dev','st_ino','st_mode',
    'st_nlink','st_size','st_mtime_ns','st_ctime_ns','st_blocks'])
def test_pinned_content_version_excludes_only_access_time(tmp_path,monkeypatch,changed_field):
    path=tmp_path/'control.json';raw=b'{"unchanged":true}';path.write_bytes(raw)
    pin={'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}
    original=Path.stat;info=original(path);calls=[]
    fields=('st_atime_ns','st_dev','st_ino','st_mode','st_nlink','st_size','st_mtime_ns','st_ctime_ns','st_blocks')
    before={name:getattr(info,name) for name in fields}
    def observed(p,*a,**kw):
        if p!=path or kw.get('follow_symlinks') is False:return original(p,*a,**kw)
        values=dict(before)
        if calls:values[changed_field]+=1
        calls.append(True)
        return SimpleNamespace(**values)
    monkeypatch.setattr(Path,'stat',observed)
    if changed_field=='st_atime_ns':assert m.pinned(pin)=={'unchanged':True}
    else:
        with pytest.raises(ValueError,match='Pinned source/control changed'):m.pinned(pin)
    assert len(calls)==2
