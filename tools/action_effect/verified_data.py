"""Reusable native-to-compact verification, with a checked immutable file inventory.

A path existing is insufficient. Initial validation replays every native receipt
and the deterministic compact builder. Reuse checks the compact bytes, source
closure and every dependency's device/inode/size/mtime/ctime. A changed file forces
full checksum/semantic verification; no whole native bank is decoded per step.
"""
from pathlib import Path
import fcntl,time
from .common import read_json,write_json,verify_identity,verify_artifact,artifact,digest,sha256


def stamp(path):
    s=Path(path).stat()
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]


def source_contract():
    root=Path(__file__).resolve().parents[2]
    paths=['tools/action_effect/verified_data.py','tools/action_effect/common.py','tools/action_effect/build_effect_pairs.py',
        'tools/action_effect/simulate_interventions.py','tools/action_effect/feedback_schema.py',
        'tools/action_effect/navsim_feedback_adapter.py','starVLA/model/modules/st_world/consequence.py']
    return {p:sha256(root/p) for p in paths}


def _cached(path,expected):
    if not path.exists():return None
    r=verify_identity(read_json(path))
    if r['dependency_identity']!=expected:raise ValueError('Verification receipt identity conflict')
    for x in r['inventory']:
        if stamp(x['path'])!=x['stat']:return None
    return r


def _dataset(path,kind):
    if kind not in ('pair','repair'):raise ValueError('Explicit compact dataset kind required')
    m=verify_identity(read_json(path))
    if (m.get('schema')!=f'action_effect_{kind}_dataset_v1' or m.get('status')!='COMPLETE'
            or m.get('split')!='train' or m.get('deployment_inputs') is not False):
        raise ValueError('Complete TRAIN compact schema required')
    return m,artifact(path)


def _find_cached(root,expected):
    versions=root/'validation_receipts'/digest(expected)
    for cache in sorted(versions.glob('*.json'),reverse=True):
        cached=_cached(cache,expected)
        if cached:return {'receipt':artifact(cache),'reused':True,'inventory_files':len(cached['inventory'])}
    return None


def _inventory(root,result,observed):
    """Hash the complete dependency graph without decoding native bodies again.

    Only called immediately after this module's own strict build. Small scene and
    receipt metadata enumerate every body checked by that build; their pins and
    joins are independently checked here before the inventory is sealed.
    """
    pins={}
    def add(pin):
        p=Path(pin['path']).resolve();key=str(p);before=stamp(p)
        if key not in dataset_paths:
            expected=observed.get(key)
            if expected is None or expected['sha256']!=pin['sha256'] or expected['stat']!=before:
                raise ValueError('Dependency changed since native semantic validation: '+key)
        if key in pins:
            if pins[key]['sha256']!=pin['sha256'] or pins[key]['stat']!=before:
                raise ValueError('Conflicting/changed validation dependency: '+key)
            return p
        verify_artifact(pin)
        if stamp(p)!=before:raise ValueError('Dependency changed during verification: '+key)
        pins[key]={'path':key,'sha256':pin['sha256'],'stat':before}
        return p
    dataset_paths={str(Path(result[k+'_dataset']['path']).resolve()) for k in ('pair','repair')}
    regpin=artifact(root/'REGISTRATION.json');reg=verify_identity(read_json(add(regpin)))
    completepin=artifact(root/'COMPLETE.json');complete=read_json(add(completepin))
    if (reg.get('split')!='train' or complete['registration_identity']!=reg['identity']
            or len(reg['population'])!=reg['max_samples'] or len(complete['scenes'])!=reg['max_samples']):
        raise ValueError('Incomplete/foreign TRAIN population while sealing verification')
    for kind in ('pair','repair'):
        pin=result[kind+'_dataset'];m,actual=_dataset(add(pin),kind)
        if (pin!=actual or m['registration']!=regpin or m['population_completion']!=completepin
                or m['registration_identity']!=reg['identity']):
            raise ValueError('Dataset population changed during verification')
    for item,pin in zip(reg['population'],complete['scenes']):
        receipt=read_json(add(pin));scene=read_json(add(receipt['scene_record']))
        entry=item['context']
        if (receipt['registration_identity']!=reg['identity'] or scene['registration_identity']!=reg['identity']
                or scene['context_entry']!=entry
                or any(scene[k]!=entry[k] for k in ('scene_id','source_log','role'))):
            raise ValueError('Scene/context changed during verification')
        for row in scene['candidates']:
            if 'feedback' not in row:continue
            add(row['feedback']);native=read_json(add(row['canonical_receipt']))
            add(native['canonical'])
            admitted=Path(row['canonical_receipt']['path']).parent/'ADMITTED.json'
            add({'path':str(admitted),'sha256':native['admission_sha256']})
    for item in pins.values():
        if stamp(item['path'])!=item['stat']:raise ValueError('Dependency changed before verification seal')
    return list(pins.values())


def _capture_dependency(pin,observed,*,verified_stat=None):
    """Capture a file before semantic use, or merge that version from a worker."""
    p=Path(pin['path']).resolve();key=str(p);before=stamp(p)
    if verified_stat is not None and before!=verified_stat:
        raise ValueError('Dependency changed after worker native validation: '+key)
    if key in observed:
        if observed[key]['sha256']!=pin['sha256'] or observed[key]['stat']!=before:
            raise ValueError('Dependency changed during native validation: '+key)
        return
    verify_artifact(pin)
    if stamp(p)!=before:raise ValueError('Dependency changed during native validation: '+key)
    observed[key]={'sha256':pin['sha256'],'stat':before}


def build_and_validate_datasets(root,*,workers=1):
    """Perform one strict native reconstruction and seal BOTH compact datasets.

    No external result or claimed-valid flag is accepted. A valid immutable
    inventory can be reused; otherwise this entry point owns the actual build.
    The root lock prevents concurrent pair/repair parsers duplicating that work.
    """
    if type(workers) is not int or not 1<=workers<=4:raise ValueError('CPU build workers must lie in 1..4')
    root=Path(root);directory=root/'validation_receipts';directory.mkdir(parents=True,exist_ok=True)
    with (directory/'FULL_BUILD.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        source=source_contract();cached={};existing={}
        for kind in ('pair','repair'):
            path=root/(kind.upper()+'_DATASET.json')
            if not path.exists():break
            _,pin=_dataset(path,kind);existing[kind+'_dataset']=pin
            cached[kind]=_find_cached(root,{'dataset':pin,'source':source,'kind':kind})
        if len(cached)==2 and all(cached.values()):
            return {'status':'COMPLETE',**existing,'validation':cached,'reused':True}
        started=time.perf_counter()
        from .build_effect_pairs import build
        observed={}
        def observe(pin,*,verified_stat=None):_capture_dependency(pin,observed,verified_stat=verified_stat)
        result=build(root,_dependency_observer=observe,workers=workers)
        inventory=_inventory(root,result,observed)
        if source_contract()!=source:raise ValueError('Validation source changed during reconstruction')
        seconds=time.perf_counter()-started;validated={}
        for kind in ('pair','repair'):
            expected={'dataset':result[kind+'_dataset'],'source':source,'kind':kind}
            r={'schema':'action_effect_native_validation_receipt_v1','dependency_identity':expected,
                'inventory':inventory,'full_validation_seconds':seconds,
                'semantics':'native receipt and deterministic compact reconstruction; stat-checked immutable reuse'}
            r['identity']=digest(r);cache=directory/digest(expected)/(r['identity']+'.json');write_json(cache,r)
            validated[kind]={'receipt':artifact(cache),'reused':False,'inventory_files':len(inventory),'seconds':seconds}
        return {**result,'validation':validated,'reused':False}


def validate_compact_dataset(path,*,kind):
    m,a=_dataset(path,kind);root=verify_artifact(m['registration']).parent
    expected={'dataset':a,'source':source_contract(),'kind':kind}
    cached=_find_cached(root,expected)
    if cached:return cached
    result=build_and_validate_datasets(root)
    if result[kind+'_dataset']['sha256']!=a['sha256']:
        raise ValueError('Compact rows differ from their native feedback/scene receipts')
    return result['validation'][kind]
