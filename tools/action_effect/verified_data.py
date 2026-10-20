"""Reusable native-to-compact verification, with a checked immutable file inventory.

A path existing is insufficient. Initial validation replays every native receipt
and the deterministic compact builder. Reuse checks the compact bytes, source
closure and every dependency's device/inode/size/mtime/ctime. A changed file forces
full checksum/semantic verification; no whole native bank is decoded per step.
"""
from pathlib import Path
from contextlib import contextmanager
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


def validate_compact_dataset(path,*,kind):
    if kind not in ('pair','repair'):raise ValueError('Explicit compact dataset kind required')
    path=Path(path);m=verify_identity(read_json(path));a=artifact(path)
    if m.get('schema')!=f'action_effect_{kind}_dataset_v1' or m.get('status')!='COMPLETE':
        raise ValueError('Complete compact schema required')
    regpath=verify_artifact(m['registration']);root=regpath.parent
    expected={'dataset':a,'source':source_contract(),'kind':kind}
    key=digest(expected);versions=root/'validation_receipts'/key
    versions.mkdir(parents=True,exist_ok=True)
    def find_cached():
        for cache in sorted(versions.glob('*.json'),reverse=True):
            cached=_cached(cache,expected)
            if cached:return {'receipt':artifact(cache),'reused':True,'inventory_files':len(cached['inventory'])}
        return None
    reused=find_cached()
    if reused:return reused
    with (versions/'BUILD.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        reused=find_cached()
        if reused:return reused
        started=time.perf_counter()
        from .build_effect_pairs import build,read_population
        result=build(root)
        if result[kind+'_dataset']['sha256']!=a['sha256']:
            raise ValueError('Compact rows differ from their native feedback/scene receipts')
        reg,scenes=read_population(root);complete=read_json(verify_artifact(m['population_completion']))
        pins={}
        def add(pin):
            p=verify_artifact(pin);pins[str(p.resolve())]={'path':str(p.resolve()),'sha256':pin['sha256'],'stat':stamp(p)}
        add(a);add(m['registration']);add(m['population_completion'])
        add(result['pair_dataset']);add(result['repair_dataset'])
        for scene,pin in zip(scenes,complete['scenes']):
            add(pin);sr=read_json(pin['path']);add(sr['scene_record'])
            for row in scene['candidates']:
                if 'feedback' not in row:continue
                add(row['feedback']);add(row['canonical_receipt'])
                native=read_json(row['canonical_receipt']['path']);add(native['canonical'])
                admitted=Path(row['canonical_receipt']['path']).parent/'ADMITTED.json'
                add({'path':str(admitted),'sha256':native['admission_sha256']})
        r={'schema':'action_effect_native_validation_receipt_v1','dependency_identity':expected,
            'inventory':list(pins.values()),'full_validation_seconds':time.perf_counter()-started,
            'semantics':'native receipt and deterministic compact reconstruction; stat-checked immutable reuse'}
        r['identity']=digest(r);cache=versions/(r['identity']+'.json');write_json(cache,r)
        return {'receipt':artifact(cache),'reused':False,'inventory_files':len(pins),'seconds':r['full_validation_seconds']}
