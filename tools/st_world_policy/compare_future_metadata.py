"""Read-only, <=32-scene legacy-reference qualification; JSON result to stdout.

No teacher, tensor payload, image decode, new request bank or cache-root writes.
Build and reference have separate <=4 GiB/450 s halves of one qualification.
Lead must account every invocation/retry and capture this <=1 MiB result in the
same registered 32 MiB control allowance; this tool does not issue permission.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

from PIL import Image

from tools.st_world_policy import prepare_future_metadata as f
from tools.st_world_policy import prepare_history as legacy
from starVLA.dataloader import st_world_history as legacy_images
from starVLA.dataloader.st_world_future_provenance import LogCoverage, validate_ledger, LEDGER_SCHEMA
from starVLA.dataloader.st_world_schema import identity_hash, require_hash

REFERENCE_READ_BYTES = 4*f.GiB
REFERENCE_SECONDS = 450.
MAX_SCENES = 32
HEADER_BYTES = f.MiB


def read_json(path,budget,*,limit=f.MiB,expected_sha=None):
    path=f.metadata_path(path); before=f.version(path)
    if not 0<before[2]<=limit: raise ValueError('Reference control metadata exceeds bounded size')
    with f.BudgetedReader(path,budget) as stream: data=stream.read(before[2]+1)
    if f.version(path)!=before or len(data)!=before[2]: raise ValueError('Reference control metadata changed')
    if expected_sha is not None and f.sha_bytes(data)!=require_hash(expected_sha,'reference input SHA'):
        raise ValueError('Reference control checksum mismatch')
    budget.check(); value=json.loads(data); budget.check()
    return value,f.sha_bytes(data)


def code_with_io(function,replacements):
    """Preserve the original code object/defaults; never mutate module globals."""
    scope=dict(function.__globals__); scope.update(replacements)
    result=FunctionType(function.__code__,scope,function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=dict(function.__kwdefaults__ or {})
    return result


class HeaderReader(f.BudgetedReader):
    def __init__(self,path,budget):
        super().__init__(path,budget); self.header_bytes=0

    def read(self,n=-1):
        if n<0: raise ValueError('Reference image header attempted unbounded read')
        if self.header_bytes+n>HEADER_BYTES: raise ValueError('Reference image header exceeded 1 MiB cap')
        data=super().read(n); self.header_bytes+=len(data)
        return data


def reference_scene_function(budget):
    @contextmanager
    def header_open(path):
        # PIL opens only its original header; never load()/convert()/resize().
        with HeaderReader(path,budget) as stream:
            with Image.open(stream) as image: yield image
    image_reference=code_with_io(legacy_images.image_reference,{
        'file_sha256':lambda path:f.budget_sha(path,budget),
        'Image':SimpleNamespace(open=header_open),
    })
    scene_function=code_with_io(legacy.build_target_scene,{'image_reference':image_reference})
    assert scene_function.__code__ is legacy.build_target_scene.__code__
    assert image_reference.__code__ is legacy_images.image_reference.__code__
    return scene_function


class Comparison:
    def __init__(self): self.checked_leaves=0; self.mismatch_count=0; self.first_mismatches=[]

    def difference(self,path,actual,expected):
        self.mismatch_count+=1
        if len(self.first_mismatches)<64:
            self.first_mismatches.append({'field':path,'actual':repr(actual)[:512],'expected':repr(expected)[:512]})

    def compare(self,actual,expected,path):
        if type(actual) is not type(expected): self.difference(path,actual,expected); return
        if isinstance(expected,dict):
            if set(actual)!=set(expected): self.difference(path+'.keys',sorted(actual),sorted(expected))
            for key in sorted(set(actual)&set(expected)): self.compare(actual[key],expected[key],path+'.'+key)
        elif isinstance(expected,list):
            if len(actual)!=len(expected): self.difference(path+'.length',len(actual),len(expected))
            for i,(a,b) in enumerate(zip(actual,expected)): self.compare(a,b,path+'['+str(i)+']')
        else:
            self.checked_leaves+=1
            if actual!=expected: self.difference(path,actual,expected)


def estimate_reference_reads(p,manifest,metadata_paths):
    """Conservative pre-read estimate, from pinned metadata and file stat only."""
    refs=[ref for scene in manifest['scenes'] for frame in scene['frames'] for ref in frame['views'] if ref is not None]
    # Legacy hashes each occurrence. Repeated frames/views are deliberately not
    # memoized in this reference. Header cap includes repeated/seeked reads.
    raw=2*sum(row['bytes'] for row in p['raw_log_stats'])
    image_hashes=sum(ref['bytes'] for ref in refs)
    headers=len(refs)*HEADER_BYTES
    student=Path(p['source']['student_root']); current=Path(p['original_selection']['path']).parent
    controls=[(student/'identity.json',1),(student/'index.json',2),
        (Path(p['source']['partition_path']),2),(student/'ego_identity.json',2),
        (current/'REGISTRATION.json',2),(current/'COMPLETE.json',2),(current/'SELECTED_SCENES.json',3)]
    source_controls=sum(path.stat().st_size*count for path,count in controls)
    source_records=sum((student/'current'/(row['token']+'.json')).stat().st_size+
        (student/'ego'/(row['token']+'.pt')).stat().st_size for row in p['selected_rows'])
    # Covers complete-receipt/shard replay and bounded control validation. Actual
    # reads are separately prepaid/count-tracked below and remain authoritative.
    generated_controls=8*sum(Path(path).stat().st_size for path in metadata_paths)
    return {'raw_log_hash_and_deserialization_bytes':raw,'legacy_repeated_image_hash_bytes':image_hashes,
        'image_header_upper_bytes':headers,'source_control_bytes':source_controls,
        'opaque_source_record_bytes':source_records,'generated_control_upper_bytes':generated_controls,
        'remaining_upper_bytes':raw+image_hashes+headers+source_controls+source_records+generated_controls,
        'valid_image_occurrences':len(refs),'unique_image_paths':len({ref['path'] for ref in refs})}


def compare(args):
    plan_path=f.metadata_path(args.plan); root=plan_path.parent
    limits={'workers':1,'read_mib_per_second':64.,'max_seconds':REFERENCE_SECONDS,
        'max_source_read_bytes':REFERENCE_READ_BYTES,'max_metadata_bytes':32*f.MiB,
        'max_rss_bytes':4*f.GiB,'max_raw_log_bytes':128*f.MiB}
    budget=f.PreparationBudget(root,limits)
    p,plan_sha=read_json(plan_path,budget,limit=8*f.MiB,expected_sha=args.plan_sha256)
    f.validate_self(p,f.PLAN_SCHEMA)
    if (p['scope']!='qualification' or not 1<=p['selected_scene_count']<=MAX_SCENES or
            p['limits']['max_source_read_bytes']>REFERENCE_READ_BYTES or p['limits']['max_seconds']>REFERENCE_SECONDS or
            p['limits']['max_metadata_bytes']>32*f.MiB or p['limits']['workers']!=1):
        raise ValueError('Requires a <=32 scene single-worker PLAN with build <=4 GiB/450 s and metadata <=32 MiB')
    code_root=Path(f.__file__).resolve().parents[2]
    implementation=[]
    for name in p['implementation']:
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or path.suffix!='.py': raise ValueError('Unsafe source pin')
        source=code_root/path
        if source.stat().st_size>f.MiB: raise ValueError('Source pin exceeds bounded code metadata')
        implementation.append(source)
    # Original registration verifier is reused unchanged, with exact small
    # reads prepaid, including its second plan hash and source-code hashes.
    budget.read_allowance(Path(args.registration).stat().st_size+2*plan_path.stat().st_size+
        sum(path.stat().st_size for path in implementation),'registration_revalidation')
    original,reg=f.verify_registration(plan_path,args.registration,args.registration_sha256)
    if original!=p: raise ValueError('Plan changed between pinned read and original verifier')
    done,complete_sha=read_json(root/'COMPLETE.json',budget,expected_sha=args.complete_sha256)
    build_cost=done.get('cost',{})
    if (type(build_cost.get('source_bytes_read')) is not int or not 0<=build_cost['source_bytes_read']<=REFERENCE_READ_BYTES or
            type(build_cost.get('elapsed_s')) not in (int,float) or not 0<=build_cost['elapsed_s']<=REFERENCE_SECONDS):
        raise ValueError('Completed build cost exceeds its half of the qualification budget')
    manifest,request_sha=read_json(root/'TEACHER_REQUESTS.json',budget,limit=32*f.MiB,
        expected_sha=done['files']['TEACHER_REQUESTS.json'])
    if len(manifest.get('scenes',[]))!=p['selected_scene_count']: raise ValueError('Reference scene population changed')
    ledger_path=root/'RAW_LOG_PROVENANCE.json'
    if manifest['source_logs_ledger']['path']!=str(ledger_path): raise ValueError('Foreign raw-log ledger')
    ledger,ledger_sha=read_json(ledger_path,budget,limit=4*f.MiB,expected_sha=done['files']['RAW_LOG_PROVENANCE.json'])
    rows=p['selected_rows']; groups=f.groups_for(rows)
    metadata_paths=[plan_path,Path(args.registration),root/'COMPLETE.json',root/'TEACHER_REQUESTS.json',ledger_path,
        root/'SHARD_RECEIPTS.json',*[f.shard_path(root,log) for log in groups]]
    estimate=estimate_reference_reads(p,manifest,metadata_paths)
    if budget.bytes+estimate['remaining_upper_bytes']>REFERENCE_READ_BYTES:
        raise ValueError('Reference pre-read upper estimate exceeds its 4 GiB half')
    source,index,binding,pins=f.source_with_pins(p['source']['student_root'],p['source']['partition_path'],
        p['source']['split'],Path(p['original_selection']['path']).parent,budget)
    selected={row['token'] for row in rows}
    if binding!=p['source'] or pins!=p['original_selection'] or [row for row in index if row['token'] in selected]!=rows:
        raise ValueError('Original source population/ordered selection mismatch')
    del index
    budget.read_allowance((root/'SHARD_RECEIPTS.json').stat().st_size,'generated_metadata')
    f.validate_completed(done,p,reg,rows,groups,pins,root,budget)
    if (manifest['identity']!=done['request_identity'] or
            identity_hash({k:v for k,v in manifest.items() if k!='identity'})!=manifest['identity']):
        raise ValueError('Request semantic identity changed')
    coverage=LogCoverage()
    for row in rows: coverage.add(row['token'],row['log'])
    budget.read_allowance(ledger_path.stat().st_size,'generated_metadata')
    validate_ledger(manifest,coverage=coverage.records())
    if [(row['token'],row['log']) for row in manifest['scenes']]!=[(row['token'],row['log']) for row in rows]:
        raise ValueError('Request order differs from fixed original selection')
    compare_values=Comparison(); expected_raw=[]; generated={}; scene_function=reference_scene_function(budget)
    reference_digest=hashlib.sha256(); missing=0
    for log,log_rows in groups.items():
        budget.check(); raw=Path(p['raw_log_root'])/(log+'.pkl'); stamp=f.version(raw)
        if stamp[2]>limits['max_raw_log_bytes']: raise ValueError('Raw log exceeds bounded reference input')
        raw_sha=f.budget_sha(raw,budget)
        with f.BudgetedReader(raw,budget) as stream: frames=legacy.TrustedNumpyUnpickler(stream).load()
        if f.version(raw)!=stamp: raise ValueError('Raw log changed during reference')
        cover=LogCoverage()
        for row in log_rows:
            budget.check(); token=row['token']; cover.add(token,log)
            student=Path(p['source']['student_root'])
            current,_=read_json(student/'current'/(token+'.json'),budget,expected_sha=row['current_record_sha256'])
            if f.budget_sha(student/'ego'/(token+'.pt'),budget)!=row['ego_record_sha256']:
                raise ValueError('Opaque ego source SHA changed')
            expected=scene_function(frames,token,log,p['source']['split'],p['sensor_roots'],
                times_s=f.TIMES,tolerance_s=f.TOLERANCE)
            budget.check()
            if (expected['timestamp_us']!=current['timestamp'] or
                    [ref['path'] for ref in expected['frames'][0]['views']]!=[str(Path(v).resolve()) for v in current['image_paths']]):
                raise ValueError('Legacy reference no longer matches original current input')
            generated[token]=expected
        expected_raw.append({**cover.records()[0],'path':str(raw),'sha256':raw_sha,'bytes':stamp[2],'mtime_ns':stamp[3]})
        del frames
    for i,row in enumerate(rows):
        expected=generated.pop(row['token']); compare_values.compare(manifest['scenes'][i],expected,'scenes['+str(i)+']')
        reference_digest.update(f.canonical(expected).encode())
        missing+=sum(ref is None for frame in expected['frames'] for ref in frame['views'])
    shared={'split':p['source']['split'],'source_student_identity':source['identity'],
        'source_index_sha256':source['index_sha256'],'partition_sha256':source['partition_sha256'],
        'selection_identity':p['selection_identity'],'registration_identity':reg['identity']}
    expected_ledger=f.signed({'schema':LEDGER_SCHEMA,**shared,'logs':expected_raw})
    compare_values.compare(ledger,expected_ledger,'raw_log_ledger')
    expected_header={'schema':'st_teacher_requests_v1',**shared,'view_names':list(legacy_images.VIEWS),
        'timestamp_tolerance_s':f.TOLERANCE,'times_s':list(f.TIMES),
        'source':'original_raw_log_future_frames_v1','source_logs_ledger':manifest['source_logs_ledger']}
    compare_values.compare({k:v for k,v in manifest.items() if k not in ('scenes','identity')},expected_header,'request_header')
    reference_sources={str(Path(module.__file__).relative_to(code_root)):f.budget_sha(module.__file__,budget)
        for module in (legacy,legacy_images)}
    comparator_sha=f.budget_sha(__file__,budget)
    budget.check(); cost=budget.snapshot()
    return {'schema':'st_future_metadata_reference_v1','status':'REFERENCE_MATCH' if compare_values.mismatch_count==0 else 'REFERENCE_MISMATCH',
        'plan_sha256':plan_sha,'plan_identity':p['identity'],'complete_sha256':complete_sha,
        'requests_sha256':request_sha,'ledger_sha256':ledger_sha,'scenes':len(rows),'logs':len(groups),
        'time_view_slots':len(rows)*5*3,'missing_view_slots':missing,'checked_leaf_fields':compare_values.checked_leaves,
        'mismatch_count':compare_values.mismatch_count,'first_mismatches':compare_values.first_mismatches,
        'reference_scene_digest':reference_digest.hexdigest(),'reference_sources':reference_sources,
        'comparator_source_sha256':comparator_sha,
        'reference_code_objects_reused':True,'original_module_globals_modified':False,
        'reference_read_estimate':estimate,'build_invocation_cost':build_cost,'reference_invocation_cost':cost,
        'combined_observed_source_bytes':build_cost['source_bytes_read']+cost['source_bytes_read'],
        'combined_observed_seconds':build_cost['elapsed_s']+cost['elapsed_s'],
        'per_arm_limits':{'max_source_read_bytes':REFERENCE_READ_BYTES,'max_seconds':REFERENCE_SECONDS},
        'combined_limits':{'max_source_read_bytes':8*f.GiB,'max_seconds':900},
        'prior_failed_invocations_accounted':False,'outer_cumulative_invocation_ledger_required':True,
        'teacher_extractions':0,'tensor_disk_bytes':0,'new_cache_files_written':0,'scientific_result':False,
        'training_admission':False}


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','plan-sha256','registration','registration-sha256','complete-sha256'):
        p.add_argument('--'+name,required=True)
    return p


def main():
    try:
        result=compare(parser().parse_args()); status=0 if result['status']=='REFERENCE_MATCH' else 2
    except Exception as exc:
        result={'schema':'st_future_metadata_reference_v1','status':'REFERENCE_FAILED','error':str(exc),
            'new_cache_files_written':0,'tensor_disk_bytes':0,'training_admission':False}; status=2
    encoded=json.dumps(result,sort_keys=True,allow_nan=False)
    if len(encoded.encode())>f.MiB: raise ValueError('Reference stdout exceeds 1 MiB control allowance')
    print(encoded)
    return status


if __name__=='__main__': raise SystemExit(main())
