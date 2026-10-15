"""Stage byte-identical current A images on shared project storage.

The legacy S0 reader already supports SHA256(path)-named files. This only copies
original JPEG bytes, with bounded threads; it does not encode visual features,
resize images, change cameras, or include B in the driving population.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib,os,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import check_host,read,atomic_json,identity,digest,source_sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('population-contract','current-root','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--original-root',type=Path,required=True);p.add_argument('--fallback-root',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--max-seconds',type=float,required=True)
    a=p.parse_args();resources=check_host();start=time.monotonic();cpu=time.process_time()
    if not 1<=a.workers<=16 or a.max_seconds<=0:raise ValueError('Finite I/O workers and time required')
    contract=read(a.population_contract);audit=read(contract['source_audit']);rows=read(contract['index'])
    proof=read(contract['recording_frame_audit'])
    if (contract['role']!='A_train' or audit['identity']!=contract['source_audit_identity'] or
        not proof['population_ready'] or proof['source_audit_identity']!=audit['identity'] or
        any(audit['roles'][r['log'].rsplit('_',2)[0]]!='A_train' for r in rows)):
        raise ValueError('Only the audited A training role may enter this cache')
    current=read(a.current_root/'identity.json')
    declaration=dict(schema='a_only_current_image_bytes_v1',population_contract_sha256=digest(a.population_contract),
        source_current_identity=current['identity'],source_audit_identity=audit['identity'],index_identity=identity(rows),
        original_root=str(a.original_root),fallback_root=str(a.fallback_root),code_sha=source_sha(ROOT),
        image_transformation='none; exact original bytes; legacy current ROI/resize happens on read',future_inputs=False)
    ident=identity(declaration);a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'identity.json').exists() and read(a.output/'identity.json')['identity']!=ident:raise ValueError('Preserve old image cache identity')
    atomic_json(a.output/'identity.json',dict(identity=ident,**declaration))
    for folder in ('images','records'):(a.output/folder).mkdir(exist_ok=True)
    def current_paths(row):
        obj=read(a.current_root/'current'/(row['token']+'.json'))
        if obj['identity']!=current['identity'] or obj['token']!=row['token'] or len(obj['image_paths'])!=3:
            raise ValueError('Wrong current input row')
        return obj['image_paths']
    with ThreadPoolExecutor(a.workers) as pool:paths=sorted(set(path for group in pool.map(current_paths,rows) for path in group))
    def copy(path):
        if time.monotonic()-start>=a.max_seconds:return None
        name=hashlib.sha256(path.encode()).hexdigest();dest=a.output/'images'/(name+'.jpg');rec=a.output/'records'/(name+'.json')
        if rec.exists():
            data=read(rec)
            if data['identity']!=ident or data['logical_source']!=path or digest(dest)!=data['sha256']:raise ValueError('Changed staged bytes')
            return data
        source=Path(path);fallback=False
        if not source.is_file():
            source=a.fallback_root/source.relative_to(a.original_root);fallback=True
        if not source.is_file():raise FileNotFoundError('Current image absent in registered original and fallback roots: '+path)
        tmp=dest.with_suffix(f'.{os.getpid()}.tmp');shutil.copyfile(source,tmp)
        source_sha=digest(source);copied_sha=digest(tmp)
        if source_sha!=copied_sha:raise ValueError('Source changed while staging')
        tmp.replace(dest);data=dict(identity=ident,logical_source=path,resolved_source=str(source),
            sha256=source_sha,bytes=dest.stat().st_size,fallback_used=fallback)
        atomic_json(rec,data);return data
    with ThreadPoolExecutor(a.workers) as pool:results=[r for r in pool.map(copy,paths) if r is not None]
    report=dict(identity=ident,expected_images=len(paths),completed_images=len(results),scenes=len(rows),
        bytes=sum(r['bytes'] for r in results),fallback_images=sum(r['fallback_used'] for r in results),
        wall_seconds=time.monotonic()-start,parent_CPU_seconds=time.process_time()-cpu,resources=resources,
        complete=len(results)==len(paths),feature_encoding_calls=0)
    atomic_json(a.output/('COMPLETE.json' if report['complete'] else 'INCOMPLETE.json'),report);print(report)
    if not report['complete']:raise SystemExit(1)


if __name__=='__main__':main()
