"""Create a new front-camera path view of existing copies; never change source data."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import pickle


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('manifest','logs','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--source',action='append',required=True)
    a=p.parse_args();output=Path(a.output)
    if output.exists():raise FileExistsError('Preserve previous sensor views: '+str(output))
    sources=[Path(s).resolve() for s in a.source];groups={}
    for r in json.loads(Path(a.manifest).read_text())['records']:groups.setdefault(r['log'],[]).append(r['token'])
    def plan(item):
        log,tokens=item
        with (Path(a.logs)/(log+'.pkl')).open('rb') as f:raw=pickle.load(f)
        index={r['token']:i for i,r in enumerate(raw)};paths=set()
        for token in tokens:
            i=index[token]
            for off in (0,1,3,8):
                if i+off<len(raw):
                    rel=Path(raw[i+off]['cams']['CAM_F0']['data_path'])
                    if rel.is_absolute() or '..' in rel.parts or rel.parts[:2]!=(log,'CAM_F0'):
                        raise ValueError('Unexpected raw camera path '+str(rel))
                    paths.add(rel)
        available=[{r for r in paths if (s/r).is_file()} for s in sources]
        missing=paths-set().union(*available)
        if missing:return dict(log=log,missing=sorted(map(str,missing)))
        whole=next((i for i,found in enumerate(available) if found==paths),None)
        if whole is not None:return dict(log=log,source=str(sources[whole]/log),required=len(paths))
        return dict(log=log,files={str(r):str(next(s/r for s,found in zip(sources,available) if r in found)) for r in sorted(paths)},required=len(paths))
    with ThreadPoolExecutor(16) as pool:plans=list(pool.map(plan,sorted(groups.items())))
    missing=[f for r in plans for f in r.get('missing',[])]
    identity=hashlib.sha256(json.dumps(plans,sort_keys=True).encode()).hexdigest()
    if missing:
        report=output.with_suffix('.missing.json');report.write_text(json.dumps(dict(status='FAIL',missing=missing),indent=2))
        raise FileNotFoundError('Images absent from all existing copies: '+str(len(missing))+'; '+str(report))
    output.mkdir(parents=True)
    for r in plans:
        if 'source' in r:(output/r['log']).symlink_to(r['source'],target_is_directory=True)
        else:
            (output/r['log']/'CAM_F0').mkdir(parents=True)
            for rel,source in r['files'].items():(output/rel).symlink_to(source)
    report=dict(status='PATHS_AVAILABLE_NOT_YET_DECODED',source_roots=list(map(str,sources)),logs=len(plans),
        required_unique_frames=sum(r['required'] for r in plans),whole_log_links=sum('source' in r for r in plans),
        manifest_sha256=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),view_plan_sha256=identity,
        sources_modified=False,plans=plans)
    output.with_suffix('.provenance.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='plans'}),flush=True)


if __name__=='__main__':main()
