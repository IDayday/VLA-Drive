"""Create a reproducible small review archive from committed git objects only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def git(*args):return subprocess.check_output(['git',*args])


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--base',default='6e1d9f8c6f91f5ddb2d04461554e0a7d27208084')
    p.add_argument('--tested-commit',required=True);p.add_argument('--delivery-commit',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError('Never overwrite a review archive')
    reports_only=git('diff','--name-only',a.tested_commit,a.delivery_commit).decode().splitlines()
    permitted_roots=('docs/','reports/','SOURCE_PROVENANCE.md','IMPLEMENTATION_MATRIX.md','VALIDATION_REPORT.md')
    if any(not n.startswith(permitted_roots) for n in reports_only):raise ValueError('Production/config/tests changed after TESTED_CODE_COMMIT')
    names=git('diff','--name-only','--diff-filter=ACMR',a.base,a.delivery_commit).decode().splitlines()
    payload={'review.diff':git('diff','--binary',a.base,a.delivery_commit)}
    allowed={'.py','.sh','.yaml','.yml','.md','.json','.xml','.log','.txt'}
    for name in names:
        if Path(name).suffix not in allowed:raise ValueError('Non-source/report file excluded: '+name)
        data=git('show',a.delivery_commit+':'+name)
        if len(data)>5*1024**2:raise ValueError('Large artifact is not allowed in review bundle: '+name)
        payload['files/'+name]=data
    manifest=dict(base=a.base,tested_code_commit=a.tested_commit,delivery_commit=a.delivery_commit,
        delivery_changes_reports_only=True,delivery_only_paths=reports_only,
        files={n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in payload.items()})
    payload['MANIFEST.json']=json.dumps(manifest,indent=2,sort_keys=True).encode()
    payload['SHA256SUMS']=('\n'.join(hashlib.sha256(b).hexdigest()+'  '+n for n,b in sorted(payload.items()))+'\n').encode()
    with zipfile.ZipFile(a.output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(payload.items()):archive.writestr(name,data)
    print(json.dumps(dict(output=a.output,sha256=hashlib.sha256(Path(a.output).read_bytes()).hexdigest(),files=len(payload))))


if __name__=='__main__':main()
