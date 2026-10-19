#!/usr/bin/env python3
"""Publish already prepared weight assets via an existing Git credential helper.
Never accepts tokens on the command line or writes them to files.
"""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
import requests
REPO='IDayday/VLA-Drive';API='https://api.github.com/repos/'+REPO

def main():
    p=argparse.ArgumentParser();p.add_argument('--repository',required=True);p.add_argument('--assets',required=True);p.add_argument('--commit',required=True);p.add_argument('--tag',default='privileged-grpo-base-20261019');p.add_argument('--receipt',required=True)
    a=p.parse_args();assets=Path(a.assets);files={}
    for path in sorted(assets.glob('*.manifest.json')):
        m=json.loads(path.read_text())
        files[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        for part in m['parts']:
            name=part['name']
            if Path(name).name!=name:raise SystemExit('Invalid asset path')
            file=assets/name
            if file.stat().st_size!=part['bytes']:raise SystemExit('Incorrect asset size: '+name)
            h=hashlib.sha256()
            with file.open('rb') as f:
                for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
            if h.hexdigest()!=part['sha256']:raise SystemExit('Corrupt local asset: '+name)
            files[name]=part['sha256']
    if not files:raise SystemExit('No assets')
    result=subprocess.run(['git','-C',a.repository,'credential','fill'],input='protocol=https\nhost=github.com\npath='+REPO+'.git\n\n',text=True,capture_output=True,env={**os.environ,'GIT_TERMINAL_PROMPT':'0'})
    credential=dict(x.split('=',1) for x in result.stdout.splitlines() if '=' in x)
    token=credential.get('password')
    if result.returncode or not token:raise SystemExit('No configured Git credential. Run the interactive login helper first.')
    session=requests.Session();session.headers.update({'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28','User-Agent':'DriveExpert-weight-migration'})
    del credential,token,result
    def request(method,url,**kw):
        response=session.request(method,url,timeout=(30,3600),**kw)
        if not response.ok:raise RuntimeError('GitHub API failed: HTTP '+str(response.status_code))
        return response.json()
    # Search drafts too, so interrupted uploads resume without duplicate releases.
    releases=[];page=1
    while True:
        data=request('GET',API+'/releases',params={'per_page':100,'page':page});releases+=data
        if len(data)<100:break
        page+=1
    matches=[r for r in releases if r['tag_name']==a.tag]
    if len(matches)>1:raise RuntimeError('Ambiguous release')
    if matches:release=matches[0]
    else:release=request('POST',API+'/releases',json={'tag_name':a.tag,'target_commitish':a.commit,'name':'Action-only DiT Base and initialization weights (2026-10-19)','draft':True,'body':'Frozen Base checkpoint for the privileged-feedback GRPO experiment. This is not a GRPO-trained expert. Restore using experiments/privileged_grpo_20261019/tools/weights.py. SHA256 and byte counts are recorded in each manifest. Includes Qwen initialization weights required by the original model constructor.'})
    existing={r['name']:r for r in request('GET',API+'/releases/'+str(release['id'])+'/assets',params={'per_page':100})}
    for name,h in files.items():
        if name in existing:
            item=existing[name]
            if item.get('digest')!='sha256:'+h or item.get('size')!=(assets/name).stat().st_size:raise RuntimeError('Existing asset differs; refusing overwrite: '+name)
        else:
            print('Uploading '+name,flush=True)
            file=assets/name
            with file.open('rb') as f:
                item=request('POST',release['upload_url'].split('{')[0],params={'name':name},headers={'Content-Type':'application/octet-stream','Content-Length':str(file.stat().st_size)},data=f)
            if item.get('digest')!='sha256:'+h:raise RuntimeError('Server asset digest not confirmed: '+name)
        existing[name]=item
    if release['draft']:release=request('PATCH',API+'/releases/'+str(release['id']),json={'draft':False})
    receipt={'repository':REPO,'commit':a.commit,'tag':a.tag,'url':release['html_url'],'assets':{n:{'sha256':files[n],'url':existing[n]['browser_download_url'],'bytes':existing[n]['size']} for n in files}}
    dest=Path(a.receipt);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'status':'PUBLISHED','url':release['html_url']}))
if __name__=='__main__':
    try:main()
    except Exception as exc:
        # HTTP exceptions may contain sensitive request details; do not print tracebacks.
        print('Upload failed: '+(str(exc) if isinstance(exc,RuntimeError) else type(exc).__name__),file=sys.stderr);sys.exit(1)
