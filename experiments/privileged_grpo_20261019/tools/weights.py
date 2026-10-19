#!/usr/bin/env python3
"""Lossless release asset split/restore with per-part and complete SHA256."""
import argparse,hashlib,json,os
from pathlib import Path
BLOCK=8*1024**2

def split(source,directory,part_bytes=1792*1024**2,expected=None):
    source=Path(source);directory=Path(directory)
    if not 0 < part_bytes < 2*1024**3:raise ValueError('Each Release asset must be under 2 GiB')
    directory.mkdir(parents=True,exist_ok=True)
    manifest=directory/(source.name+'.manifest.json')
    if manifest.exists():raise FileExistsError(manifest)
    result={'filename':source.name,'bytes':source.stat().st_size,'sha256':None,'parts':[]};full=hashlib.sha256()
    with source.open('rb') as src:
        left=result['bytes'];i=0
        while left:
            name=f'{source.name}.part{i:03d}';path=directory/name;size=min(left,part_bytes);h=hashlib.sha256()
            with path.open('xb') as dst:
                remaining=size
                while remaining:
                    data=src.read(min(remaining,BLOCK))
                    if not data:raise IOError('Source truncated while reading')
                    dst.write(data);full.update(data);h.update(data);remaining-=len(data)
            result['parts'].append({'name':name,'bytes':size,'sha256':h.hexdigest()});left-=size;i+=1
        if src.read(1):raise IOError('Source grew while reading')
    result['sha256']=full.hexdigest()
    if expected and result['sha256']!=expected:raise ValueError('Complete checkpoint SHA256 mismatch')
    with manifest.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    return result

def restore(manifest,destination):
    manifest=Path(manifest);info=json.loads(manifest.read_text());destination=Path(destination)
    if destination.exists():raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True);temp=destination.with_name(destination.name+'.partial')
    total=0;full=hashlib.sha256();owned=False
    try:
        with temp.open('xb') as output:
            owned=True
            for part in info['parts']:
                name=part['name']
                if Path(name).name!=name:raise ValueError('Invalid part filename')
                path=manifest.parent/name;h=hashlib.sha256();size=0
                with path.open('rb') as src:
                    for data in iter(lambda:src.read(BLOCK),b''):
                        output.write(data);h.update(data);full.update(data);size+=len(data)
                if size!=part['bytes'] or h.hexdigest()!=part['sha256']:raise ValueError('Corrupt part: '+name)
                total+=size
        if total!=info['bytes'] or full.hexdigest()!=info['sha256']:raise ValueError('Complete checksum mismatch')
        # Atomic no-clobber publication on the destination filesystem.
        os.link(temp,destination);temp.unlink()
    except BaseException:
        if owned and temp.exists():temp.unlink()
        raise
    return {'output':str(destination),'bytes':total,'sha256':full.hexdigest()}

def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('split');a.add_argument('source');a.add_argument('directory');a.add_argument('--expected-sha256');a.add_argument('--part-bytes',type=int,default=1792*1024**2)
    a=s.add_parser('restore');a.add_argument('manifest');a.add_argument('destination')
    a=p.parse_args()
    result=split(a.source,a.directory,a.part_bytes,a.expected_sha256) if a.command=='split' else restore(a.manifest,a.destination)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
