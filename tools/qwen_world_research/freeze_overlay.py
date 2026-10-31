"""Publish a small immutable Git-delta overlay of an existing frozen runtime.

Only changed ancestors are listed; unchanged siblings/subtrees are symlinks to
the trusted frozen base. Changed bytes come from Git blobs at an exact commit,
never dirty worktree files. No archive or traversal of historical resources.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import time
import uuid


def sha_bytes(value):return hashlib.sha256(value).hexdigest()


def fingerprint(value):
    return sha_bytes(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())


def git(repo,*args):
    return subprocess.run(['git','-C',str(repo),*args],check=True,stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE).stdout


def fixed_delta(repo,base_commit,commit):
    """Raw Git records carry full blob IDs/modes; no worktree/index scanning."""
    raw=git(repo,'diff','--raw','--no-abbrev','--no-renames','-z',base_commit,commit)
    fields=raw.split(b'\0');out=[]
    for i in range(0,len(fields)-1,2):
        header=fields[i].decode();path=os.fsdecode(fields[i+1])
        old_mode,new_mode,old_oid,new_oid,status=header.removeprefix(':').split()
        parts=PurePosixPath(path).parts
        if not parts or path.startswith('/') or '..' in parts or path=='FROZEN_SOURCE.json':
            raise ValueError('Unsafe/reserved changed Git path: '+path)
        if status not in ('A','M','D','T') or new_mode not in ('000000','100644','100755','120000'):
            raise ValueError('Unsupported overlay delta status/mode: '+header)
        data=None if new_mode=='000000' else git(repo,'cat-file','blob',new_oid)
        out.append(dict(path=path,status=status,old_mode=old_mode,new_mode=new_mode,
            old_git_blob=old_oid,new_git_blob=new_oid,
            sha256=None if data is None else sha_bytes(data),bytes=0 if data is None else len(data),data=data))
    return out


def freeze_overlay(repo,base,commit,output):
    repo=Path(repo).resolve();base=Path(base).resolve();output=Path(output).absolute()
    if output.exists() or output.is_symlink():raise FileExistsError('Published overlay is immutable: '+str(output))
    if output==base or base in output.parents:raise ValueError('Overlay must not mutate/descend into base')
    base_raw=(base/'FROZEN_SOURCE.json').read_bytes();base_meta=json.loads(base_raw)
    base_commit=git(repo,'rev-parse','--verify',base_meta['commit']+'^{commit}').decode().strip()
    commit=git(repo,'rev-parse','--verify',str(commit)+'^{commit}').decode().strip()
    entries=fixed_delta(repo,base_commit,commit)
    if not entries:raise ValueError('No committed delta; reuse the existing frozen base')
    tree={}
    for e in entries:
        node=tree
        for part in PurePosixPath(e['path']).parts:node=node.setdefault(part,{})
        node['__entry__']=e
    def validate_tree(node):
        if '__entry__' in node:
            if len(node)>1:raise ValueError('File/directory structural transitions require a fresh base snapshot')
        else:
            for child in node.values():validate_tree(child)
    validate_tree(tree)
    output.parent.mkdir(parents=True,exist_ok=True)
    attempt=output.with_name(output.name+'.incomplete.'+uuid.uuid4().hex)
    attempt.mkdir();linked=[];expanded=[]

    def materialize(relative,node):
        dst=attempt/relative;src=base/relative
        if '__entry__' in node:
            e=node['__entry__']
            if e['new_mode']=='000000':return
            if e['new_mode']=='120000':os.symlink(os.fsdecode(e['data']),dst)
            else:
                with dst.open('xb') as stream:stream.write(e['data'])
                dst.chmod(0o555 if e['new_mode']=='100755' else 0o444)
            return
        if relative:dst.mkdir()
        expanded.append(relative or '.')
        children={p.name:p for p in src.iterdir()} if src.is_dir() else {}
        for name in sorted(set(children)|set(node)):
            if name=='FROZEN_SOURCE.json' and not relative:continue
            rel=str(PurePosixPath(relative)/name)
            if name in node:materialize(rel,node[name])
            else:
                os.symlink(str(children[name]),attempt/rel,target_is_directory=children[name].is_dir())
                linked.append(rel)

    materialize('',tree)
    delta=[{k:v for k,v in e.items() if k!='data'} for e in entries]
    metadata=dict(schema='immutable_frozen_runtime_git_delta_overlay_v1',commit=commit,
        base=dict(path=str(base),commit=base_commit,frozen_source_sha256=sha_bytes(base_raw),
                  archive_sha256=base_meta.get('archive_sha256')),
        delta=delta,delta_sha256=fingerprint(delta),created_unix=time.time(),
        copied_regular_files=sum(e['new_mode'] in ('100644','100755') for e in entries),
        copied_bytes=sum(e['bytes'] for e in entries),symlink_count=len(linked),expanded_directories=expanded,
        unchanged_paths='absolute symlinks to trusted immutable frozen base; base must remain frozen',
        source_bytes='only exact committed Git blobs; dirty working tree ignored',
        scope='runtime scientific overlay of the frozen base, not a new full historical-resource Git archive',
        recommended_runtime_environment=dict(PYTHONDONTWRITEBYTECODE='1'))
    metadata['sha256']=fingerprint(metadata)
    (attempt/'FROZEN_SOURCE.json').write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')
    (attempt/'FROZEN_SOURCE.json').chmod(0o444)
    # Chmod only newly created real directories, never follow inherited links.
    for rel in sorted(expanded,key=lambda p:len(PurePosixPath(p).parts),reverse=True):
        (attempt if rel=='.' else attempt/rel).chmod(0o555)
    os.rename(attempt,output)
    return dict(output=str(output),commit=commit,base=str(base),delta_sha256=metadata['delta_sha256'],
                copied_regular_files=metadata['copied_regular_files'],copied_bytes=metadata['copied_bytes'],
                expanded_directories=expanded,symlink_count=len(linked))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--repo',default=str(Path.cwd()))
    parser.add_argument('--base',required=True);parser.add_argument('--commit',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(freeze_overlay(args.repo,args.base,args.commit,args.output),indent=2))


if __name__=='__main__':main()
