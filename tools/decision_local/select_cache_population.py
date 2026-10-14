"""Immutable, explicitly bounded subset of already committed compact caches."""
import argparse
from pathlib import Path
import sys
import torch
from safetensors.torch import load_file,save_file
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,digest,identity,read,source_sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('features','labels','index','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();index=read(a.index);tokens=[r['token'] for r in index]
    if len(tokens)!=len(set(tokens)):raise ValueError('Duplicate subset scene')
    fmeta,lmeta=read(a.features/'identity.json'),read(a.labels/'identity.json')
    if lmeta['anchor_cache_identity']!=fmeta['identity']:raise ValueError('Incompatible parent caches')
    actual={r['token']:r['log'] for r in fmeta['index']}
    if any(actual.get(r['token'])!=r['log'] for r in index):raise ValueError('Subset not in shared original population')
    locations={}
    for p in (a.features/'records').glob('*.json'):
        if not (a.labels/'records'/p.name).exists():continue
        f,l=read(p),read(a.labels/'records'/p.name)
        if f['tokens']!=l['tokens']:raise ValueError('Different scene order')
        for at,t in enumerate(f['tokens']):locations[t]=(p.stem,at)
    if not set(tokens)<=set(locations):raise ValueError('Selected fixed population is not yet completely cached')
    features={k:v for k,v in fmeta.items() if k!='identity'}
    features.update(index=index,parent_cache_identity=fmeta['identity'],population_view='explicit subset, not full train/dev',
        view_source_sha=source_sha(ROOT),view_tool_sha256=digest(__file__),chunk_size=128)
    fid=identity(features);features['identity']=fid
    labels={k:v for k,v in lmeta.items() if k!='identity'}
    labels.update(anchor_cache_identity=fid,parent_label_identity=lmeta['identity']);lid=identity(labels);labels['identity']=lid
    if a.output.exists():raise FileExistsError('Do not overwrite an immutable subset')
    for name,meta in [('features',features),('labels',labels)]:
        root=a.output/name;root.mkdir(parents=True)
        for folder in ('records','chunks'):(root/folder).mkdir()
        atomic_json(root/'identity.json',meta);atomic_json(root/'index.json',index)
    chunk_cache={};hashes={}
    for start in range(0,len(tokens),128):
        members=tokens[start:start+128];acc={'features':[],'labels':[]}
        for token in members:
            chunk,at=locations[token]
            if chunk not in chunk_cache:
                chunk_cache={}
                for name,root in [('features',a.features),('labels',a.labels)]:
                    path=root/'chunks'/(chunk+'.safetensors');meta=read(root/'records'/(chunk+'.json'))
                    key=(str(root),chunk)
                    if key not in hashes:
                        if digest(path)!=meta['sha256']:raise ValueError('Parent chunk changed')
                        hashes[key]=meta['sha256']
                    chunk_cache.setdefault(chunk,{})[name]=load_file(str(path))
            for name in acc:acc[name].append({k:v[at] for k,v in chunk_cache[chunk][name].items()})
        for name,ident in [('features',fid),('labels',lid)]:
            tensors={k:torch.stack([r[k] for r in acc[name]]).contiguous() for k in acc[name][0]}
            path=a.output/name/'chunks'/('%06d.safetensors'%(start//128));save_file(tensors,str(path),metadata={'identity':ident})
            atomic_json(a.output/name/'records'/('%06d.json'%(start//128)),{'identity':ident,'tokens':members,'sha256':digest(path),'bytes':path.stat().st_size})
    for name,ident in [('features',fid),('labels',lid)]:atomic_json(a.output/name/'COMPLETE.json',{'identity':ident,'scenes':len(tokens),'full_original_population':False,'subset_index_sha256':digest(a.index)})


if __name__=='__main__':main()
