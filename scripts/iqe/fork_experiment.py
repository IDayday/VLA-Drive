#!/usr/bin/env python
"""Bind an ablation to the same locked S0, role manifests and immutable Base bank.

Small metadata is copied; immutable cache objects use hardlinks with copy fallback.
Large source weights remain addressed by their verified original checkpoint path.
"""
import argparse
from pathlib import Path
import shutil
import os
from iqe.config import load_config
from iqe.io import read_json,atomic_json,file_hash,digest
from iqe.contracts import require
from iqe.registry import ExpertRegistry


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--source-config',required=True);p.add_argument('--config',required=True)
    a=p.parse_args();source=load_config(a.source_config);dest=load_config(a.config)
    src,dst=Path(source['output_root']),Path(dest['output_root'])
    require(src.resolve()!=dst.resolve() and not dst.exists(),'fork requires a new isolated output directory')
    contract=read_json(src/'LOCKED_S0.json');require(file_hash(contract['query_checkpoint'])==contract['query_checkpoint_hash'],'locked S0 changed')
    require(source['s0']['contract']==dest['s0']['contract'],'ablation must inherit the same framework contract')
    dst.mkdir(parents=True)
    copied=[]
    for name in ('LOCKED_S0.json','scenes.json','OFFICIAL_SCENES.json','base_exposure.json','admitted_sources.json','candidate_index.json','score_index.json','protocol_audit/SCORE_CONTEXT_EVIDENCE.json','rounds/round_000/candidates.json','rounds/round_000/scored.json'):
        s=src/name
        if s.exists():
            out=dst/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s,out);copied.append({'path':name,'hash':file_hash(s)})
    def link_or_copy(s,d):
        try:os.link(s,d)
        except OSError:shutil.copy2(s,d)
    for name in ('features','bank','target_scores','candidate_lookup','score_lookup'):
        if (src/name).exists():shutil.copytree(src/name,dst/name,copy_function=link_or_copy,ignore=shutil.ignore_patterns('*.lock','.*'))
    original=ExpertRegistry(src/'registry.json',digest(contract));base=original.latest(original.read())['expert_0']
    ExpertRegistry(dst/'registry.json',digest(contract)).append(base)
    atomic_json(dst/'FORK.json',{'source':str(src.resolve()),'source_config':digest(source),'destination_config':digest(dest),'locked_S0_hash':contract['query_checkpoint_hash'],'copied':copied,
        'same_role_population':True,'new_optimizer_states':True,'base_training_repeated':False},immutable=True)
    print(__import__('json').dumps({'status':'COMPLETE','output':str(dst)}))


if __name__=='__main__':main()
