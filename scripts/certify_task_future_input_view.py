#!/usr/bin/env python3
"""Add missing tokenizer provenance in a NEW read-only view, never edit old records."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from transformers import AutoTokenizer
from navsim.planning.training.dataset import load_feature_target_from_pickle
from navsim.planning.training.input_only_cache import build_input_only_cache_record
from build_task_future_lite_inputs import tokenizer_manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','output','tokenizer'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--sample-count',type=int,default=32)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(1)
    os.environ['PLANREG_PROMPT_VERSION']='single_front_v1p1'
    source=a.source.resolve();path=source/'planreg_input_only_manifest.json'
    raw=path.read_bytes();manifest=json.loads(raw)
    assert manifest['protocol_version']=='task_future_lite' and manifest['schema_version']==2
    tokenizer=AutoTokenizer.from_pretrained(str(a.tokenizer),trust_remote_code=True,use_fast=False,local_files_only=True)
    sampled=random.Random(0).sample(manifest['rows'],min(a.sample_count,len(manifest['rows'])))
    for row in sampled:
        directory=source/row['log']/row['token']
        record_path=directory/'planreg_input_only.gz'
        assert hashlib.sha256(record_path.read_bytes()).hexdigest()==row['record_sha256']
        record=load_feature_target_from_pickle(record_path)
        rebuilt=build_input_only_cache_record(
            load_feature_target_from_pickle(directory/'internvl_feature.gz'),
            load_feature_target_from_pickle(directory/'trajectory_target_task_future_lite_v1.gz'),tokenizer=tokenizer)
        for key in ('input_ids','attention_mask','prompt_contract_hash'):
            assert torch.equal(record['features'][key],rebuilt['features'][key]),(row['token'],key)
    manifest.update(tokenizer_manifest(tokenizer,a.tokenizer))
    manifest.update(source_records_root=str(source),source_records_manifest_sha256=hashlib.sha256(raw).hexdigest(),
                    certification_scope='All immutable records reused by log-directory symlink; sampled prompt retokenization parity, not a claim of exhaustive retokenization',
                    retokenization_audit=dict(seed=0,sample_count=len(sampled),all_equal=True,tokens=[x['token'] for x in sampled]))
    a.output.mkdir(parents=True)
    logs=sorted({row['log'] for row in manifest['rows']})
    for log in logs:(a.output/log).symlink_to(source/log,target_is_directory=True)
    manifest['log_count']=len(logs)
    destination=a.output/'planreg_input_only_manifest.json'
    destination.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output),record_count=manifest['record_count'],
                         tokenizer_vocab_sha256=manifest['tokenizer_vocab_sha256'],
                         sample_count=len(sampled),all_equal=True,
                         manifest_sha256=hashlib.sha256(destination.read_bytes()).hexdigest()),indent=2))


if __name__=='__main__':main()
