"""Freeze F4 feature scales from all512 shared TRAIN scenes, without inference."""
import argparse
import os
from pathlib import Path
import time

import torch

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_sourceflow.contracts import file_sha,fingerprint
from starVLA.rl.qwen_flow_exploration.protocol import read_protocol,require_dispatch,require_budget_lock
from starVLA.rl.qwen_flow_exploration.feedback import process
from scripts.analysis.qwen_flow_exploration.feedback_search_bank import indexed_paths


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--protocol',required=True)
    p.add_argument('--bank-index',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Explicit CPU-only scale fitting required')
    require_dispatch();protocol=read_protocol(a.protocol);require_budget_lock(protocol,phase='DIAGNOSTIC_LOCKED')
    import json
    output=Path(a.output)
    if output.exists():raise FileExistsError('Freeze scale once; use a new study ID for any recalibration')
    paths=indexed_paths(json.loads(Path(a.bank_index).read_text()),'TRAIN')
    expected=set(protocol['diagnostic']['TRAIN'])
    if len(paths)!=512 or len(expected)!=512:raise ValueError('F4 normalization requires the fixed512 TRAIN population')
    started=time.monotonic();seen=set();hashes={};features=[]
    for path in paths:
        seal=json.loads(path.with_suffix('.json').read_text());digest=file_sha(path)
        if seal['status']!='COMPLETE' or seal['protocol_sha256']!=protocol['sha256'] or seal['payload_sha256']!=digest:
            raise ValueError('Changed/uncommitted common bank payload')
        bank=torch.load(path,map_location='cpu',weights_only=False);token=bank['scene_token']
        if bank['split']!='TRAIN' or token not in expected or token in seen or bank['checkpoint_id']!=protocol['base_model_sha256']:
            raise ValueError('Wrong population or base in novelty calibration')
        seen.add(token);hashes[token]=digest
        if len(bank['candidate_records'])!=32:raise ValueError('Exactly32 real scored candidates per TRAIN scene required')
        for row in bank['candidate_records']:
            value=process.process_summary(row['feedback'])['novelty_features']
            if value is None:raise ValueError('Missing novelty feedback is not a zero feature')
            features.append(value)
    if seen!=expected:raise ValueError('Incomplete fixed TRAIN calibration population')
    scale=process.NoveltyScale.fit_train(features,split='TRAIN')
    result=dict(schema='feedback_frozen_TRAIN_novelty_scale_v1',status='COMPLETE',**scale.identity(),
        common_protocol_sha256=protocol['sha256'],base_model_sha256=protocol['base_model_sha256'],
        training_scene_count=512,candidate_count=len(features),scene_set_sha256=fingerprint(sorted(seen)),
        bank_index_sha256=file_sha(a.bank_index),source_bank_manifest_sha256=fingerprint(hashes),
        source_file_sha256=file_sha(__file__),feature_function_sha256=file_sha(process.__file__),
        actual_new_model_forwards=0,actual_new_scores=0,historical_shared_bank_candidate_scores=len(features),
        wall_seconds=time.monotonic()-started,no_DEV_or_NAVTEST=True)
    result['sha256']=fingerprint(result);atomic_json(output,result);print(result)


if __name__=='__main__':main()
