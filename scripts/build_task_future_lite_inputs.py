#!/usr/bin/env python3
"""New-root input-only cache with Scene-derived logged future poses; no labels/model outputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def tokenizer_manifest(tokenizer, path):
    from navsim.agents.EpisodeDrive.formal_initialization import canonical_sha256
    return dict(tokenizer_path=str(Path(path).resolve()),
                tokenizer_vocab_sha256=canonical_sha256({k:int(v) for k,v in tokenizer.get_vocab().items()}))


def build_log(task):
    log,args=task
    import torch
    torch.set_num_threads(1)
    from transformers import AutoTokenizer
    from navsim.common.dataclasses import Scene, SensorConfig
    from navsim.agents.EpisodeDrive.layers.world_model.logged_future_pose import build_logged_future_pose_metadata
    from navsim.planning.training.dataset import load_feature_target_from_pickle, dump_feature_target_to_pickle
    from navsim.planning.training.input_only_cache import build_input_only_cache_record
    global _TOKENIZER
    if '_TOKENIZER' not in globals():
        _TOKENIZER=AutoTokenizer.from_pretrained(args['tokenizer'],trust_remote_code=True,use_fast=False,local_files_only=True)
    raw_path=Path(args['raw_logs'])/(log+'.pkl')
    with raw_path.open('rb') as stream:
        raw=pickle.load(stream)
    raw_hash=hashlib.sha256(raw_path.read_bytes()).hexdigest()
    token_indices={frame['token']:i for i,frame in enumerate(raw)}
    rows=[]
    for target_path in sorted((Path(args['cache_root'])/log).glob('*/trajectory_target_planreg_wm_v1.gz')):
        token=target_path.parent.name
        # Match the original 103,288-sample input-only dataset, not the 40,493
        # extra target-only directories that were never eligible for training.
        if not (target_path.parent/'planreg_input_only.gz').is_file():
            continue
        if args.get('tokens') is not None and token not in args['tokens']:
            continue
        idx=token_indices.get(token)
        if idx is None or idx<3 or idx+10>=len(raw):
            raise ValueError(f'Missing complete raw scene for cached train token {log}/{token}')
        source=raw[idx-3:idx+11]
        scene=Scene.from_scene_dict_list(source,Path(args['sensor_root']),4,10,
                                         SensorConfig.build_no_sensors(),load_image_path=True)
        features=load_feature_target_from_pickle(target_path.parent/'internvl_feature.gz')
        targets=load_feature_target_from_pickle(target_path)
        targets.update(build_logged_future_pose_metadata(scene))
        record=build_input_only_cache_record(features,targets,tokenizer=_TOKENIZER)
        out=Path(args['output_root'])/log/token
        out.mkdir(parents=True,exist_ok=True)
        destination=out/'planreg_input_only.gz'
        if destination.exists():
            raise FileExistsError(destination)
        dump_feature_target_to_pickle(destination,record)
        # CacheOnlyDataset discovers target/feature names before consolidated load.
        dump_feature_target_to_pickle(out/'internvl_feature.gz',features)
        dump_feature_target_to_pickle(out/'trajectory_target_task_future_lite_v1.gz',targets)
        rows.append(dict(log=log,token=token,raw_log_sha256=raw_hash,
                         record_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                         future_valid=targets['future_valid_mask'].tolist()))
    return rows


def main():
    from omegaconf import OmegaConf
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('cache-root','raw-logs','sensor-root','output-root','tokenizer','train-config'):
        parser.add_argument('--'+field,type=Path,required=True)
    parser.add_argument('--tokens',type=Path,help='Optional train-only token JSON for a bounded smoke')
    parser.add_argument('--jobs',type=int,default=16)
    args=parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError('New root required; never overwrite V1.1 caches: '+str(args.output_root))
    cfg=OmegaConf.load(args.train_config)
    train,val=set(cfg.train_logs),set(cfg.val_logs)
    if train & val:
        raise ValueError('Source train/val log split must be disjoint before final-fit union')
    logs=sorted(log for log in train|val if (args.cache_root/log).is_dir())
    params={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    params['tokens']=None if args.tokens is None else set(json.loads(args.tokens.read_text()))
    if params['tokens'] is not None:
        logs=[log for log in logs if any((args.cache_root/log/token).is_dir() for token in params['tokens'])]
    args.output_root.mkdir(parents=True)
    os.environ['PLANREG_PROMPT_VERSION']='single_front_v1p1'
    rows=[]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for batch in pool.map(build_log,[(log,params) for log in logs]):
            rows.extend(batch)
            print(f'completed_records={len(rows)}',flush=True)
    if params['tokens'] is not None and {row['token'] for row in rows} != params['tokens']:
        raise ValueError('Requested tokens not all found in trainval: no Navtest fallback')
    if params['tokens'] is None and len(rows) != 103288:
        raise ValueError(f'Formal dataset must retain exactly 103288 eligible tokens, got {len(rows)}')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(args.tokenizer),trust_remote_code=True,use_fast=False,local_files_only=True)
    manifest=dict(schema_version=2,cache_mode='input_only',protocol_version='task_future_lite',
        prompt_version='single_front_v1p1',record_count=len(rows),front_camera_only=True,sensor_camera_count=1,
        required_source_files_complete=True,  # every required current/feature/target file was read above
        source_completeness_definition='Required input/target files; explicitly masked future images are permitted, never imputed supervision',
        scenes_all_future_valid=sum(all(row['future_valid']) for row in rows),
        scenes_with_masked_future=sum(not all(row['future_valid']) for row in rows),
        train_val_overlap=0,train_source_config_sha256=hashlib.sha256(args.train_config.read_bytes()).hexdigest(),
        cached_fields=['current_image_path','future_image_paths','future_valid_mask','logged_future_poses',
            'logged_future_pose_valid','physical_map_name','gt_trajectory','long_trajectory','ego_status',
            'navigation_command','input_ids','attention_mask','image_original_size','tile_metadata'],
        rows=rows,**tokenizer_manifest(tokenizer,args.tokenizer))
    (args.output_root/'planreg_input_only_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='rows'},indent=2))


if __name__=='__main__':
    main()
