"""Once-bound original S0 observation/context cache. No simulation or training."""
from __future__ import annotations
import argparse
from pathlib import Path
from dataclasses import asdict
import io
import json
import time
import subprocess
import os
from importlib.metadata import version

from .common import (artifact,verify_artifact,digest,sha256,read_json,write_json,
                     atomic_bytes,load_config,check_space,lock)


def prepare(cfg):
    from tools.st_world_policy.base_binding import read_binding
    root=Path(cfg['artifact_root']);check_space(cfg)
    repo=Path(__file__).resolve().parents[2]
    contracts=['tools/action_effect/collect_base_trajectories.py',
        'tools/foresight/checkpoints.py','starVLA/dataloader/foresight_dataset.py',
        'starVLA/model/framework/DDPForesight.py','starVLA/model/framework/QwenOFT.py',
        'starVLA/model/modules/action_model/GR00T_ActionHeader.py']
    extraction_contract={'schema':'action_effect_s0_context_fp32_v1',
        'code':{p:sha256(repo/p) for p in contracts},
        'packages':{p:version(p) for p in ['torch','transformers','deepspeed']},
        'visual_and_language_precision':'original_FP32_master_eval_TF32_off',
        'scene_noise':'foresight-action-v1','shape_source':'actual_bound_model'}
    path=root/'CAMPAIGN_BINDING.json'
    if path.exists():
        binding=read_json(path)
        if binding['config_hash']!=digest(cfg):raise ValueError('Use new campaign for changed configuration')
        if binding['extraction_contract']!=extraction_contract:raise ValueError('Context extraction code/runtime changed')
        verify_artifact(binding['base_binding']);verify_artifact(binding['train_manifest'])
        return binding
    base=read_binding(cfg['base_binding']);m=read_json(cfg['train_manifest'])
    if (m.get('split')!='train' or m['base_binding_hash']!=base['binding_sha256'] or
        m['data_identity']!=base['parent_identity']['data']):raise ValueError('Foreign S0/population')
    rows=sorted(m['entries'],key=lambda r:(digest(['effect_cohort_v1',cfg['seed'],r['scene_id']]),r['scene_id']))[:cfg['cohort_size']]
    if not rows:raise ValueError('No legal training observations')
    entries=[]
    for row in rows:
        role='fit' if int(digest(['effect_log_role_v1',cfg['seed'],row['source_log']])[:16],16)%10<8 else 'validation'
        entries.append({**row,'effect_role':role})
    cohort={'schema':'action_effect_observation_manifest_v1','campaign_id':cfg['campaign_id'],
            'split':'train','base_binding_hash':base['binding_sha256'],'parent_seen_population':True,
            'source_manifest':artifact(cfg['train_manifest']),'seed':cfg['seed'],
            'selection':'fixed hash before scoring; group roles by complete source_log',
            'data_root':m['root'],'entries':entries,'source_population_scenes':len(m['entries'])}
    cohort['identity']=digest(cohort);write_json(root/'COHORT.json',cohort)
    value={'schema':'action_effect_campaign_binding_v1','campaign_id':cfg['campaign_id'],
           'source_start_commit':'36493208ba4d8c25bedfcc43935a4ce4527131cc',
           'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
           'config_hash':digest(cfg),'base_binding':artifact(cfg['base_binding']),
           'extraction_contract':extraction_contract,
           'train_manifest':artifact(cfg['train_manifest']),'cohort':artifact(root/'COHORT.json'),
           'base_checkpoint':base['checkpoint'],'base_inference_contract':base['inference_contract']}
    value['identity']=digest(value);write_json(path,value);write_json(root/'RESOLVED_CONFIG.json',cfg)
    return value


def collect(cfg, *, max_samples, device):
    import torch
    from omegaconf import OmegaConf
    from starVLA.dataloader.foresight_dataset import ForesightTrainingDataset,decode_ego
    from tools.st_world_policy.base_binding import read_binding
    from tools.foresight.checkpoints import load_student,scene_noise
    from starVLA.model.modules.flow_posttraining.policy_snapshot import tensor_hash
    binding=prepare(cfg);root=Path(cfg['artifact_root']);cohort=read_json(verify_artifact(binding['cohort']))
    if not 1<=max_samples<=len(cohort['entries']):raise ValueError('Explicit valid extraction budget required')
    entries=cohort['entries'][:max_samples];out=root/'contexts';out.mkdir(exist_ok=True)
    manifest=out/f'MANIFEST_{max_samples:06d}.json'
    if manifest.exists():
        m=read_json(manifest)
        if m['binding_identity']!=binding['identity']:raise ValueError('Foreign context cache')
        for a in m['shards']:verify_artifact(a)
        return {'status':'PASS','reused':True,'manifest':str(manifest),'scenes':max_samples}
    if not device.startswith('cuda') or not torch.cuda.is_available():raise ValueError('Real allocated GPU required')
    torch.set_num_threads(2);torch.cuda.set_device(device);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    base=read_binding(cfg['base_binding']);dataset=ForesightTrainingDataset(cohort['data_root'])
    started=time.perf_counter();rows=[];shards=[]
    with lock(out/'BUILD.lock'):
        pending=[]
        for at in range(0,len(entries),cfg['feature_shard_scenes']):
            chunk=entries[at:at+cfg['feature_shard_scenes']]
            tag=digest([binding['identity'],[r['scene_id'] for r in chunk]])
            receipt=out/(tag+'.COMPLETE.json')
            if receipt.exists():
                done=read_json(receipt);verify_artifact(done['shard']);rows.extend(done['entries']);shards.append(done['shard'])
            else:pending.append((tag,chunk))
        if pending:
            model=load_student(base['run'],base['tag'],base['parent_identity'],device=device,precision='fp32')
            model.requires_grad_(False).eval();load_seconds=time.perf_counter()-started
            head=out/'S0_ACTION_HEAD.pt';head_receipt=out/'S0_ACTION_HEAD.COMPLETE.json'
            if not head_receipt.exists():
                value={'schema':'action_effect_original_action_head_v1','base_binding_hash':base['binding_sha256'],
                    'config':OmegaConf.to_container(model.action_model.full_config,resolve=True),
                    'state_dict':{k:v.detach().cpu() for k,v in model.action_model.state_dict().items()},
                    'inference_contract':base['inference_contract']}
                buf=io.BytesIO();torch.save(value,buf);atomic_bytes(head,buf.getvalue())
                write_json(head_receipt,{'schema':value['schema'],'artifact':artifact(head),
                    'base_binding_hash':base['binding_sha256'],'strict_original_master':True})
            hooks={'visual':0,'language':0}
            def hook(name):
                def count(*_):hooks[name]+=1
                return count
            hs=[model.qwen_vl_interface.model.model.visual.register_forward_hook(hook('visual')),
                model.qwen_vl_interface.model.model.language_model.register_forward_hook(hook('language'))]
            try:
                with torch.no_grad():
                    for tag,chunk in pending:
                        check_space(cfg);values=[];small=[]
                        for row in chunk:
                            # Each dependency is verified once before image/target access.
                            for dep in [row['current'],row['ego'],*row['images']]:verify_artifact(dep)
                            obs,target=dataset[row['dataset_index']]
                            if obs['token']!=row['scene_id'] or set(target)!={'ego'}:raise ValueError('Observation/target join mismatch')
                            before=hooks.copy();encoded=model.encode_current([obs]);condition=model.build_planner_condition(encoded)
                            if {k:hooks[k]-before[k] for k in hooks}!={'visual':1,'language':1}:raise ValueError('Encoder must run once per observation')
                            noise=scene_noise(obs['token'],cfg['seed'],device)
                            raw=model.action_model.predict_action(condition,initial_noise=noise);physical=decode_ego(raw.float())
                            extra=[]
                            for delta in (1,2):
                                x=model.action_model.predict_action(condition,initial_noise=scene_noise(obs['token'],cfg['seed']+delta,device))
                                extra.append(decode_ego(x.float())[0].cpu())
                            if not all(torch.isfinite(x).all() for x in [raw,physical,encoded['W'],condition]):raise FloatingPointError('Invalid S0 context/prediction')
                            record={'scene_id':row['scene_id'],'source_log':row['source_log'],'role':row['effect_role'],
                                    'observation_hash':row['observation_hash'],'base_binding_hash':base['binding_sha256'],
                                    'memory':torch.cat((encoded['W'],encoded['action_queries']),1)[0].cpu(),
                                    'ego':torch.as_tensor(obs['state']).float().cpu(),'condition':condition[0].cpu(),
                                    'W_count':encoded['W'].shape[1],'raw':raw[0].cpu(),'physical':physical[0].cpu(),
                                    'noise':noise[0].cpu(),'extra_policy_physical':extra,'target_raw':target['ego'].cpu(),
                                    'raw_hash':tensor_hash(raw),'physical_hash':tensor_hash(physical)}
                            values.append(record);small.append({'scene_id':row['scene_id'],'source_log':row['source_log'],
                                'role':row['effect_role'],'shard_offset':len(values)-1,'raw_hash':record['raw_hash'],
                                'physical_hash':record['physical_hash'],'observation_hash':row['observation_hash']})
                        path=out/(tag+'.pt');buf=io.BytesIO();torch.save(values,buf);atomic_bytes(path,buf.getvalue())
                        a=artifact(path)
                        for r in small:r['shard']=a
                        done={'schema':'action_effect_context_shard_v1','entries':small,'shard':a,
                              'base_binding_hash':base['binding_sha256'],'single_encoder_per_scene':True}
                        write_json(out/(tag+'.COMPLETE.json'),done);rows.extend(small);shards.append(a)
                        print(json.dumps({'status':'CONTEXT_SHARD_COMPLETE','scenes':len(rows),'requested':max_samples}),flush=True)
            finally:
                for h in hs:h.remove()
            del model;torch.cuda.synchronize()
        else:load_seconds=0.
        ordering={r['scene_id']:i for i,r in enumerate(entries)};rows.sort(key=lambda r:ordering[r['scene_id']])
        if len(rows)!=len(entries) or len({r['scene_id'] for r in rows})!=len(entries):raise ValueError('Incomplete context coverage')
        m={'schema':'action_effect_context_manifest_v1','binding_identity':binding['identity'],'cohort':binding['cohort'],
           'entries':rows,'shards':shards,'head_receipt':artifact(out/'S0_ACTION_HEAD.COMPLETE.json'),
           'max_samples':max_samples,'dtype':'float32','single_encoder_per_scene':True,
           'elapsed_seconds':time.perf_counter()-started,'load_seconds':load_seconds,'status':'COMPLETE'}
        m['identity']=digest(m);write_json(manifest,m)
    return {'status':'PASS','reused':False,'manifest':str(manifest),'scenes':max_samples}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--max-samples',type=int,required=True);p.add_argument('--device',required=True)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--resume',action='store_true')
    args=p.parse_args();cfg=load_config(args.config)
    if args.dry_run:print(json.dumps({'status':'NOT_RUN','config_hash':digest(cfg),'max_samples':args.max_samples}));return
    result=collect(cfg,max_samples=args.max_samples,device=args.device);print(json.dumps(result))


if __name__=='__main__':main()
