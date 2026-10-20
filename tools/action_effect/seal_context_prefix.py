"""Seal only a complete prelocked prefix while later context shards are building.

No model runs, sampling changes, or partial population pretending to be complete.
Every referenced16-scene shard must already have its independent COMPLETE.
"""
from pathlib import Path
import argparse,json
from .common import load_config,read_json,verify_artifact,verify_identity,digest,artifact,write_json
from .collect_base_trajectories import prepare,validate_shard,validate_context_manifest


def seal(cfg,*,max_samples):
    if not 1<=max_samples<=cfg['cohort_size'] or max_samples%cfg['feature_shard_scenes']:
        raise ValueError('Prefix must end at a complete shard boundary')
    binding=prepare(cfg);cohort=verify_identity(read_json(verify_artifact(binding['cohort'])))
    entries=cohort['entries'][:max_samples];out=Path(cfg['artifact_root'])/'contexts'
    path=out/f'MANIFEST_{max_samples:06d}.json'
    if path.exists():validate_context_manifest(read_json(path),binding,entries);return {'status':'PASS','manifest':str(path),'reused':True}
    rows=[];shards=[]
    for at in range(0,max_samples,cfg['feature_shard_scenes']):
        chunk=entries[at:at+cfg['feature_shard_scenes']]
        tag=digest([binding['identity'],[r['scene_id'] for r in chunk]])
        p=out/(tag+'.COMPLETE.json')
        if not p.is_file():return {'status':'BLOCKED','reason':'awaiting_complete_shard','requested_scenes':max_samples,'complete_scenes':len(rows)}
        r=validate_shard(read_json(p),binding,chunk);rows.extend(r['entries']);shards.append(r['shard'])
    m={'schema':'action_effect_context_manifest_v1','binding_identity':binding['identity'],'cohort':binding['cohort'],
       'entries':rows,'shards':shards,'head_receipt':artifact(out/'S0_ACTION_HEAD.COMPLETE.json'),
       'max_samples':max_samples,'dtype':'float32','single_encoder_per_scene':True,
       'elapsed_seconds':0.,'load_seconds':0.,'status':'COMPLETE',
       'construction':'sealed_checked_complete_prefix; no new inference'}
    m['identity']=digest(m);validate_context_manifest(m,binding,entries);write_json(path,m)
    return {'status':'PASS','manifest':str(path),'reused':False,'scenes':max_samples}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--max-samples',required=True,type=int)
    a=p.parse_args();r=seal(load_config(a.config),max_samples=a.max_samples);print(json.dumps(r));return 0 if r['status']=='PASS' else 2


if __name__=='__main__':raise SystemExit(main())
