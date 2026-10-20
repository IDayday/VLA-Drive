"""Explicit resource-only namespace fork; preserve every original cache/result.

The immutable scientific Base, cohort, extraction function, dtype, sampler and
training recipe must be identical. Complete payloads are hardlinked only after
checksum validation; receipts are re-bound, not forged as original provenance.
"""
from pathlib import Path
import argparse,json,os
from .common import load_config,read_json,verify_identity,verify_artifact,artifact,digest,write_json
from .collect_base_trajectories import prepare,validate_shard


def migrate(old_cfg,new_cfg):
    scientific=lambda c:{k:v for k,v in c.items() if k not in ('artifact_root','resources')}
    if scientific(old_cfg)!=scientific(new_cfg):raise ValueError('Only explicit resource namespace changes are compatible')
    old=verify_identity(read_json(Path(old_cfg['artifact_root'])/'CAMPAIGN_BINDING.json'))
    if old['config_hash']!=digest(old_cfg) or old_cfg['artifact_root']==new_cfg['artifact_root']:
        raise ValueError('Source config must match its immutable binding; use a separate execution root')
    cohort=verify_identity(read_json(verify_artifact(old['cohort'])))
    new=prepare(new_cfg)
    if (old['base_checkpoint']!=new['base_checkpoint'] or old['base_checkpoint_binding_hash']!=new['base_checkpoint_binding_hash'] or
        old['extraction_contract']!=new['extraction_contract'] or old['cohort']['sha256']!=new['cohort']['sha256']):
        raise ValueError('Scientific/numeric/cohort identity changed during resource fork')
    src=Path(old_cfg['artifact_root'])/'contexts';dst=Path(new_cfg['artifact_root'])/'contexts';dst.mkdir(exist_ok=True)
    hr=read_json(src/'S0_ACTION_HEAD.COMPLETE.json');verify_artifact(hr['artifact'])
    write_json(dst/'S0_ACTION_HEAD.COMPLETE.json',hr)
    rows=[]
    for at in range(0,len(cohort['entries']),old_cfg['feature_shard_scenes']):
        chunk=cohort['entries'][at:at+old_cfg['feature_shard_scenes']]
        ot=digest([old['identity'],[r['scene_id'] for r in chunk]])
        receipt=src/(ot+'.COMPLETE.json')
        if not receipt.exists():continue
        r=validate_shard(read_json(receipt),old,chunk)
        nt=digest([new['identity'],[r['scene_id'] for r in chunk]])
        path=dst/(nt+'.pt')
        if not path.exists():os.link(r['shard']['path'],path)
        a=artifact(path)
        if a['sha256']!=r['shard']['sha256']:raise ValueError('Migration changed payload bytes')
        entries=[{**x,'shard':a} for x in r['entries']]
        nr={'schema':r['schema'],'entries':entries,'shard':a,'binding_identity':new['identity'],
            'base_binding_hash':r['base_binding_hash'],'single_encoder_per_scene':True}
        nr['identity']=digest(nr);validate_shard(nr,new,chunk);write_json(dst/(nt+'.COMPLETE.json'),nr)
        rows.append({'old_receipt':artifact(receipt),'new_receipt':artifact(dst/(nt+'.COMPLETE.json')),
                     'same_payload_sha256':a['sha256'],'scene_ids':[x['scene_id'] for x in chunk]})
    # Immutable content-addressed candidate results already support population
    # reuse. Reuse that store explicitly; new calls remain individually receipted.
    source=(Path(old_cfg['artifact_root'])/'g0/candidate_cache').resolve();link=Path(new_cfg['artifact_root'])/'g0/candidate_cache'
    if not source.is_dir():raise ValueError('Missing source immutable candidate cache')
    link.parent.mkdir(exist_ok=True)
    if os.path.lexists(link):
        if not link.is_symlink() or link.resolve()!=source:raise ValueError('Wrong/dangling shared candidate cache link')
    else:link.symlink_to(source,target_is_directory=True)
    if not link.is_dir() or link.resolve()!=source:raise ValueError('Wrong shared immutable candidate cache')
    result={'schema':'action_effect_resource_namespace_migration_v1','status':'PASS',
        'old_binding':old['identity'],'new_binding':new['identity'],'old_config':old_cfg,'new_config':new_cfg,
        'original_Base_unchanged':True,'scientific_contract_unchanged':True,'new_inference_calls':0,
        'new_simulation_calls':0,'new_model_updates':0,'completed_shards_reused':len(rows),
        'scenes_reused':sum(len(x['scene_ids']) for x in rows),'payloads':rows,
        'native_cache':str(source),'native_reuse':'only exact request identities; each native receipt/checksum still verified',
        'old_artifact_bytes_modified':False,'note':'hardlink may update inode ctime/linkcount; immutable tensor bytes, old receipts and checkpoints are unchanged'}
    result['identity']=digest(result);write_json(Path(new_cfg['artifact_root'])/'EXECUTION_NAMESPACE_MIGRATION.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--old-config',required=True);p.add_argument('--config',required=True)
    a=p.parse_args();r=migrate(load_config(a.old_config),load_config(a.config));print(json.dumps({k:v for k,v in r.items() if k not in ('payloads','old_config','new_config')}))


if __name__=='__main__':main()
