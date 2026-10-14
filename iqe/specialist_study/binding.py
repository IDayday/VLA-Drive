"""Independent immutable campaign binding; no production deployment gate."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import os
import shutil
import subprocess
from datetime import datetime, timezone
from iqe.io import atomic_json, atomic_torch, read_json, file_hash, digest, lock
from iqe.contracts import require
from iqe.data.manifests import load_scenes, save_scenes
from iqe.data.splits import validate_isolation


def bind(config):
    with lock(config.root/'campaign.lock'):
        destination = config.root/'CAMPAIGN_BINDING.json'
        if destination.exists():
            b=read_json(destination)
            require(b['config_hash']==config.identity, 'campaign configuration changed')
            for name, expected in b['artifacts'].items():
                require(file_hash(config.root/name)==expected, 'campaign artifact changed: '+name)
            return b
        source=Path(config.reuse_root)
        original=read_json(config.contract)
        from iqe.internal_baseline.binding import resolve_base_checkpoint
        selection_config=config
        preparation=config.root/'SOURCE_PREPARATION.json'
        if preparation.exists():
            prepared=read_json(preparation)
            require(prepared['campaign_config_hash']==config.identity and
                    prepared['source_binding_hash']==file_hash(source/'BASE_BINDING.json'),'source preparation identity changed')
            selection_config=SimpleNamespace(base_run=config.base_run,checkpoint=prepared['checkpoint'],root=config.root)
        selected,state=resolve_base_checkpoint(selection_config,original)
        require(not state.get('mode') in ('smoke','profile'), 'real own trained Base required')
        old=read_json(source/'BASE_BINDING.json')
        require(selected['checksum']==old['checkpoint_hash'],
                'requested Base has no matching reusable context; build its full context under a new campaign before binding')
        require(digest(read_json(source/'OFFLINE_BASE_CONTRACT.json')) == old['contract_hash'], 'source contract identity changed')
        require(file_hash(source/'BASE_HEAD.pt') == old['head_hash'], 'source action head corrupted')
        selected_manifest=source/'scenes.json' if preparation.exists() else Path(config.scene_manifest)
        require(file_hash(selected_manifest)==old['scene_manifest_hash'], 'source scene population changed')
        scenes=load_scenes(selected_manifest)
        require(all(s.split_role!='final_test' for s in scenes),'final_test cannot enter campaign')
        isolation=validate_isolation(scenes)
        partition=read_json(source/'DATA_PARTITION.json')
        require(partition['hash']==digest({k:v for k,v in partition.items() if k!='hash'}), 'source partition corrupted')
        require(partition['partition_seed']==config.partition_seed, 'source split seed mismatch')
        de=set(partition['roles']['expert_fit']['scene_ids'])
        ds_all=set(partition['roles']['scorer_fit']['scene_ids'])
        ds=set(sorted(ds_all,key=lambda x:digest([config.selection_seed,'scorer_fit',x]))[:config.scorer_fit_limit])
        require(not de&ds, 'expert/scorer fit overlap')
        selected_scenes=[s for s in scenes if s.scene_id in de|ds or s.split_role!='incremental_fit']
        save_scenes(config.root/'scenes.json', selected_scenes)
        roles={}
        for role in ('expert_fit','scorer_fit','stage_val','selector_cal','dev_report'):
            rows=[s for s in selected_scenes if s.scene_id in (de if role=='expert_fit' else ds)] if role.endswith('_fit') else [s for s in selected_scenes if s.split_role==role]
            roles[role]={'scene_ids':[s.scene_id for s in rows], 'source_logs':sorted({s.source_log_id for s in rows}),
                         'source_groups':sorted({s.source_group_id for s in rows}), 'scenes':len(rows),
                         'base_seen':{v:sum(s.s0_seen==v for s in rows) for v in ('yes','no','unknown')}}
        require(not set(roles['expert_fit']['source_logs'])&set(roles['scorer_fit']['source_logs']), 'source log leakage')
        data={'schema':'specialist_study_roles_v1','roles':roles,'source_partition_hash':file_hash(source/'DATA_PARTITION.json'),
              'selected_before_new_scores':True,'D_S_meaning':'held out from newly trained specialists; may have been seen by Base',
              'final_test_accessed':False,'isolation':isolation}
        data['hash']=digest(data)
        atomic_json(config.root/'DATA_PARTITION.json',data,immutable=True)
        for name in ('OFFLINE_BASE_CONTRACT.json','BASE_HEAD.pt','SCORE_PROTOCOL.json'):
            dest=config.root/name
            if not dest.exists(): os.link(source/name,dest)
        b=dict(old,experiment_id=config.campaign_id,config_hash=config.identity,
               source_binding_hash=file_hash(source/'BASE_BINDING.json'),
               scene_manifest_hash=file_hash(config.root/'scenes.json'),
               bound_at_utc=datetime.now(timezone.utc).isoformat(), research_only=True,
               role_counts={k:v['scenes'] for k,v in roles.items()},
               source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
        atomic_json(config.root/'BASE_BINDING.json',b,immutable=True)
        campaign={'schema':'specialist_scorer_campaign_v1','campaign_id':config.campaign_id,'config_hash':config.identity,
          'checkpoint':selected,'source_commit':b['source_commit'],'source_root':str(Path(__file__).resolve().parents[2]),
          'authorization':{'expert_runs':6,'expert_updates_each':5000,'scorer_runs':4,'scorer_updates_each':6000,
                           'navtest':False,'ACTIVE':False,'push':False},
          'artifacts':{n:file_hash(config.root/n) for n in ('BASE_BINDING.json','OFFLINE_BASE_CONTRACT.json','BASE_HEAD.pt','DATA_PARTITION.json','scenes.json','SCORE_PROTOCOL.json')}}
        atomic_json(destination,campaign,immutable=True)
        return campaign


def scenes_by_role(config):
    scenes=load_scenes(config.root/'scenes.json'); lookup={s.scene_id:s for s in scenes}
    roles=read_json(config.root/'DATA_PARTITION.json')['roles']
    return {role:[lookup[i] for i in rows['scene_ids']] for role,rows in roles.items()}
