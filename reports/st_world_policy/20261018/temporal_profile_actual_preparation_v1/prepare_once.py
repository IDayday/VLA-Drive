#!/usr/bin/env python3
"""T0 global32 metadata binding of the existing two-CLI supervisor; no GPU or retry."""
import time
START=time.monotonic()
with open('/proc/self/io') as f:
    BOOT_IO={k:int(v) for k,v in (line.split(':') for line in f)}
import hashlib
import os
from pathlib import Path
import types

HERE=Path(__file__).resolve().parent
DONOR=Path('/mnt/project/VLA-Drive-st-temporal-profile-runtime-v1-20261018/reports/st_world_policy/20261018/temporal_T3_cpu_preparation_v2/prepare_once.py')
DONOR_SHA='80da95a21f7918b0524d1d71045da13d43c1828747f046e62d9bdb26e418bd01'
RUNTIME=Path('/mnt/project/VLA-Drive-st-temporal-profile-runtime-v1-20261018')
COMMIT='38f86a15a92171e583bb5019cce89a63fb7a1d43'
ROOT=Path('/mnt/project/st-world-policy-20261016/profile/temporal_global32_qualification_v1/T0_capacity4/metadata')
PLAN_SHA='9217f72273de0eeb80c2d872e061f69944a404fe1ff066a8f96e62ee0d79fbcc'
LIMITS=dict(wall_seconds=1500,cleanup_seconds=40,read_bytes=64<<30,rss_bytes=4<<30)
OUTPUT_LIMIT=8<<20
OUTPUTS=('SOURCE_MIGRATION.json','configured/RECIPE.json','configured/PROVIDER.json',
         'configured/PROVIDER_CHANNEL_SCOPE.json','configured/TARGET_CONTRACT.json',
         'configured/training_config.json','configured/PREPARATION_COMPLETE.json')


def donor():
    raw=DONOR.read_bytes()
    if len(raw)>32<<10 or hashlib.sha256(raw).hexdigest()!=DONOR_SHA:
        raise ValueError('Reviewed T3 binding changed')
    module=types.ModuleType('reviewed_T3_CPU_binding');module.__file__=str(DONOR)
    exec(compile(raw,str(DONOR),'exec'),module.__dict__)
    return module


def commands(plan):
    p=plan['inputs'];path=lambda name:p[name]['path']
    return [('tools.st_world_policy.qualify_indexed_provider',[
        'prepare-profile-migration','--parent-descriptor',path('parent_descriptor'),
        '--parent-runtime',plan['parent_runtime'],'--output',str(ROOT/'SOURCE_MIGRATION.json')]),
      ('tools.st_world_policy.temporal_formal',[
        '--full-profile-descriptor',path('full_profile_descriptor'),
        '--source-migration',str(ROOT/'SOURCE_MIGRATION.json'),'--decision',path('decision'),
        '--pca-fit-cache-root',plan['pca_fit_cache_root'],
        '--pca-fit-cache-index',str(Path(path('pca_fit_index_complete')).parent),
        '--world-size','4','--micro-batch','1','--scope','PROFILE','--max-seconds','3600',
        '--cell','T0','--profile-mode','none','--output',str(ROOT/'configured'),
        '--training-output-root',plan['training_output_root']])]


def bound_source(d):
    text=d.bind_source(d.original_bytes())
    changes=[
      ('st_T3_cpu_preparation_admission_v1','st_T0_global32_cpu_preparation_admission_v1',1),
      ('APPROVED_T3_METADATA_PREPARATION_ONLY','APPROVED_T0_GLOBAL32_METADATA_PREPARATION_ONLY',1),
      ('st_T3_cpu_preparation_plan_v1','st_T0_global32_cpu_preparation_plan_v1',1),
      ("(None,'','/root/st_indexprep_boundary_fix')", "(None,'','/root/st_indexprep_boundary_fix','/root/temporal_profile_prepare','/root')",1),
      ("HERE/'PLAN.json'","HERE/'CPU_PLAN.json'",2),
      ('[2,3]','[12,13]',3),('{2,3}','{12,13}',2),('(2,3)','(12,13)',1),
      ('else 340','else 1540',1),
      ("history=reads.document(plan['inputs']['history_complete'])\n    if history.get('status')!='COMPLETE' or history.get('scenes')!=32:raise ValueError('Exact32 history has not completed')",
       "current=reads.document(plan['inputs']['current_index_complete'])\n    if current.get('status')!='COMPLETE' or current.get('scene_count')!=101592:raise ValueError('Full TRAIN current metadata has not completed')",1),
    ]
    for old,new,count in changes:
        if text.count(old)!=count:raise ValueError('Existing supervisor binding changed: '+old)
        text=text.replace(old,new)
    return text


def controller():
    d=donor()
    m=types.ModuleType('reviewed_T0_global32_metadata');m.__file__=str(HERE/'prepare_once.py')
    exec(compile(bound_source(d),str(d.ORIGINAL),'exec'),m.__dict__)
    m.__dict__.update(START=START,BOOT_IO=BOOT_IO,HERE=HERE,ROOT=ROOT,CONTROL=ROOT/'controls',
        RUNTIME=RUNTIME,COMMIT=COMMIT,PLAN_SHA=PLAN_SHA,LIMITS=LIMITS,OUTPUT_LIMIT=OUTPUT_LIMIT,
        OUTPUTS=OUTPUTS,commands=commands,REVIEWED_HOME=d.ORIGINAL.parent,
        BOOTSTRAP=d.ORIGINAL.parent.parent/'future_index_sequential_control_v1/prepare_once.py',
        composition_inventory=d.composition_inventory,output_path=d.output_path)
    return m


if __name__=='__main__':raise SystemExit(controller().main())
