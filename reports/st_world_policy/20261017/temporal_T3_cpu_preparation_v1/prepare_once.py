#!/usr/bin/env python3
"""Fixed T3 binding of the reviewed two-CLI metadata supervisor; no dispatch."""
import time
START = time.monotonic()
with open('/proc/self/io') as _io:
    BOOT_IO = {k:int(v) for k,v in (line.split(':') for line in _io)}
import hashlib
import os
from pathlib import Path
import re
import stat
import types

HERE = Path(__file__).resolve().parent
ORIGINAL = Path('/mnt/project/VLA-Drive-st-full-population-provider-preparation-20261017/reports/st_world_policy/20261017/full_population_provider_preparation_v1/prepare_once.py')
ORIGINAL_SHA = '9ef4b038705613d38fee96a26ff5798d21ad7c4daa50009243685bd6a940ce24'
RUNTIME = Path('/mnt/project/VLA-Drive-st-temporal-combination-20261017')
ROOT = Path('/mnt/project/st-world-policy-20261016/audits/temporal_T3_indexed32_preparation_v1')
PLAN_SHA = '83ca35fd1a51e759f896aba90e8f89afd66c4f84a531c89fb84273a571a108a7'
LIMITS = dict(wall_seconds=300, cleanup_seconds=40, read_bytes=4<<30, rss_bytes=2<<30)
OUTPUT_LIMIT = 8<<20
OUTPUTS = ('SOURCE_MIGRATION.json', 'configured/T2_REFERENCE.json', 'configured/T3_RECIPE.json',
           'configured/PROVIDER.json', 'configured/PROVIDER_CHANNEL_SCOPE.json',
           'configured/training_config.json', 'configured/PREPARATION_COMPLETE.json')


def original_bytes():
    fd = os.open(ORIGINAL, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as f:
        before = os.fstat(f.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > 64<<10:
            raise ValueError('Small regular reviewed runner required')
        raw = f.read((64<<10)+1); after = os.fstat(f.fileno())
    version = lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if version(before)!=version(after) or len(raw)!=before.st_size or hashlib.sha256(raw).hexdigest()!=ORIGINAL_SHA:
        raise ValueError('Reviewed runner changed')
    return raw


def publication_target(root):
    """Only the existing CLI's one local immutable publication link is allowed."""
    p = root/'configured'
    if not p.is_symlink():
        if p.exists(): raise ValueError('Expected original atomic composition publication')
        return None
    target = os.readlink(p)
    if not re.fullmatch(r'\.configured\.staging-[a-zA-Z0-9_-]+', target):
        raise ValueError('Foreign composition publication target')
    stage = root/target
    if stage.is_symlink() or not stage.is_dir() or stage.resolve().parent != root.resolve():
        raise ValueError('Publication must name its actual local immutable version')
    return stage


def composition_inventory(metadata, root):
    """Reuse original strict inventories for all files, count the sole link once."""
    if not root.exists(): return {'charged_bytes':0}
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or root.resolve()!=root:
        raise ValueError('Plain canonical preparation root required')
    amount=max(info.st_size,info.st_blocks*512); entries=1
    target=publication_target(root)
    children=list(root.iterdir())
    if len(children)>64: raise ValueError('Composition control entry bound')
    for child in children:
        if child.name=='configured' and target is not None:
            s=child.lstat(); amount+=max(s.st_size,s.st_blocks*512); entries+=1
        else:
            inv=metadata.tree_inventory(child,max_depth=2,max_entries=64)
            amount+=inv['charged_bytes'];entries+=len(inv['entries'])
        if entries>64:raise ValueError('Composition control entry bound')
    if publication_target(root)!=target:raise ValueError('Publication changed during census')
    return {'charged_bytes':amount}


def output_path(path):
    if path.parent.name=='configured':
        target=publication_target(path.parent.parent)
        if target is None:raise ValueError('Composition has not published its immutable result')
        return target/path.name
    return path


def commands(plan):
    p=plan['inputs'];path=lambda key:p[key]['path']
    return [('tools.st_world_policy.qualify_indexed_provider',[
        'prepare-profile-migration','--parent-descriptor',path('parent_descriptor'),
        '--parent-runtime',plan['parent_runtime'],'--output',str(ROOT/'SOURCE_MIGRATION.json')]),
      ('tools.st_world_policy.temporal_composition',[
        '--cell','T3','--parent-descriptor',path('parent_descriptor'),
        '--reference-config',path('reference_config'),'--data-manifest',path('selected'),
        '--history-complete',path('history_complete'),'--source-migration',str(ROOT/'SOURCE_MIGRATION.json'),
        '--pca-fit-cache-root',plan['pca_fit_cache_root'],
        '--pca-fit-cache-index',str(Path(path('pca_fit_index_complete')).parent),
        '--output',str(ROOT/'configured'),'--training-output-root',plan['training_output_root'],
        '--max-seconds','1800'])]


def bind_source(raw):
    source=raw.decode()
    changes=[
      ('st_full_population_provider_preparation_admission_v1','st_T3_cpu_preparation_admission_v1',1),
      ('APPROVED_PROVIDER_METADATA_PREPARATION_ONLY','APPROVED_T3_METADATA_PREPARATION_ONLY',1),
      ('st_full_population_provider_preparation_plan_v1','st_T3_cpu_preparation_plan_v1',1),
      ('[0,1]','[2,3]',3), ('{0,1}','{2,3}',2), ('(0,1)','(2,3)',1),
      ('else 1540','else 340',1),
      ("fit=reads.document(plan['inputs']['fit_result'])\n    if fit.get('status')!='COMPLETE' or fit.get('stage')!='fit':raise ValueError('Original fit300 has not completed')",
       "history=reads.document(plan['inputs']['history_complete'])\n    if history.get('status')!='COMPLETE' or history.get('scenes')!=32:raise ValueError('Exact32 history has not completed')",1),
      ("b.load(HERE.parent/'w_storage_bridge_v1/storage_bridge.py'", "b.load(REVIEWED_HOME.parent/'w_storage_bridge_v1/storage_bridge.py'",1),
      ("metadata.tree_inventory(ROOT,max_depth=3,max_entries=64)", 'composition_inventory(metadata,ROOT)',1),
      ("            atomic_json(ROOT/'PROFILE_RECIPE.json',plan['recipe'],immutable=True)\n            atomic_json(ROOT/'CONSUMPTION.json',plan['consumption'],immutable=True)\n",'',1),
      ("for name in ('SOURCE_MIGRATION.json','PROVIDER.json','configured/training_config.json',\n                         'configured/PROVIDER_CHANNEL_SCOPE.json','configured/PROVIDER_CONFIG_LINEAGE.json'):", 'for name in OUTPUTS:',1),
      ("raw=output_reads.raw(ROOT/name);outputs[name]={'path':str(ROOT/name)", "raw=output_reads.raw(output_path(ROOT/name));outputs[name]={'path':str(output_path(ROOT/name))",1),
    ]
    for old,new,count in changes:
        if source.count(old)!=count:raise ValueError('Reviewed runner no longer matches narrow T3 adaptation: '+old)
        source=source.replace(old,new)
    return source


def controller():
    module=types.ModuleType('reviewed_T3_metadata_preparation');module.__file__=str(HERE/'prepare_once.py')
    exec(compile(bind_source(original_bytes()),str(ORIGINAL),'exec'),module.__dict__)
    module.__dict__.update(START=START, BOOT_IO=BOOT_IO, ROOT=ROOT, CONTROL=ROOT/'controls', RUNTIME=RUNTIME,
        COMMIT='a8f8cb879a806ea674ed7be9b1ff7d82483e8b2d', PLAN_SHA=PLAN_SHA, LIMITS=LIMITS,
        OUTPUT_LIMIT=OUTPUT_LIMIT, OUTPUTS=OUTPUTS, commands=commands,
        REVIEWED_HOME=ORIGINAL.parent, BOOTSTRAP=ORIGINAL.parent.parent/'future_index_sequential_control_v1/prepare_once.py',
        composition_inventory=composition_inventory, output_path=output_path)
    return module


if __name__=='__main__':raise SystemExit(controller().main())
