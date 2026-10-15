"""Release only this run's verified disposable RAM copies after all readers exit."""
import json,os,shutil,sys,time
from pathlib import Path
w=Path('/mnt/project/VLA-Drive-minimal-repair-scoring-v15-20261015');sys.path.insert(0,str(w))
from tools.decision_local.common import check_host,read,atomic_json,digest
r=Path('/mnt/project/minimal-repair-artifacts/20261015');o=Path('/mnt/project/decision-local-consequence-artifacts/20261014');resource=check_host();start=time.monotonic();cpu=time.process_time()
assert resource['host'] in ['training-vla-zt','training-vlawm-tsinghua-zt4','training-vlawm-tsinghua-zt5']
assert read(r/'STAGE_B_DISPOSITION_v15.json')['status']=='NOT_RUN'
assert all((Path(j['scores'])/'COMPLETE.json').exists() for j in read(r/'FINAL_NAVTEST_V15_REGISTRY.json')['jobs'])
assert all((Path(j['output'])/'COMPLETE.json').exists() for j in read(r/'BALANCED_V14_BEHAVIOR_REGISTRY.json')['jobs'])
canonical=[o/'anchor_train_full_v1',o/'anchor_dev_full_v1',r/'targets_foldB_v5',r/'targets_dev1696_v5',r/'targets_train64_v5']
assert all((p/'COMPLETE.json').exists() for p in canonical)
roots=[Path('/dev/shm')/name for name in ['decision-local-minimal-repair-v5-20261015','decision-local-repair-labels-v12-20261015','decision-local-repair-labels-smoke-v12-20261015']]
readers=[]
for maps in Path('/proc').glob('[0-9]*/maps'):
 try:content=maps.read_text()
 except (OSError,UnicodeDecodeError):continue
 used=[str(p) for p in roots if str(p)+'/' in content]
 if used:readers.append(dict(pid=int(maps.parent.name),paths=used))
removed=[]
if not readers:
 for p in roots:
  if not p.exists():continue
  assert not p.is_symlink() and p.resolve().parent==Path('/dev/shm') and p.stat().st_dev==Path('/dev/shm').stat().st_dev
  total=sum((Path(root)/name).stat().st_size for root,_,files in os.walk(p) for name in files)
  shutil.rmtree(p);removed.append(dict(path=str(p),logical_file_bytes=total))
result=dict(status='IN_USE_PRESERVED' if readers else 'RELEASED',readers=readers,removed=removed,canonical_shared_assets_retained=[str(p) for p in canonical],
    no_weight_label_or_shared_cache_deleted=True,pressure_scripts_stopped=0,pressure_restoration_required=False,CPU_process_seconds=time.process_time()-cpu,elapsed_seconds=time.monotonic()-start,resource_check=resource,script_sha256=digest(__file__))
atomic_json(r/('TEMP_RAM_RELEASE_'+resource['host']+'_v15.json'),result);print(json.dumps(result))
