"""Small JSON and exact-root stat only. Never open payloads or metadata indices."""
from pathlib import Path
import copy,hashlib,json,time
HERE=Path(__file__).resolve().parent
FACT=Path('/mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_native2_v1')
HAND=Path('/mnt/project/VLA-Drive-st-indexed-composition-20261017/reports/st_world_policy/20261017/indexed_training2_handoff_v1')
reads=[]
def read(path,sha=None):
 path=Path(path);assert path.suffix=='.json' and not path.is_symlink() and path.stat().st_size<1<<20
 body=path.read_bytes();h=hashlib.sha256(body).hexdigest();assert sha is None or h==sha
 reads.append(dict(path=str(path),bytes=len(body),sha256=h));return json.loads(body)
result=read(FACT/'preparation_control_v1/RESULT.json','afb0c77ccea1b6e368995ad68efa7140e30e1f3a8fa1ab97e716a04f27c704d4')
a=read(FACT/'preparation_control_v1/ADMITTED.json');c=read(FACT/'preparation_control_v1/COUNTERS.json')
assert a['script_sha256']=='0804b3c5406db7bd5f7a53fb206028331e0f5173c53178fe131178d86b708b41'
assert result['status']=='COMPLETE' and result['resources']['exit_code']==0
assert result['resources']['limit_reason'] is None and c['limit_exceeded'] is None
assert result['resources']['counters']==c and result['resources']['process_cleanup']['live_owned_remaining']==0
assert result['resources']['wall_seconds']<=a['wall_seconds']==180
assert result['resources']['sampled_tree_rss_peak_bytes']<=a['rss_budget_bytes']==2<<30
conservative_read=c['proc_self_io_delta']['rchar']+a['known_existing_pca_read_bytes']+result['parent_control_read_reservation_bytes']
assert conservative_read<=a['read_budget_bytes']==4<<30
assert c['gpu_peak_allocated_bytes']==c['gpu_peak_reserved_bytes']==c['native_load_calls']==0
assert result['new_teacher_tensor_bytes']==0 and result['training_started'] is False
paths=[FACT,*FACT.rglob('*')];assert len(paths)<=64 and all(not p.is_symlink() for p in paths)
assert not any(p.suffix in ('.pt','.pth','.safetensors','.bin','.npy','.npz') for p in paths)
physical=sum(p.stat().st_blocks*512 for p in paths);logical=sum(p.stat().st_size for p in paths);charged=sum(max(p.stat().st_size,p.stat().st_blocks*512) for p in paths)
assert charged<=a['output_reservation_bytes']==32<<20
scope=read(FACT/'configured/PROVIDER_CHANNEL_SCOPE.json','dc9c7c5f6be9213a9d755f17d9c12a34a46c4ddf2f10b3c80c952a03bc6ae41e')
lineage=read(FACT/'configured/PROVIDER_CONFIG_LINEAGE.json','e5203bdfcb7eef1b9fb7ffd59b42d524c037e23488daf60830a75509e3d429d1')
pca=read(scope['pca_receipt']['path'],scope['pca_receipt']['sha256'])
assert scope['statistics_refitted'] is False and lineage['pca_refitted'] is False
assert scope['pca_identity']==pca['identity']=='268723c8da3151492fd20a81b0f6f541f348a1187222132b055f4e4a528bfd46'
assert scope['pca_tensor_sha256']==pca['tensor_sha256']
assert scope['fit_cache_identity']==pca['fit_cache_identity']
inputs=read(HAND/'REGISTRATION_INPUTS.json');cfg=read(inputs['known_artifacts']['config']['path'],inputs['known_artifacts']['config']['sha256'])
old=read(FACT/'configured/training_config.json','f6b8bd3ad59a763355130ef8bb0ef9228473f424516c62e630def80e87db948d')
newroot=Path('/mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_training2_v1')
assert Path(cfg['runtime']['output_root'])==newroot/'runs'
assert FACT not in newroot.parents and newroot not in FACT.parents and FACT!=newroot
expected=copy.deepcopy(old);expected['runtime']['output_root']=str(newroot/'runs');assert expected==cfg
assert cfg['execution']['scope']=='SMOKE' and cfg['execution']['stop_after']==2 and cfg['execution']['max_updates']==20000
budget=inputs['proposed_checkpoint_budget'];assert budget['reservation_bytes']==120<<30
stat_inventory=[]
for row in inputs['old_native_receipts']:
 receipt=read(row['receipt']['path'],row['receipt']['sha256']);directory=Path(row['receipt']['path']).parent
 assert receipt['files']==row['files']
 actual={name:(directory/name).stat().st_size for name in row['all_file_sizes']}
 assert actual==row['all_file_sizes']
 assert sum(actual.values())==row['all_directory_bytes']
 stat_inventory.append(dict(receipt=row['receipt'],logical_file_bytes=sum(actual.values()),files_count=len(actual),payload_opened=False))
assert budget['three_slot_bytes']==3*max(x['logical_file_bytes'] for x in stat_inventory)==112097785965
assert budget['reservation_bytes']-budget['three_slot_bytes']==budget['remaining_for_partial_metadata_variation_bytes']==16751232915
out=dict(status='PASS_ACTUAL_FACTORY_CLOSURE_AND_TRAINING_NAMESPACE_SIZING',checked_at_unix=time.time(),pins=reads,small_json_bytes_read=sum(x['bytes'] for x in reads),result_metrics=dict(wall_seconds=result['resources']['wall_seconds'],peak_sampled_rss_bytes=result['resources']['sampled_tree_rss_peak_bytes'],child_rchar_bytes=c['proc_self_io_delta']['rchar'],child_physical_read_bytes=c['proc_self_io_delta']['read_bytes'],conservative_read_with_reserved_overhead_bytes=conservative_read),factory_root_stat=dict(entries=len(paths),logical_root_inclusive_bytes=logical,allocated_root_inclusive_bytes=physical,charged_root_inclusive_bytes=charged),original_pca_unchanged=True,new_teacher_tensor_disk_outputs=0,training_handoff=dict(commit='294b40ea429ca3066cd4229029f65e5aaf9cd671',root=str(newroot),root_exists_at_review=newroot.exists(),registration_exists_at_review=(newroot/'REGISTRATION.json').exists(),one_field_config_change='runtime.output_root',checkpoint_budget_bytes=120<<30,three_slot_logical_reference_bytes=112097785965,headroom_bytes=16751232915,reference_native_stats=stat_inventory),training_or_gpu_admission=False,payload_reads=0,index_database_reads=0,live_ledger_policy_writes=0)
(HERE/'EVIDENCE.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='pins'},sort_keys=True))
