from pathlib import Path
import json,hashlib,gc,struct
import torch,numpy as np
E=Path('/mnt/project/qwen-world-safety-exploration-20261031');H=E/'improvement_training_20261101';RP=Path('/mnt/project/VLA-Drive-qwen-world-safety-exploration-20261031/reports/qwen_world_safety_exploration_20261031');p=H/'analysis/COMMON_10K_REGRESSION_CONFIG_AUDIT.json';d=json.loads(p.read_text())
def stable(x):
 h=hashlib.sha256()
 def visit(v):
  if torch.is_tensor(v):
   if v.numel()>100000:raise ValueError('RNG audit refuses large tensors')
   a=v.detach().cpu().contiguous();h.update(str((str(a.dtype),tuple(a.shape))).encode());h.update(a.numpy().tobytes())
  elif isinstance(v,np.ndarray):h.update(str((str(v.dtype),v.shape)).encode());h.update(v.tobytes())
  elif isinstance(v,dict):
   for k in sorted(v):h.update(str(k).encode());visit(v[k])
  elif isinstance(v,(tuple,list)):
   h.update(type(v).__name__.encode())
   for t in v:visit(t)
  elif isinstance(v,np.generic):visit(v.item())
  else:h.update(json.dumps(v,sort_keys=True,allow_nan=False).encode())
 visit(x);return h.hexdigest()
for a in d['arms']:
 runroot=Path(a['native_checkpoints'][0]['path']).parents[1]
 for c in a['native_checkpoints']:
  if not c.get('available'):continue
  for r in c['ranks']:
   own=torch.load(Path(c['path'])/f"rank{r['rank']}.pt",map_location='cpu',weights_only=False,mmap=True);r['rng_small_state_sha256']={k:stable(v)for k,v in own['rng'].items()};r['rng_hash_method']='canonical value bytes/dtype/shape, not pickle storage identifiers'
   if r['curriculum_summary']:r['curriculum_summary']['private_RNG_sha256']=stable(own['state']['curriculum']['rng_state'])
   del own
 probes=[]
 for collection in(2500,2501,3000):
  values=[]
  for rank in range(8):
   f=runroot/'behavior'/f'collection{collection:06}'/f'rank{rank}.pt'
   if not f.exists():values.append(dict(rank=rank,available=False));continue
   r=torch.load(f,map_location='cpu',weights_only=False,mmap=True);raw=r['structured_advantage']['A_raw'];adv=r['advantages'];values.append(dict(rank=rank,available=True,path=str(f),body_file_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),behavior_update=r['behavior_version']['update'],normalization_std=float(r['normalization_std']),canonical_global_std=float(r['global_std']),rawA_mean=float(raw.mean()),rawA_local_std=float(raw.std(unbiased=False)),actualAdv_mean=float(adv.mean()),actualAdv_RMS=float(adv.square().mean().sqrt()),actualAdv_max_abs=float(adv.abs().max()),actualAdv_cap5_candidates=int(adv.abs().gt(5).sum()),mask_kept=int(r['adv_mask'].sum())));del r
  ns=[v['normalization_std']for v in values if v.get('available')];probes.append(dict(collection=collection,first_optimizer_update=4*(collection-1)+1,scope='Exact onecollection/all8rank saved material, not fullwindow rate',all8_available=len(ns)==8,all8_same_normalization_std=len(set(ns))==1,normalization_std=ns[0]if ns else None,ranks=values))
 a['saved_credit_normalization_probes']=probes
 gc.collect()
d['conclusions_zh'].append('原始Safety rawA归一化std另直接读取2500/2501/3000三个保存collection×全8rank；这与日志canonical rewardstd分开，不把rewardstd冒充advantage标准差。Endpoint的总梯度与联合likelihood单位不同，其早晚都频繁触及clip，不能作为其它W3路线共同10k断点原因。')
d['scope']['saved_behavior_groups_read_for_actual_advstd']=8*3*8;d['scope']['only_small_detached_credit_tensors_touched']=True
d.pop('sha256');d['sha256']=hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest()
for q in[p,RP/p.name]:q.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),normalization=[(a['arm'],[(q['collection'],q['normalization_std'],q['all8_same_normalization_std'])for q in a['saved_credit_normalization_probes']])for a in d['arms']]),indent=2))
