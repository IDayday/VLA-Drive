"""Audit real training actions through actual dataset loader and released decoder."""
import argparse,csv,hashlib,json,pickle,subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from omegaconf import OmegaConf
from tools.structured_world.runtime import load_dataset
from infer import deal_action_1225


def main():
 p=argparse.ArgumentParser()
 for k in ['config','manifest','data-root','old-root','output']:p.add_argument('--'+k,required=True)
 p.add_argument('--limit',type=int,default=64)
 a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 cfg=OmegaConf.load(a.config)
 OmegaConf.update(cfg,'datasets.reward_data',{'load_reward_data':False},force_add=True)
 OmegaConf.update(cfg,'datasets.vla_data',{'w_neg_traj':None},force_add=True)
 ds=load_dataset(SimpleNamespace(model_config=cfg),a.manifest,a.data_root,a.limit)
 rows=[]
 for i in range(len(ds)):
  example=ds[i];token=example['token']
  with (Path(a.data_root)/'meta/train'/f'{token}.pkl').open('rb') as f:raw=pickle.load(f)
  poses=np.asarray(raw['glo_status']['global_poses'],dtype=np.float64)[:12]
  origin=poses[3];yaw=origin[2];c,s=np.cos(yaw),np.sin(yaw)
  rotation=np.array([[c,-s],[s,c]])
  xy=(poses[4:12,:2]-origin[:2])@rotation
  heading=np.arctan2(np.sin(poses[4:12,2]-yaw),np.cos(poses[4:12,2]-yaw))
  decoded=deal_action_1225(example['action'][None],act_norm=cfg.datasets.vla_data.act_norm)[0]
  ye=np.arctan2(np.sin(decoded[:,2]-heading),np.cos(decoded[:,2]-heading))
  rows.append({'token':token,'xy_max_error_m':float(np.abs(decoded[:,:2]-xy).max()),'yaw_max_error_rad':float(np.abs(ye).max()),'steps':len(decoded), 'prompt_contains_future_action': 'action' in example['lang'] and str(example['action'].tolist()) in example['lang']})
 keys=list(rows[0])
 with (out/'ego_training_roundtrip.csv').open('w') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
 root=Path(a.old_root)
 def read(run):
  with (root/f'{run}_pdms/scenes.csv').open() as f:return {r['token']:r for r in csv.DictReader(f)}
 b,t=read('A0'),read('A1_seed42')
 if set(b)!=set(t):raise ValueError('Unpaired scene sets')
 metrics=['score','drivable_area_compliance','no_at_fault_collisions','time_to_collision_within_bound','ego_progress']
 paired=[]
 for token in sorted(b):
  if b[token]['status']!='ok' or t[token]['status']!='ok':raise ValueError('Failed score row')
  r={'token':token,'log':b[token]['log'],'a0_high_to_a1_zero':float(b[token]['score'])>=.9 and float(t[token]['score'])==0}
  for m in metrics:r.update({f'A0_{m}':float(b[token][m]),f'A1_{m}':float(t[token][m]),f'delta_{m}':float(t[token][m])-float(b[token][m])})
  paired.append(r)
 with (out/'a1_vs_a0.csv').open('w') as f:w=csv.DictWriter(f,list(paired[0]));w.writeheader();w.writerows(paired)
 summary={'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'actual_loader_samples':len(rows),'max_xy_roundtrip_m':max(r['xy_max_error_m'] for r in rows),'max_yaw_roundtrip_rad':max(r['yaw_max_error_rad'] for r in rows),'act_norm':cfg.datasets.vla_data.act_norm,'ver_1225':cfg.ver_1225,'reference':'independent SE2 in ego t0, absolute future displacement; raw metadata first 12 frames, current index3','raw_log_time_and_pose_check':'PENDING','config_sha256':hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),'a1_vs_a0':{m:{k:float(np.mean([r[f'{k}_{m}'] for r in paired])) for k in ['A0','A1','delta']} for m in metrics},'high_to_zero':sum(r['a0_high_to_a1_zero'] for r in paired),'high_to_zero_failure_factors':{m:sum(r['a0_high_to_a1_zero'] and r['A1_'+m]==0 for r in paired) for m in metrics[1:]}}
 (out/'EGO_LABEL_AUDIT.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
 if summary['max_xy_roundtrip_m']>1e-4 or summary['max_yaw_roundtrip_rad']>1e-5:raise AssertionError('Ego training representation mismatch')
if __name__=='__main__':main()
