"""Check processed ego supervision against independent raw-log poses and time."""
import argparse,csv,hashlib,json,pickle,subprocess
from collections import defaultdict
from pathlib import Path
import numpy as np


def main():
 p=argparse.ArgumentParser()
 for name in ['manifest','data-root','raw-log-root','output']:p.add_argument('--'+name,required=True)
 p.add_argument('--limit',type=int,default=64);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 grouped=defaultdict(list);tokens=json.loads(Path(a.manifest).read_text())[:a.limit]
 for token in tokens:
  raw=pickle.load((Path(a.data_root)/'meta/train'/f'{token}.pkl').open('rb'))
  log=Path(raw['glo_images']['cam_f0']['image_paths'][3]).parts[-3]
  grouped[log].append((token,raw))
 rows=[];log_hashes={}
 for log,items in grouped.items():
  path=Path(a.raw_log_root)/(log+'.pkl');data=path.read_bytes();log_hashes[log]=hashlib.sha256(data).hexdigest();frames=pickle.loads(data);index={f['token']:i for i,f in enumerate(frames)}
  for token,raw in items:
   i=index[token];window=frames[i-3:i+9]
   if len(window)!=12:raise ValueError('Incomplete ego supervision horizon')
   poses=np.asarray(raw['glo_status']['global_poses'])[:12]
   transforms=np.stack([f['ego2global'] for f in window]);xy=transforms[:,:2,3]
   yaw=np.arctan2(transforms[:,1,0],transforms[:,0,0]);err=np.arctan2(np.sin(poses[:,2]-yaw),np.cos(poses[:,2]-yaw))
   times=np.array([f['timestamp'] for f in window]);dt=np.diff(times)/1e6
   image_match=all(Path(raw['glo_images'][cam.lower()]['image_paths'][3]).name==Path(window[3]['cams'][cam]['data_path']).name for cam in ['CAM_F0','CAM_L0','CAM_R0'])
   rows.append({'token':token,'xy_error_m':float(abs(poses[:,:2]-xy).max()),'yaw_error_rad':float(abs(err).max()),'interval_min_s':float(dt.min()),'interval_max_s':float(dt.max()),'current_token_match':window[3]['token']==token,'current_images_match':image_match})
 with (out/'raw_ego.csv').open('w') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
 report={'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'scenes':len(rows),'raw_log_hashes':log_hashes,'max_xy_error_m':max(r['xy_error_m'] for r in rows),'max_yaw_error_rad':max(r['yaw_error_rad'] for r in rows),'interval_min_s':min(r['interval_min_s'] for r in rows),'interval_max_s':max(r['interval_max_s'] for r in rows),'current_images_and_time_match':all(r['current_token_match'] and r['current_images_match'] for r in rows)}
 report['status']='PASS' if report['max_xy_error_m']<1e-4 and report['max_yaw_error_rad']<1e-5 and report['current_images_and_time_match'] and report['interval_min_s']>.49 and report['interval_max_s']<.51 else 'FAIL'
 (out/'RAW_EGO_AUDIT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
 if report['status']!='PASS':raise AssertionError('Raw ego contract mismatch')
if __name__=='__main__':main()
