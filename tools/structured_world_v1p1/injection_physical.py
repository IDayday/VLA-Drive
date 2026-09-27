"""Decode immutable P1 actions with the actual released decoder, without VLM."""
import argparse,json
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
from infer import deal_action_1225


def main():
 p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 cfg=OmegaConf.load(a.config);rows=[]
 for path in sorted(Path(a.predictions).glob('*.npz')):
  with np.load(path) as z:
   original=deal_action_1225(z['A0'],act_norm=cfg.datasets.vla_data.act_norm)
   for name in ['legacy_pre_action','append_tail','append_tail_gate0']:
    decoded=deal_action_1225(z[name],act_norm=cfg.datasets.vla_data.act_norm)
    yaw=decoded[...,2]-original[...,2];yaw=np.arctan2(np.sin(yaw),np.cos(yaw))
    xy=np.linalg.norm(decoded[...,:2]-original[...,:2],axis=-1)
    rows.append({'token':path.stem,'variant':name,'xy_max_m':float(xy.max()),'xy_mean_m':float(xy.mean()),'yaw_max_rad':float(abs(yaw).max())})
 result={name:{k:max(r[k] for r in rows if r['variant']==name) for k in ['xy_max_m','xy_mean_m','yaw_max_rad']} for name in ['legacy_pre_action','append_tail','append_tail_gate0']}
 output=Path(a.output)
 if output.exists():raise FileExistsError(output)
 output.write_text(json.dumps({'scene_variants':len(rows),'results':result,'decoder':'infer.deal_action_1225','act_norm':cfg.datasets.vla_data.act_norm},indent=2));print(json.dumps(result))
if __name__=='__main__':main()
