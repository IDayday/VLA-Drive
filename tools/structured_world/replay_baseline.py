"""Capture unmodified baseline outputs before adding structured-world tokens."""
import argparse,copy,json,os,random,sys
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--vlm',required=True);p.add_argument('--data-root',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--split',default='train');p.add_argument('--seed',type=int,default=20260926);p.add_argument('--limit',type=int,default=4);a=p.parse_args()
 os.environ['BASE_VLM']=a.vlm;os.environ['VLM_ATTN_IMPLEMENTATION']='sdpa';os.environ['NAVSIM_USE_FEATURE_CACHE']='0';os.environ.pop('NAVSIM_FEATURE_CACHE_ROOT',None)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 from infer import VLAAgent,NavSimDataset
 agent=VLAAgent(a.checkpoint,device='cuda',qwen_forward_mode='auto')
 original=torch.load(Path(a.checkpoint)/'pytorch_model.pt',map_location='cpu',mmap=True,weights_only=True)
 actual=agent.model.state_dict();missing=set(actual)-set(original);unexpected=set(original)-set(actual)
 assert not missing,sorted(missing)
 assert all(k.startswith('rgb_model.') for k in unexpected),sorted(unexpected)
 cfg=copy.deepcopy(agent.model_config);cfg.datasets.video_data.load_2d_data=0;cfg.datasets.gs_data.load_3d_data=0;cfg.w_depth=0;cfg.enable_image_aug=0
 ds=NavSimDataset(a.manifest,split=a.split,video_data_cfg=cfg.datasets.video_data,gs_data_cfg=cfg.datasets.gs_data,reward_data_cfg=cfg.datasets.reward_data,ver_1225=cfg.ver_1225,dataset_cfg=cfg.datasets.vla_data,all_cfg=cfg,data_root=a.data_root,max_samples=a.limit)
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);records=[]
 for i in range(len(ds)):
  example=ds[i];random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
  with torch.inference_mode():result=agent.predict([example])
  arrays={k:(v.detach().float().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)) for k,v in result.items()}
  np.savez(out/f'{example["token"]}.npz',**arrays)
  records.append({'token':example['token'],'output_shapes':{k:list(v.shape) for k,v in arrays.items()},'finite':all(np.isfinite(v).all() for v in arrays.values())})
 (out/'replay.json').write_text(json.dumps({'samples':records,'seed':a.seed,'qwen_mode':agent.qwen_forward_mode,'missing_keys':sorted(missing),'unused_wan_keys':sorted(unexpected),'dtype':str(next(agent.model.parameters()).dtype)},indent=2))
 print(json.dumps(records),flush=True)
if __name__=='__main__':main()
