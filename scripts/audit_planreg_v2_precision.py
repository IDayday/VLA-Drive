"""Inspect ACTUAL serialized parameters, optimizer moments and persistent EMA masters."""
import argparse
import json
from pathlib import Path
import torch

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    c=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    masters={n:str(v.dtype) for n,v in c['model'].items() if n.startswith('ema_teacher.master_')}
    moments={f'{i}.{n}':str(v.dtype) for i,state in c['optimizer']['state'].items() for n,v in state.items() if n in ('exp_avg','exp_avg_sq')}
    visual={n:str(v.dtype) for n,v in c['model'].items() if n.startswith('backbone.model.vision_model.') and '_lora_' in n}
    language={n:str(v.dtype) for n,v in c['model'].items() if n.startswith('backbone.model.language_model.') and '.lora_' in n}
    report=dict(ema_master_count=len(masters),ema_master_dtypes=sorted(set(masters.values())),ema_schema=c['model'].get('ema_teacher._extra_state'),
        optimizer_moment_count=len(moments),optimizer_moment_dtypes=sorted(set(moments.values())),
        visual_lora_linear_count=len(visual),visual_lora_dtypes=sorted(set(visual.values())),
        language_lora_linear_count=len(language),language_lora_dtypes=sorted(set(language.values())),
        optimizer_updates=int(c['model']['optimizer_updates']),ema_updates=int(c['model']['ema_teacher.updates']))
    Path(args.output).write_text(json.dumps(report,indent=2))
    assert masters and moments and all(d=='torch.float32' for d in list(masters.values())+list(moments.values())+list(visual.values())+list(language.values()))
    print(json.dumps({k:v for k,v in report.items() if k!='ema_schema'},indent=2))
