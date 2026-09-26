"""Evaluate the trained provider's auxiliary perception readout on a fixed train manifest."""
import argparse,json
from pathlib import Path
import torch
from adapt_provider import DenseHead,evaluate
from provider_cli import load_provider


def main():
 p=argparse.ArgumentParser()
 for name in ['weights','manifest','target-cache','sensor-root','output']:p.add_argument('--'+name,required=True)
 p.add_argument('--device',default='cuda');a=p.parse_args()
 provider=load_provider(a.weights,a.device);head=DenseHead().to(a.device).eval();state=torch.load(a.weights,map_location=a.device,weights_only=False);head.load_state_dict(state['dense_head'],strict=True)
 result=evaluate(provider,head,json.loads(Path(a.manifest).read_text()),Path(a.target_cache),a.sensor_root,a.device);result.update(device=a.device,precision='FP32',target_cache=a.target_cache,scope='fixed64 training scenes; diagnostic readout not deployed; no updates')
 Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'}))
if __name__=='__main__':main()
