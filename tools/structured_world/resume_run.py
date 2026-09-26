"""Resume only remaining originally budgeted steps from an explicit trusted checkpoint."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import torch


def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);a=p.parse_args()
 payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True);args=dict(payload['arguments'])
 ledger=json.loads(Path(args['ledger']).read_text());entry=ledger['runs'][args['run_id']]
 if payload['step']>entry['consumed'] or entry['reserved']!=args['steps']:raise ValueError('Checkpoint/ledger mismatch')
 if payload['step']==args['steps']:
  print(json.dumps({'status':'already_complete','step':payload['step'],'additional_steps':0}));return
 if payload['step']!=entry['consumed']:raise ValueError('Uncheckpointed charged steps exist; do not silently replay budget. Inspect the run ledger.')
 args['resume']=str(Path(a.checkpoint).resolve());args['stop_after']=None
 cmd=[sys.executable,str(Path(__file__).with_name('train.py'))]
 for name,value in args.items():
  if value is None or value is False:continue
  flag='--'+name.replace('_','-');cmd.append(flag)
  if value is not True:cmd.append(str(value))
 os.execv(sys.executable,cmd)
if __name__=='__main__':main()
