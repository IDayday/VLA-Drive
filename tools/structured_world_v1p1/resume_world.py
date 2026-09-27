"""Resume exact run configuration from a terminal, accounted checkpoint."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import torch


def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--worktree',required=True);a=p.parse_args()
 saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);original=saved['identity']['arguments']
 sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.worktree,text=True).strip()
 if sha!=saved['identity']['code_sha']:raise ValueError('Use run-pinned worktree for identical code')
 ledger=json.loads(Path(original['ledger']).read_text());run=next(r for r in ledger['runs'] if r['id']==original['run_id'])
 if run['status']=='running':raise ValueError('Run is live or needs process reconciliation; never launch duplicate')
 if run['optimizer_steps']!=saved['step']:raise ValueError('Checkpoint is behind accounted updates; cannot silently replay')
 if saved['step']>=original['steps']:print('already_complete');return
 args=[sys.executable,'tools/structured_world_v1p1/train_world.py']
 for name,value in original.items():
  if name in ['resume','stop_after'] or value is None:continue
  args.extend(['--'+name.replace('_','-'),str(value)])
 args.extend(['--resume',str(Path(a.checkpoint).resolve())])
 print(json.dumps({'code_sha':sha,'resume_step':saved['step'],'command':args,'cwd':a.worktree}),flush=True)
 os.chdir(a.worktree);os.execv(sys.executable,args)
if __name__=='__main__':main()
