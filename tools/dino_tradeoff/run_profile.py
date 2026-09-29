"""Launch a bounded full-student profile from generic weights, never an effect run."""
import argparse,json,os,subprocess
from pathlib import Path


def main():
 p=argparse.ArgumentParser(__doc__)
 for key in ('candidate','campaign-root','local-root','index','run-id','qwen','sources'):p.add_argument('--'+key,required=True)
 p.add_argument('--gpus',type=int,required=True);p.add_argument('--updates',type=int,default=120);p.add_argument('--micro-batch',type=int,default=1);p.add_argument('--master-port',type=int,default=29801);a=p.parse_args()
 if a.gpus not in (4,8) or a.updates not in (4,120):raise ValueError('Registered4/8GPU startup/profile only')
 local=Path(a.local_root);targets=local/'targets'/a.candidate;ident=json.loads((targets/'identity.json').read_text())
 env=dict(os.environ,FORESIGHT_QWEN=a.qwen,FORESIGHT_SOURCES=a.sources,TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
 cmd=['/root/miniconda3/envs/ddp/bin/python','-m','torch.distributed.run','--nproc_per_node',str(a.gpus),'--master_port',str(a.master_port),'-m','tools.foresight.train_student','--config',f'configs/dino_tradeoff/{a.candidate}.yaml','--data',str(local/'student_train_v1'),'--local-image-root',str(local/'images'),'--dino-root',str(targets),'--dino-index',a.index,'--dino-identity',ident['identity'],'--campaign-root',a.campaign_root,'--run-id',a.run_id,'--global-batch','32','--micro-batch',str(a.micro_batch),'--updates',str(a.updates),'--schedule-updates','100000','--warmup','5000','--save-every','60','--milestones','0','--max-seconds','1800' if a.updates==4 else '14400','--campaign-gpu-hours','96','--scope','startup' if a.updates==4 else 'profile','--limit','3840']
 raise SystemExit(subprocess.call(cmd,env=env))
if __name__=='__main__':main()
