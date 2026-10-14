#!/usr/bin/env python3
"""Bounded expert matrix dispatch; preserve other jobs and use fixed explicit GPUs."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from iqe.io import read_json,atomic_json,file_hash,lock
from iqe.contracts import require
from iqe.specialist_study.config import load_config
from iqe.specialist_study.resource_guard import check


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--gpus',nargs='+',type=int,required=True)
    p.add_argument('--wait-hours',type=float,default=4.);a=p.parse_args();c=load_config(a.config)
    require(1<=len(a.gpus)<=3 and len(set(a.gpus))==len(a.gpus),'one to three distinct authorized GPUs')
    require(0<a.wait_hours<=12,'bounded prerequisite wait')
    repo=Path(__file__).resolve().parents[2];output=c.root/'runtime/expert_matrix_driver';output.mkdir(parents=True,exist_ok=True)
    with lock(output/'RUN.lock'):
        check(c,'train-expert-cell');started=time.monotonic()
        while not (c.root/'target_audit/TARGET_OPPORTUNITY_AUDIT.json').exists():
            require(time.monotonic()-started<a.wait_hours*3600,'target audit prerequisite wait exhausted')
            atomic_json(output/'STATUS.json',{'status':'WAITING_INPUTS','dependency':'current Base target audit','no_training_started':True})
            time.sleep(30)
        def cli(command,*extra,gpu=None,stdout=None):
            check(c,command);env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(repo))
            if gpu is not None:
                current=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
                free={int(row.split(',')[0]):int(row.split(',')[1]) for row in current.splitlines()}
                require(free[gpu]>=4096,'BLOCKED_RESOURCE: insufficient free memory for bounded head trainer')
                env['CUDA_VISIBLE_DEVICES']=str(gpu)
            else:env['CUDA_VISIBLE_DEVICES']=''
            return subprocess.Popen([sys.executable,'-u','-m','iqe.specialist_study.cli',command,'--config',a.config,*extra],cwd=repo,env=env,stdout=stdout,stderr=subprocess.STDOUT)
        for stage in ('select-pilot','build-paired-expert-data'):
            with open(output/(stage+'.log'),'a') as stream:require(cli(stage,stdout=stream).wait()==0,'prerequisite failed: '+stage)
        matrix=read_json(c.root/'EXPERT_MATRIX_DATA.json');queue=[cell for cell in ('E00','E01','E10','E11') if matrix['cells'][cell].get('status','READY')=='READY'];active={};done={}
        while queue or active:
            for gpu in a.gpus:
                if not queue or gpu in active:continue
                cell=queue.pop(0);folder=c.root/'experts'/cell
                if (folder/'result.json').exists() and read_json(folder/'result.json')['status']=='COMPLETE':done[cell]='REUSED_COMPLETE';continue
                extra=['--cell',cell,'--device','cuda:0']
                if list(folder.glob('step_*.pt.COMPLETE.json')):extra+=['--resume','auto']
                stream=open(output/(cell+'.log'),'a');process=cli('train-expert-cell',*extra,gpu=gpu,stdout=stream)
                active[gpu]=(cell,process,stream)
                atomic_json(output/(cell+'_LAUNCH.json'),{'pid':process.pid,'gpu':gpu,'cmdline':process.args,
                    'cwd':str(repo),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
                    'config_hash':c.identity,'manifest_hash':matrix['hash'],'other_jobs_preserved':True})
            for gpu,(cell,process,stream) in list(active.items()):
                code=process.poll()
                if code is None:continue
                stream.close();done[cell]={'exit_code':code};del active[gpu]
            atomic_json(output/'STATUS.json',{'status':'RUNNING' if active or queue else 'FINISHED','active':{str(g):{'cell':v[0],'pid':v[1].pid} for g,v in active.items()},'queued':queue,'finished':done})
            if active:time.sleep(30)
        require(all(v=='REUSED_COMPLETE' or v['exit_code']==0 for v in done.values()),'one or more matrix cells failed; evidence retained, no unbounded retry')
        print(json.dumps(done))

if __name__=='__main__':main()
