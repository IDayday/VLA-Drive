import argparse,json,os,signal,sys,time,hashlib,importlib.util
from pathlib import Path
P=argparse.ArgumentParser();P.add_argument('--host',required=True);a=P.parse_args()
N=Path('/mnt/project/qwen-world-abcd-rl-best-20261031');T=N/'termination_20261101'
assert json.loads((N/'USER_TERMINATION.json').read_text())['status']=='STOPPING_BY_USER'
S=N/'runtime/abcd_resume_recovery_v1';sys.path.insert(0,str(S));os.environ['WORLD_COMPARE_ROOT']=str(N);os.environ['WORLD_COMPARE_TASK']='qwen_world_abcd_rl_best_20261031'
spec=importlib.util.spec_from_file_location('abcd_termination_worker',S/'tools/qwen_world_abcd_research/job_worker.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
result=[]
for target in json.loads((T/'HOST_JOBS.json').read_text())[a.host]:
 J=Path(target['job']);r=json.loads((J/'RUNNING.json').read_text());assert r['host']==a.host
 if(J/'RESULT.json').exists():
  result.append(dict(arm=target['arm'],job=str(J),already_terminal=json.loads((J/'RESULT.json').read_text())));continue
 leader=Path('/proc')/str(r['child_pid']);assert int((leader/'stat').read_text().split()[21])==r['child_start_ticks']
 rows=m.task_ranks(r);assert len(rows)==8,(target['arm'],'expected8 exact train ranks',rows)
 handles=[];local_ranks=[];sent=[]
 try:
  for row in rows:
   proc=Path('/proc')/str(row['pid']);fd=os.pidfd_open(row['pid']);handles.append((fd,row))
   assert int((proc/'stat').read_text().split()[21])==row['start_ticks'];args=[v for v in(proc/'cmdline').read_bytes().decode().split('\0')if v];env=dict(v.split('=',1)for v in(proc/'environ').read_bytes().decode().split('\0')if '='in v)
   assert row['usr1_handler']and row['entry']=='tools.qwen_world_abcd_research.train';assert args[args.index('--output')+1]==str(N/'runs'/target['arm']);assert env['WORLD_COMPARE_ROOT']==str(N)and env['WORLD_COMPARE_TASK']=='qwen_world_abcd_rl_best_20261031'
   local_ranks.append(int(env['RANK']));row['cmdline_sha256']=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest()
  assert set(local_ranks)==set(range(8))
  for fd,row in handles:
   signal.pidfd_send_signal(fd,signal.SIGUSR1);sent.append(dict(row,signal='SIGUSR1',at_unix=time.time()))
 finally:
  for fd,_ in handles:os.close(fd)
 (J/'STOP_REQUESTED').touch();result.append(dict(arm=target['arm'],job=str(J),signed_ranks=sent,native_boundary_save_requested=True))
out=dict(status='COOPERATIVE_OWNED_RANK_STOP_REQUESTED',host=a.host,results=result,no_other_experiment_signals=True,at_unix=time.time());p=T/(a.host+'_SIGNALS.json');p.write_text(json.dumps(out,indent=2));print(json.dumps(dict(host=a.host,signalled=sum(len(v.get('signed_ranks',[]))for v in result),arms=[v['arm']for v in result])))
