"""Re-run only the actual catchable-signal ownership boundary after owner fix."""
import argparse,hashlib,json,pathlib,resource,subprocess,sys,time
p=argparse.ArgumentParser();p.add_argument('--source-commit',required=True);p.add_argument('--controller-sha256',required=True);args=p.parse_args()
root=pathlib.Path(__file__).resolve().parent;runtime=pathlib.Path('/mnt/project/VLA-Drive-st-endpoint-supervisor-20261018');file=runtime/'resources/st_endpoint_supervisor_v1.py'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=runtime,text=True).strip()==args.source_commit
assert not subprocess.check_output(['git','status','--porcelain'],cwd=runtime).strip()
assert hashlib.sha256(file.read_bytes()).hexdigest()==args.controller_sha256
old=(root/'SIGTERM_WINDOW_REPRODUCTION.py').read_text();start=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN);rows=[]
for number in (15,2):
 name=f'SIGNAL_WINDOW_FINAL_{number}.json'
 script=old.replace('os.kill(os.getpid(),signal.SIGTERM)',f'os.kill(os.getpid(),{number})').replace('SIGTERM_WINDOW_REPRODUCTION.json',name)
 argv=[sys.executable,'-c',"exec(compile("+repr(script)+", "+repr(str(root/'SIGTERM_WINDOW_REPRODUCTION.py'))+", 'exec'), {'__file__': "+repr(str(root/'SIGTERM_WINDOW_REPRODUCTION.py'))+", '__name__': '__main__'})"]
 with (root/f'SIGNAL_WINDOW_FINAL_{number}_OUTPUT.txt').open('x') as log:child=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=30)
 assert child.returncode==0
 value=json.loads((root/name).read_text());value['signal_number']=number
 assert value['controller_sha256']==args.controller_sha256 and value['controller_sha256_after']==args.controller_sha256
 assert value['spawned_count']==1 and value['durable_admission_count']==1 and value['reported_invocations_started']==1
 assert not value['child_left_alive'] and value['fixture_cleanup_confirmed']
 rows.append(value)
after=resource.getrusage(resource.RUSAGE_CHILDREN)
report={'status':'PASS','source_commit':args.source_commit,'controller_sha256':args.controller_sha256,'tests':rows,'wall_seconds':time.monotonic()-start,'waited_child_CPU_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,'maximum_individual_waited_child_RSS_bytes':after.ru_maxrss*1024,'GPU_model_DEV_metriccache_official_calls':0}
(root/'SIGNAL_DELTA_EXECUTION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
