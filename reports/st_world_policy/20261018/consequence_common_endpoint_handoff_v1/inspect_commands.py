"""Read-only small metadata/source handoff; never open CSV or binary payloads."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,shlex
OUT=Path(__file__).resolve().parent
ROOT=Path('/mnt/project/st-world-policy-20261016')
EVAL=Path('/mnt/project/VLA-Drive-st-evaluation-agent-20261016/reports/st_world_policy/20261016/evaluation')
PREPS={'C1':EVAL/'full_dev_execution_v1','C2':EVAL/'full_dev_C2_C3_execution_v1','C3':EVAL/'full_dev_C3_execution_v1','C4':Path('/mnt/project/VLA-Drive-st-c4-dev-preparation-20261017/reports/st_world_policy/20261016/evaluation/full_dev_C4_execution_v1')}
RUNTIME=Path('/mnt/project/VLA-Drive-st-flow-context-runtime-v3-20261017')
def read(p):return json.loads(Path(p).read_text())
def pin(p):
 p=Path(p).resolve();return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def checked(ref):
 assert pin(ref['path'])['sha256']==ref['sha256'],ref['path'];return read(ref['path'])
commands={};plans={}
for arm,prep in PREPS.items():
 src=prep/'COMMANDS.json';x=read(src)
 matches=[c for c in x['paired_analyses'] if c['comparison']==arm+'_minus_C0' and c['stage']=='COMMON_ENDPOINT']
 assert len(matches)==1
 cmd=matches[0];a=cmd['argv'];planpath=Path(a[a.index('--plan')+1]);plan=read(planpath)
 if 'analysis_plan' in cmd:checked(cmd['analysis_plan'])
 assert cmd['cwd']==str(RUNTIME) and cmd['max_seconds']==1800
 assert plan['endpoints']['consequence']['COMMON_ENDPOINT']==4096
 assert plan['comparisons'][arm+'_minus_C0']=={'control':'C0','treatment':arm}
 assert plan['populations']['dev']['scenes']==1696 and plan['populations']['dev']['logs']==16
 assert plan['statistics']['bootstrap_replicates']==2000 and plan['statistics']['bootstrap_seed']==20261017
 assert a[a.index('--stage')+1]=='COMMON_ENDPOINT' and a[a.index('--scope')+1]=='FORMAL'
 assert all('step004096' in a[a.index('--'+k)+1] for k in ['control-score','control-export','treatment-score','treatment-export'])
 bindings={}
 for side in ('C0',arm):
  b=plan['formal_runs'][side];reg=checked(b['registration']);approval=checked(b['approval']);identity=checked(b['run_identity_file']);cfg=checked(reg['artifacts']['config'])
  assert identity['registration_sha256']==b['registration']['sha256']
  assert cfg==identity['config'] and cfg['execution']['max_updates']==4096
  assert approval['fingerprint']['registration_sha256']==b['registration']['sha256'] and approval['formal_launch_allowed']
  bindings[side]={'registration':b['registration'],'approval':b['approval'],'run_identity_file':b['run_identity_file'],'runtime_identity':identity['identity']}
 commands[arm]={'source_commands':pin(src),'original_command_unchanged':cmd,'plan':pin(planpath)}
 plans[arm]={'bindings':bindings,'statistics':plan['statistics'],'population':plan['populations']['dev']['manifest']}
assert all(p['statistics']==plans['C1']['statistics'] for p in plans.values())
status={}
for arm in ('C0','C1','C2','C3','C4'):
 stpath=ROOT/f'formal/consequence/runs/{arm}_fulltrain_seed42_official_4096_v1/status.json';st=read(stpath)
 job=ROOT/f'formal/evaluation/jobs/{arm}_step004096_dev1696_seed42_v1'
 s={'observed_training_step':st['completed'],'observed_training_status':st['status'],'status_updated_unix':st['updated_unix'],'status_receipt_at_observation':pin(stpath),'job_root':str(job),'dependencies':{}}
 for name in ('export/IDENTITY.json','export/COMPLETE_rank0.json','official_score/IDENTITY.json','official_score/COMPLETE.json'):
  p=job/name;entry={'exists':p.is_file()}
  if p.is_file():
   entry.update(pin(p))
   if 'COMPLETE' in name:
    c=read(p);entry['small_summary']={k:c[k] for k in ('status','requested','requested_scenes','valid_scored','invalid','unresolved') if k in c}
  s['dependencies'][name]=entry
 s['full_scoring_complete']=s['dependencies']['official_score/COMPLETE.json']['exists']
 status[arm]=s
result={'schema':'st_common_endpoint_analysis_handoff_v1','reviewer':'/root/spatial20k_evidence','created_utc':datetime.now(timezone.utc).isoformat(),'source':pin(RUNTIME/'tools/st_world_policy/evaluate_campaign.py'),'runtime_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=RUNTIME,text=True).strip(),'commands':commands,'plans':plans,'status':status,'payloads_csv_bootstrap_model_calls':0,'dispatch':False,'analysis_ready_now':False,'existing_1024_adapter_reusable_without_change':False,'adapter_reason':'run_fixed_pairs fixed C2/C3 DIAGNOSTIC_1024 commands, pins, outputs and consumed-attempt namespaces; native evaluate_campaign is already generic.'}
(OUT/'HANDOFF.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# Existing common-endpoint commands','', '# cwd: '+str(RUNTIME),'# Run only after both exact full-dev scoring receipts are complete; Lead dispatch only.','']
for arm,r in commands.items():
 c=r['original_command_unchanged'];prefix=['env']+[k+'='+v for k,v in c['env'].items()]
 lines+=['# '+arm+' versus C0: both4096',shlex.join(prefix+c['argv']),'']
(OUT/'COMMANDS.sh.txt').write_text('\n'.join(lines)+'\n')
print(json.dumps({'status':'HANDOFF_COMPLETE_NO_ANALYSIS_RUN','steps':{k:v['observed_training_step'] for k,v in status.items()},'score_complete':{k:v['full_scoring_complete'] for k,v in status.items()},'sha256':pin(OUT/'HANDOFF.json')['sha256']}))
