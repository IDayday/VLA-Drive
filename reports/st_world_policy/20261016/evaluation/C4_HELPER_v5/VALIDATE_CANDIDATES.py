"""Finite CPU metadata tests. Parse definitions only; never launcher main."""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'W_LAUNCH_HELPER_v4'
PREP=HERE.parent/'full_dev_C4_execution_v1'
BASE=Path('/mnt/project/VLA-Drive-st-evaluation-agent-20261016/reports/st_world_policy/20261016/evaluation/full_dev_execution_v1')
STARTED=time.monotonic(); CHECKS=[]; READS={}
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])

def digest(path):
    p=Path(path); assert p.suffix in ('.json','.py') and p.stat().st_size<=2*1024*1024
    b=p.read_bytes(); READS[str(p)]=len(b)
    return hashlib.sha256(b).hexdigest()
def ident(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def check(name,value):
    assert value,name
    CHECKS.append(dict(name=name,status='PASS'))
def rejects(name,call):
    try:call()
    except (AssertionError,KeyError,ValueError,TypeError,FileNotFoundError) as e:
        CHECKS.append(dict(name=name,status='REJECTED_AS_REQUIRED',exception=type(e).__name__));return
    raise AssertionError('Accepted invalid metadata: '+name)
def git_only(args,**kwargs):
    assert args in (['git','rev-parse','HEAD'],['git','status','--porcelain'])
    assert kwargs['cwd']=='/mnt/project/VLA-Drive-st-official-runtime-v2-20261016'
    return subprocess.check_output(args,**kwargs)
def namespace(path):
    tree=ast.parse(path.read_text()); ns=dict(Path=Path,json=json,hashlib=hashlib,identity_hash=ident,subprocess=SimpleNamespace(check_output=git_only))
    allowed={'ROOT','PREP','REPO','SOURCE','C3_V3_PREP','C3_V3_PINS','C4_V5_PREP','C4_V5_PINS','ADDITIVE_V2_PREP','ADDITIVE_V2_PINS','W_ARMS_V4','W_TRAINING_SOURCE_V4','W_TRAINING_RUNTIME_V4','W_LAUNCH_SUPPLEMENT_V4'}
    for n in tree.body:
        if (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in allowed for t in n.targets)) or (isinstance(n,ast.FunctionDef) and n.name!='main'):
            exec(compile(ast.Module(body=[n],type_ignores=[]),str(path),'exec'),ns)
    ns['sha']=digest
    return ns,tree

def static(job,ns):
    r=dict(job=copy.deepcopy(job),schema='st_formal_endpoint_evaluation_binding_v1',evaluation_source_sha=ns['SOURCE'],requested_scenes=1696,seed=42,export_world_size=1,navtest=False,output_root=job['output_root'])
    if job['arm'] in ('C2','C3','C4'):
        r['additive_preparation']={'C2':ns['additive_pin_refs_v2'],'C3':ns['c3_pin_refs_v3'],'C4':ns['c4_pin_refs_v5']}[job['arm']]()
    if job['arm'] in ns['W_ARMS_V4']:r['w_launch_supplement']=ns['W_LAUNCH_SUPPLEMENT_V4']
    return r

base={name:json.loads((BASE/name).read_text()) for name in ('POLICY.json','COMMANDS.json','INTENTS.json')}
c4={name:json.loads((PREP/name).read_text()) for name in ('POLICY.json','COMMANDS.json','INTENTS.json','TRAINING_BINDINGS.json','EXECUTION_PLAN.json')}
binding=c4['TRAINING_BINDINGS.json']['bindings']['C4']
raw=c4['INTENTS.json']['jobs'][0]
values=[raw,binding]+[json.loads(Path(binding[k]['path']).read_text()) for k in ('registration','config','run_identity_file')]
for stem in ('launch_formal_dev_export','launch_completed_dev_score'):
    path=HERE/(stem+'_v5.py');ns,tree=namespace(path);old,oldtree=namespace(OLD/(stem+'_v4.py'))
    oldfunc={n.name:ast.dump(n,include_attributes=False) for n in oldtree.body if isinstance(n,ast.FunctionDef)}
    newfunc={n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
    allowed={'load_fixed_additive_v3','load_pinned_additive_v3','normalize_fixed_additive_job_v3','validate_static_job'}
    check(stem+': unchanged route/native/budget/locks/main/W and all unrelated function bodies',all(newfunc[k]==v for k,v in oldfunc.items() if k not in allowed))
    check(stem+': only new C4 pin helper',set(newfunc)-set(oldfunc)=={'c4_pin_refs_v5'})
    # The export script has top-level main; its old launch/admission body also stays byte-for-byte.
    if stem=='launch_formal_dev_export':
        marker='p = argparse.ArgumentParser(description=__doc__)'
        check(stem+': complete top-level admission/dispatch body unchanged',path.read_text().split(marker,1)[1]==(OLD/(stem+'_v4.py')).read_text().split(marker,1)[1])
    for step in (512,1024,4096):
        r=ns['load_fixed_additive_v3'](f'C4_step{step:06d}_dev1696_seed42_v1',base)
        check(stem+f': actual C4 {step} metadata accepted',r['job']['runtime_identity']==binding['runtime_identity'] and r['pins']==ns['c4_pin_refs_v5']())
        check(stem+f': C4 {step} no W reads/adapter',ns['supplement_w_launch_v4'](r['job'])==(r['job'],None))
        if 'validate_static_job' in ns:
            sealed=static(r['job'],ns);ns['validate_static_job'](sealed,r['job'])
            check(stem+f': C4 {step} static scoring accepted',True)
            for key in ('additive_preparation','job'):
                bad=copy.deepcopy(sealed)
                if key=='job':bad[key]['expected_completed']+=1
                else:bad[key]['POLICY.json']['sha256']='0'*64
                rejects(stem+f': changed score {key} {step}',lambda bad=bad:ns['validate_static_job'](bad,r['job']))
    for name in ns['C4_V5_PINS']:
        original=ns['C4_V5_PINS'][name];ns['C4_V5_PINS'][name]='0'*64
        rejects(stem+': changed '+name+' pin',lambda:ns['load_fixed_additive_v3'](raw['job_id'],base))
        ns['C4_V5_PINS'][name]=original
    for step in (0,511,513,2048,5000):
        rejects(stem+f': undeclared C4 step {step}',lambda step=step:ns['load_fixed_additive_v3'](f'C4_step{step:06d}_dev1696_seed42_v1',base))
    mutations=[('wrong arm',0,('arm',),'C3'),('wrong step',0,('expected_completed',),511),('wrong stage',0,('stage',),'COMMON_ENDPOINT'),('wrong control',0,('control_arm',),'C1'),('wrong paired step',0,('control_job_id',),'C0_step001024_dev1696_seed42_v1'),('wrong comparison',0,('comparison',),'C3_minus_C0'),('wrong source',0,('training_source_sha',),'0'*64),('wrong inference source',0,('evaluation_source_sha',),'0'*64),('null runtime',0,('runtime_identity',),None),('wrong runtime',0,('runtime_identity',),'0'*64),('wrong launch',0,('training_launch','sha256'),'0'*64),('wrong approval',0,('training_approval','sha256'),'0'*64),('wrong binding pin',0,('training_binding_ref','sha256'),'0'*64),('wrong config cell',3,('cell',),'C3'),('wrong registration',1,('registration','sha256'),'0'*64),('altered identity bytes',4,('source_sha',),'0'*64),('guessed checkpoint',0,('checkpoint_binding','tag'),'step_000512_1')]
    for name,index,keys,value in mutations:
        bad=copy.deepcopy(values);target=bad[index]
        for key in keys[:-1]:target=target[key]
        target[keys[-1]]=value
        rejects(stem+': '+name,lambda bad=bad:ns['normalize_fixed_additive_job_v3'](*bad,'C4',ns['c4_pin_refs_v5']()))
    rejects(stem+': old C3-null plan still rejected',lambda:ns['load_fixed_additive_v2']('C3_step000512_dev1696_seed42_v1',base))
    existing=[]
    for j in base['INTENTS.json']['jobs']:
        newjob,newpin=ns['supplement_w_launch_v4'](j);oldjob,oldpin=old['supplement_w_launch_v4'](j)
        check(stem+': prior '+j['job_id']+' unchanged',newjob==oldjob and newpin==oldpin)
        existing.append(newjob)
    for arm in ('C2','C3'):
        for step in (512,1024,4096):
            name=f'{arm}_step{step:06d}_dev1696_seed42_v1'
            new=ns['load_fixed_additive_v3'](name,base);previous=old['load_fixed_additive_v3'](name,base)
            check(stem+': prior '+name+' loader unchanged',new==previous);existing.append(new['job'])
    if 'validate_static_job' in ns:
        for job in existing:
            registration=static(job,ns);ns['validate_static_job'](registration,job);old['validate_static_job'](registration,job)
        check(stem+': all sixteen prior jobs accepted by unchanged static contract',len(existing)==16)

previous=json.loads((BASE.parent/'full_dev_C3_execution_v1/EXECUTION_PLAN.json').read_text());plan=c4['EXECUTION_PLAN.json']
for key in ('statistics','endpoints','comparisons','populations','inference_seed','training_seed','shortlist_rule','optimizer','score_protocol_sha256'):
    check('unchanged scientific '+key,plan[key]==previous[key])
check('original arm definitions unchanged except C4 status',all(plan['arms'][a]==v for a,v in previous['arms'].items() if a!='C4'))
check('C4 scientific definition unchanged except status',{k:v for k,v in plan['arms']['C4'].items() if k!='status'}=={k:v for k,v in previous['arms']['C4'].items() if k!='status'})
check('all previous actual training bindings unchanged',all(plan['formal_runs'][a]==v for a,v in previous['formal_runs'].items()))
check('only C4 added',set(plan['formal_runs'])-set(previous['formal_runs'])=={'C4'})
for job in c4['COMMANDS.json']['checkpoint_jobs']:
    export=job['export']['argv_template'];score=job['score']['argv']
    check(job['job_id']+': full population/world1/fixed seed',all(export[export.index(k)+1]==v for k,v in [('--max-scenes','1696'),('--world-size','1'),('--rank','0'),('--seed','42')]))
    check(job['job_id']+': scorer campaign/two workers/1728 hard cap',all(score[score.index(k)+1]==v for k,v in [('--workers','2'),('--max-calls','1728'),('--run-id','st-world-policy-20261016'),('--retry-attempts','1')]))
    check(job['job_id']+': no launch and exact tag unresolved',job['launch_authorized'] is False and export[export.index('--checkpoint-tag')+1]=='__EXACT_COMPLETE_NATIVE_TAG_REQUIRED__')
for key in ('population_manifest','inference','score','dedup_resume','resource_default'):
    check('unchanged common '+key,c4['POLICY.json'][key]==base['POLICY.json'][key])
for name,expected in c4['POLICY.json']['parent_ten_job_files_unchanged_sha256'].items():
    check('original ten-job bytes '+name,hashlib.sha256((BASE/name).read_bytes()).hexdigest()==expected)
check('deployed v4 helpers unchanged',all(digest(OLD/(stem+'_v4.py'))==digest(Path('/mnt/project/st-world-policy-20261016/resources')/(stem+'_v4.py')) for stem in ('launch_formal_dev_export','launch_completed_dev_score')))
result=dict(schema='st_C4_helper_v5_metadata_tests_v1',status='OWNER_CHECKS_ONLY_REQUIRES_INDEPENDENT_REVIEW',checks_passed=len(CHECKS),checks=CHECKS,elapsed_seconds=time.monotonic()-STARTED,unique_sha_checked_metadata_bytes=sum(READS.values()),sha_checked_metadata_paths=list(READS),no_launcher_main=True,no_GPU_models_native_payloads_scoring_dispatch=True)
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HERE/'CANDIDATE_VALIDATION.json');args=parser.parse_args()
with args.output.open('x') as f:f.write(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(checks_passed=len(CHECKS),elapsed_seconds=result['elapsed_seconds'],output=str(args.output))))
