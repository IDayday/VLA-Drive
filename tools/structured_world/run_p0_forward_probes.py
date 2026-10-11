"""CPU-default plan for four origins x three independent P0 forward probes.

Local-host only. Parent freezes/admit resources before --execute. No retry,
signals, SSH, model changes or acceptance; original legacy exit/JSON retained.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.structured_world import benchmark_engineering_execution as bench

BINDING_SHA = '44c78ea7aa563a4d6b863081ffd8db7af92d0fdb549c957c334a66a4fd608799'
PRODUCER_GIT = '6d16abf207877340afd36ecca58e6f1904d37d02'
PRODUCER_SHA = '9c03229e16e4c055724e569c0d76b3138dfad6a99c1e10a08139a2de4ba99919'
MODEL_GIT = 'be5e72248a46afd43e1bf279c21134c391c5edad'
LEGACY_SHA = '31dc8caf31df0daa1e96c074569d1d9cff39ca80ca522f94ae9331534e8d2ffd'
REGISTRATION_SHA = '8bc0cc1dec82c88cc73e8cde4e5e9eac1d4d88633339c6fcd6670a1d984a8007'
BENCH_SHA = 'd358f6969dd78828e705c19f5307c9d26f76b15f37aaadb35930b8cbeb66edee'
REL = 'tools/structured_world/validate_p0_execution_numerics.py'
CORE = {'q0','H_A','B0','Bt','q_final','loss'}
PAIRS = {'A_A','A_B','AA_B'}
PRESET = {'name':'p0_exact_v1','mode':'engineering_fast','atol':0.,'rtol':0.,'relative_L2_max':0.,
          'physical_q_final_max_position_error_m':0.,'all_forward_modules_finite_bitwise':True}


def require(value, message):
    if not value: raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+v)))


def ref(path):
    return {'path':str(Path(path).resolve()),'sha256':bench.digest(path)}


def private_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w')as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)


def sources(manifest):
    require(bench.digest(manifest['source_binding']) == BINDING_SHA,'Original prospective producer binding changed')
    b=load(manifest['source_binding'])
    require(b['producer']['git_sha']==PRODUCER_GIT and b['model']['git_sha']==MODEL_GIT and b['preset']==PRESET
            and b['registration_sha256']==REGISTRATION_SHA and b['state']=='frozen_before_candidate','Wrong P0 producer/model/preset')
    producer=Path(b['producer']['root']);model=Path(b['model']['root'])
    bench.immutable_source(producer,PRODUCER_GIT);bench.immutable_source(model,MODEL_GIT)
    require(b['producer']['files_sha256']=={REL:PRODUCER_SHA}and bench.digest(producer/REL)==PRODUCER_SHA,'Frozen P0 observer changed')
    require(bench.digest(bench.__file__)==BENCH_SHA,'Reviewed process/readiness helper changed')
    p=model/'reports/structured_world/execution_20261011/INDEPENDENT_ACCEPTANCE_PREREGISTRATION_V1.json'
    require(bench.digest(p)==REGISTRATION_SHA,'Original V1 registration changed')
    policy=load(p)
    require(policy['trial_plan']['forward_repeats']==3 and policy['trial_plan']['forward_updates']==0,'Original forward slots changed')
    require(set(b['cases'])=={'G1/initial','G1/mid','G3/initial','G3/mid'},'Four original assignments required')
    return b


def build_plan(manifest, output):
    b=sources(manifest);output=Path(output).resolve()
    require(manifest['schema']=='p0_forward_coordinator_manifest_v1','Explicit forward manifest required')
    require(len(manifest['gpu_uuids'])==len(set(manifest['gpu_uuids']))==8,'Actual full8 native forward ranks required')
    require(len(manifest['origins'])==4 and {r['case']for r in manifest['origins']}==set(b['cases']),'Explicit four origin routes required')
    jobs=[];script=str(Path(b['producer']['root'])/REL)
    for origin in manifest['origins']:
        require(Path(origin['run']).is_absolute()and origin['tag']==f"milestone_{b['cases'][origin['case']]['origin']['completed']:07d}",'Immutable registered origin tag/path required')
        for repeat in range(3):
            key=origin['case'].replace('/','_')+f'_repeat{repeat}';folder=output/key
            args=['probe','--model-source-root',b['model']['root'],'--source-binding',manifest['source_binding'],
                '--source-binding-sha256',BINDING_SHA,'--run',origin['run'],'--tag',origin['tag'],
                '--cache',manifest['cache'],'--dino-root',manifest['dino_root'],'--output',str(folder/'observed'),'--microbatch','4']
            jobs.append({'key':key,'case':origin['case'],'repeat':repeat,'host':manifest['host'],'origin':origin,
                'folder':str(folder),'observed':str(folder/'observed'),'run':str(folder/'process'),'trainer':script,
                'gpu_uuids':manifest['gpu_uuids'],'source_binding':ref(manifest['source_binding']),
                'command':[manifest['python'],'-B','-u','-m','torch.distributed.run','--standalone','--nnodes=1',
                    '--nproc_per_node=8','--max_restarts=0','--',script]+args,
                'CPU_preflight_command':[manifest['python'],'-B',script]+args+['--preflight']})
    return {'schema':'p0_forward_coordinator_plan_v1','jobs':jobs,'driver_sha256':bench.digest(__file__),
        'producer_git_SHA':PRODUCER_GIT,'model_git_SHA':MODEL_GIT,'source_binding_sha256':BINDING_SHA,
        'forward_slots':12,'GPU_jobs_started':0,'original_strict_relabelled':False,'passed':False,'formal_authorized':False,
        'limits':['Three repetitions per origin, each internally runs A/AA/B; no optimizer updates',
            'Local native origins are checked by unchanged producer CPU preflight only at execution',
            'Separate forward observations do not rewrite historical819 native bindings or supply missing full AA_B parameter gradients']}


def stat_exact(row):
    require(row['bitwise_equal']is True and row['nonfinite_elements']==0 and row['advisory_passed']is True
            and row['elements']>0 and row['max_absolute']==0 and row['difference_norm']==0,'Nonfinite/nonexact forward statistic')
    require(all(type(v)not in (float,int)or math.isfinite(v)for v in row.values()),'Nonfinite forward statistic')


def classify(job):
    folder=Path(job['observed']);report_path=folder/'P0_FORWARD_COMPARISON.json';strict_path=folder/'strict/COMPLETE.json'
    process_path=Path(job['run'])/'PROCESS_EXIT.json'
    report,strict,process=(load(p)for p in (report_path,strict_path,process_path))
    require(bench.digest(job['source_binding']['path'])==BINDING_SHA,'Actual source binding changed')
    binding=load(job['source_binding']['path']);prototype_path=Path(job['folder'])/'PROBE.json';prototype=load(prototype_path)
    require(prototype['binding']==job['source_binding']and prototype['job']['command']==job['command'],'Prospective prototype command/binding differs')
    require(process['returncode']in (0,1)and process['monitor_error']is None,'Unexpected worker/monitor failure; retain attempt, no retry')
    require(type(strict['passed'])is bool and (process['returncode']==0)==strict['passed'],'Legacy exit/strict result mismatch')
    require(report['schema']=='execution_p0_exact_forward_stats_v1'and report['preset']==PRESET
            and report['producer_sha256']==PRODUCER_SHA and report['registration_sha256']==REGISTRATION_SHA,'Wrong actual forward producer/preset')
    require(report['producer_source_binding']==job['source_binding']and job['source_binding']['sha256']==BINDING_SHA
            and report['model_source']=={'root':binding['model']['root'],'git_sha':MODEL_GIT}
            and report['original_acceptance_bindings']==binding['original_acceptance_bindings'],'Wrong source binding/model')
    require(report['checkpoint']=={'run':str(Path(job['origin']['run']).resolve()),'tag':job['origin']['tag']},'Wrong actual origin')
    original=report['original_strict']
    require(original['sha256']==bench.digest(strict_path)and Path(original['path']).resolve()==strict_path.resolve()
            and original['passed']==strict['passed']and original['unchanged_original_json']is True and original['tool_sha256']==LEGACY_SHA,'Original strict JSON changed')
    require(strict['training_identity']==binding['cases'][job['case']]['origin']['identity']and strict['checkpoint']==job['origin']['tag']
            and strict['optimized_mode']=='engineering_fast'and strict['attention_diagnostic']==strict['precision_diagnostic']=='native','Original strict origin/operator metadata differs')
    require(report['ranks']==strict['ranks']==8 and report['scene_count']==strict['scene_count']==32
            and report['full_tensor_rank_coverage']is True and report['advisory_forward_passed']is True,'Incomplete full8 actual forward')
    records=report['per_rank'];require(len(records)==8 and {r['rank']for r in records}==set(range(8)),'Missing/duplicate ranks')
    require(len(strict['per_rank'])==8 and {r['rank']for r in strict['per_rank']}==set(range(8)),'Missing strict ranks')
    tokens=[t for r in records for t in r['tokens']]
    require(len(tokens)==len(set(tokens))==32 and all(len(r['tokens'])==4 for r in records),'Actual global32 scene coverage missing')
    require({r['rank']:r['tokens']for r in records}=={r['rank']:r['tokens']for r in strict['per_rank']},'Strict/observer scene order differs')
    names=set(report['logical_forward_module_inventory']);require(CORE<=names and any(n.startswith('task_loss/')for n in names)and any(n.startswith('raw_loss/')for n in names),'Full total/weighted/raw loss inventory missing')
    expected=names|{f'rank{r}/{n}'for r in range(8)for n in names}
    require(set(report['modules'])==set(report['forward_module_inventory'])==expected,'Full module/rank coverage missing')
    for checks in report['modules'].values():
        require(set(checks)==PAIRS,'Three actual role pairs required')
        for row in checks.values():stat_exact(row)
    require(all(report['forward_exact'].get(n)is True for n in CORE|{'RNG','BN','BN_finite','BN_float','BN_counter','query_and_counts','all_buffers','gradient_inventory','buffer_inventory'}),'Actual exact flags failed/missing')
    require(set(report['forward_exact_by_pair'])==PAIRS and all(v is True for checks in report['forward_exact_by_pair'].values()for v in checks.values()),'Role pair exact flags failed')
    for record in records:
        require(set(record['roles'])=={'A','AA','B'}and set(record['comparisons'])==PAIRS,'Incomplete actual role capture')
        for checks in record['comparisons'].values():
            require(set(checks['modules'])==names and all(v is True for v in checks['exact'].values()),'Per-rank capture/exact flags incomplete')
            for row in checks['modules'].values():stat_exact(row)
        started=datetime.fromisoformat(record['roles']['B']['started_UTC'].replace('Z','+00:00'))
        frozen=datetime.fromisoformat(binding['frozen_UTC'].replace('Z','+00:00'))
        prepared=datetime.fromisoformat(prototype['created_UTC'].replace('Z','+00:00'))
        require(started.tzinfo is not None and frozen<=prepared<=started,'Prototype and binding must precede actual B')
    require(report['physical_q_final']['full_tensor_rank_coverage']is True,'Physical full tensor coverage missing')
    for pair in PAIRS:
        p=report['physical_q_final'][pair]
        require(p['finite']is True and p['max_position_error_m']==p['max_heading_error_rad']==0,'Nonfinite/nonexact physical trajectory')
    require(report['full_parameter_gradient_statistics_captured']is False and all(report[k]is False for k in ('passed','formal_authorized','trainer_consumable')),'Do not fabricate full gradient statistics or acceptance')
    cost=load(process_path);require(type(cost['wall_seconds'])in (float,int)and math.isfinite(cost['wall_seconds'])and cost['wall_seconds']>=0,'Actual process cost missing')
    return {'schema':'p0_forward_attempt_observation_v1','status':'completed_no_retry','case':job['case'],'repeat':job['repeat'],
        'original_strict_passed':strict['passed'],'actual_returncode':process['returncode'],'expected_legacy_strict_failure':process['returncode']==1,
        'PROCESS_EXIT':ref(process_path),'P0_FORWARD_COMPARISON':ref(report_path),'original_strict':ref(strict_path),
        'prospective_PROBE':ref(prototype_path),
        'wall_seconds':cost['wall_seconds'],'GPU_hours':cost['wall_seconds']*len(job['gpu_uuids'])/3600,
        'forward_exact_observed':True,'full_parameter_AA_B_gradients_captured':False,'passed':False,'formal_authorized':False}


def run_attempt(job):
    # run_job records actual exit before raising. Suppress only its known rc!=0
    # assertion after independently validating the completed scientific reports.
    failure=None
    require(not Path(job['run']).exists()and not Path(job['observed']).exists(),'NEW process/observer output required; no overwritten output')
    try:bench.run_job(job,job['_locks'])
    except ValueError as exc:
        require(str(exc)=='Real worker stage failed; logs retained, no retry','Unexpected process helper error')
        failure=exc
    try:observation=classify(job)
    except Exception as exc:
        process_path=Path(job['run'])/'PROCESS_EXIT.json'
        if process_path.is_file():
            process=load(process_path)
            private_json(Path(job['folder'])/'INCOMPLETE.json',{'schema':'p0_forward_incomplete_attempt_v1','status':'incomplete_no_retry',
                'PROCESS_EXIT':ref(process_path),'actual_returncode':process['returncode'],'error':str(exc),
                'wall_seconds':process['wall_seconds'],'GPU_hours':process['wall_seconds']*len(job['gpu_uuids'])/3600,'passed':False,'formal_authorized':False})
        if failure:raise ValueError('Incomplete/invalid expected strict failure; attempt retained, no retry')from failure
        raise
    private_json(Path(job['folder'])/'OBSERVATION.json',observation)
    return observation


def execute(manifest,plan,output):
    driver_git=bench.checked(['git','rev-parse','HEAD'],ROOT);bench.immutable_source(ROOT,driver_git)
    require(bench.checked(['git','ls-files','--',str(Path(__file__).relative_to(ROOT))],ROOT),'Freeze coordinator before execution')
    qualification=load(manifest['qualification']['path'])
    require(bench.digest(manifest['qualification']['path'])==manifest['qualification']['sha256']and qualification['passed']is True
            and qualification['host']==manifest['host']and set(qualification['physical_gpu_UUIDs'])==set(manifest['gpu_uuids']),'Actual qualified host/UUID inventory differs')
    versions=qualification['environment']['packages']
    require(set(bench.PACKAGES)<=versions.keys()and 'flash_attn'in versions,'Include all qualified runtime extras')
    code='import json,importlib.metadata as m;print(json.dumps({n:m.version(n)for n in '+repr(sorted(versions))+'}))'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    require(json.loads(bench.checked([manifest['python'],'-B','-c',code],env=env,timeout=30))==versions,'Qualified runtime changed')
    observations=[]
    for job in plan['jobs']:
        sources(manifest);folder=Path(job['folder']);require(not folder.exists(),'NEW distinct attempt output required')
        folder.mkdir(mode=0o700)
        private_json(folder/'PROBE.json',{'schema':'p0_forward_prospective_probe_v1','created_UTC':datetime.now(timezone.utc).isoformat(),
            'job':job,'binding':ref(manifest['source_binding']),'coordinator':ref(__file__),'GPU_process_started':False})
        preflight=json.loads(bench.checked(job['CPU_preflight_command'],env=env,timeout=60))
        require(preflight['preflight_passed']is True and preflight['case']==job['case']and preflight['source_binding_sha256']==BINDING_SHA,'Original producer CPU source/origin preflight failed')
        private_json(folder/'CPU_PREFLIGHT.json',preflight)
        with ExitStack()as stack:
            key=hashlib.sha256(json.dumps(sorted(manifest['gpu_uuids'])).encode()).hexdigest()
            locks=[bench.acquire(stack,Path(manifest['artifact_root'])/'physical_gpu_locks'/(key+'.lock'))]
            gate=bench.readiness(manifest,record_sample=lambda x:private_json(folder/f"RESOURCE_SAMPLE_{x['attempt']}.json",x))
            private_json(folder/'RESOURCE_GATE.json',gate)
            observations.append(run_attempt({**job,'_locks':locks}))
    private_json(Path(output)/'COMPLETE.json',{'schema':'p0_forward_coordinator_completion_v1','completed_forward_slots':len(observations),
        'observations':observations,'GPU_hours':math.fsum(o['GPU_hours']for o in observations),
        'original_strict_relabelled':False,'passed':False,'formal_authorized':False})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    require(not a.output.exists(),'NEW output required; preserve all historical attempts')
    manifest=load(a.manifest);plan=build_plan(manifest,a.output);a.output.mkdir(parents=True,mode=0o700)
    private_json(a.output/'PLAN.json',plan);private_json(a.output/'MANIFEST_BINDING.json',ref(a.manifest))
    if a.execute:execute(manifest,plan,a.output)
    else:print(json.dumps({'status':'CPU_PLANNED_ONLY','forward_slots':12,'GPU_jobs_started':0}))


if __name__=='__main__':main()
