"""Wait for fixed metadata and emit exact cache budgets/commands; never run them."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from starVLA.dataloader.st_world_history import file_sha256
from starVLA.dataloader.st_world_schema import identity_hash
from starVLA.dataloader.st_world_targets import atomic_json,LegacyC1Reader


def run(a):
    root=Path(a.metadata_root);output=Path(a.output);runtime=Path(a.cache_runtime)
    started=time.monotonic()
    if not 0<=a.wait_seconds<=7200 or a.world_size not in (1,8):raise ValueError('Stage-plan budget unsupported')
    if not a.roles or len(set(a.roles))!=len(a.roles):raise ValueError('Roles must be unique')
    while not all((root/role/'COMPLETE.json').exists() for role in a.roles):
        if time.monotonic()-started>=a.wait_seconds:raise TimeoutError('Fixed metadata is not yet complete')
        time.sleep(min(10.,a.wait_seconds))
    source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=runtime,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain=v1'],cwd=runtime,text=True).strip():
        raise ValueError('Cache execution runtime must be immutable/clean')
    if (output/'FULL_CACHE_STAGE_PLAN.json').exists():
        old=json.loads((output/'FULL_CACHE_STAGE_PLAN.json').read_text())
        if (old['identity']!=identity_hash({k:v for k,v in old.items() if k!='identity'}) or
                old['cache_execution_source_commit']!=source_sha or set(old['roles'])!=set(a.roles)):
            raise ValueError('Existing stage plan changed')
        for role,item in old['roles'].items():
            if item['requests_sha256']!=file_sha256(root/role/'TEACHER_REQUESTS.json') or item['manifest_sha256']!=file_sha256(root/role/'SELECTED_SCENES.json'):
                raise ValueError('Stage plan population changed')
        return {'status':old['status'],'identity':old['identity'],'reused':True}
    pilot=json.loads(Path(a.pilot_complete).read_text())
    if pilot['status']!='COMPLETE':raise ValueError('Teacher cost pilot not complete')
    measurements=[]
    for line in Path(a.pilot_log).read_text().splitlines():
        try:value=json.loads(line)
        except json.JSONDecodeError:continue
        if isinstance(value,dict) and 'elapsed_s_this_invocation' in value:measurements.append(value)
    if len(measurements)!=1 or measurements[0]['identity']!=pilot['identity']:
        raise ValueError('Pilot duration must come from its actual matching build log')
    pilot_seconds=measurements[0]['elapsed_s_this_invocation']
    if pilot_seconds<=0:raise ValueError('Actual measured teacher pilot elapsed required')
    roles={};commands=[];budgets=[];legacy_source=LegacyC1Reader(a.legacy_c1_root)
    for role in a.roles:
        directory=root/role;done=json.loads((directory/'COMPLETE.json').read_text())
        registration=json.loads((directory/'REGISTRATION.json').read_text())
        requests=json.loads((directory/'TEACHER_REQUESTS.json').read_text())
        selection=json.loads((directory/'SELECTED_SCENES.json').read_text())
        if (done['status']!='COMPLETE' or requests['split']!=role or requests['times_s']!=[0.] or
                done['requests_sha256']!=file_sha256(directory/'TEACHER_REQUESTS.json') or
                done['selected_sha256']!=file_sha256(directory/'SELECTED_SCENES.json') or
                requests['identity']!=identity_hash({k:v for k,v in requests.items() if k!='identity'}) or
                len(requests['scenes'])!=registration['source']['scenes'] or
                [s['token'] for s in requests['scenes']]!=[s['token'] for s in selection['scenes']]):
            raise ValueError('Final fixed population/hash/order mismatch')
        refs={v['sha256']:v for scene in requests['scenes'] for v in scene['frames'][0]['views']}
        missing_legacy=sorted(set(refs)-set(legacy_source.lookup))
        n=len(refs);shards=(n+31)//32;native=n*(18*24*1024*4+18*24)
        pool=n*(6*8*1024*4+6*8);legacy=n*(6*8*1024*2+6*8)
        roles[role]={'scenes':len(selection['scenes']),'logs':done['logs'],'unique_images':n,
            'manifest_path':str(directory/'SELECTED_SCENES.json'),'manifest_sha256':done['selected_sha256'],
            'requests_path':str(directory/'TEACHER_REQUESTS.json'),'requests_sha256':done['requests_sha256'],
            'request_identity':requests['identity'],'source':registration['source'],
            'native_fp32_logical_tensor_bytes':native,'pool3_fp32_logical_tensor_bytes':pool,
            'sp0_exact_legacy_logical_tensor_bytes':legacy,'native_shards':shards,
            'native_estimated_file_bytes':native+shards*344,
            'legacy_c1_current_images_available':n-len(missing_legacy),
            'legacy_c1_missing_current_image_hashes':missing_legacy,
            'teacher_gpu_seconds_linear_pilot_estimate':n/pilot['unique_images']*pilot_seconds,
            'estimate_is_measured_full_run':False,'metadata_preparation_elapsed_s':done['elapsed_s']}
        for variant,dtype,suffix,command in [('SP2','float32','native384','build'),
                                            ('SP1','float32','pool3','derive-pooled'),
                                            ('SP0','float16','legacy_c1','build')]:
            target=output/'cache'/role/suffix;budget_path=output/f'{role}_{variant}_CACHE_BUDGET.json'
            budget={'phase':'teacher_cache','request_manifest_identity':requests['identity'],
                'max_scenes':len(selection['scenes']),'variant':variant,'storage_dtype':dtype,
                'images_per_shard':32,'batch_size':8,'workers':2,'cpu_threads':2,
                'world_size':a.world_size,'max_seconds':7200,'min_free_gib':100,
                'execution_runtime':str(runtime),'execution_source_commit':source_sha,
                'execution_status':'NOT_STARTED','resource_dispatch':'LEAD_REQUIRED',
                'formal_training_approved':False,'current_only':True}
            atomic_json(budget_path,budget,immutable=True);budgets.append(str(budget_path))
            argv=[a.python,'-m','tools.st_world_policy.cache_teacher',command,
                '--requests',str(directory/'TEACHER_REQUESTS.json'),'--teacher-root',a.teacher_root,
                '--output',str(target),'--variant',variant,'--storage-dtype',dtype,
                '--mode','registered','--registration',str(budget_path),'--max-scenes',str(len(selection['scenes'])),
                '--images-per-shard','32','--batch-size','8','--workers','2','--cpu-threads','2',
                '--max-seconds','7200','--min-free-gib','100','--world-size',str(a.world_size)]
            if variant=='SP0':argv+=['--legacy-c1-root',a.legacy_c1_root]
            if variant=='SP1':argv+=['--source-cache',str(output/'cache'/role/'native384')]
            commands.append({'role':role,'variant':variant,'working_directory':str(runtime),'argv_prefix':argv,
                'data_status':'BLOCKED_MISSING_EXACT_C1_IMAGES' if variant=='SP0' and missing_legacy else 'AVAILABLE_OR_WAITING_NATIVE_CACHE',
                'append_required':(['--source-cache-identity','<exact COMPLETE native cache identity>'] if variant=='SP1' else [])+
                                  ['--rank','<unique rank 0..world_size-1>','--device','cuda:0' if variant=='SP2' else 'cpu'],
                'preconditions':['Lead resource assignment; no duplicate rank/output owner',
                    'Exact complete native cache required before SP1' if variant=='SP1' else
                    'Legacy C1 image SHA coverage must pass before SP0' if variant=='SP0' else 'Fixed current-only population complete'],
                'shell_prefix':shlex.join(argv),'executed':False})
    native=sum(x['native_fp32_logical_tensor_bytes'] for x in roles.values())
    total=native+sum(x['pool3_fp32_logical_tensor_bytes']+x['sp0_exact_legacy_logical_tensor_bytes'] for x in roles.values())
    plan={'status':'METADATA_COMPLETE_CACHE_NOT_STARTED','roles':roles,'commands':commands,'budget_files':budgets,
        'cache_execution_runtime':str(runtime),'cache_execution_source_commit':source_sha,
        'cache_teacher_writer_sha256':file_sha256(runtime/'tools/st_world_policy/cache_teacher.py'),
        'pilot_complete_sha256':file_sha256(a.pilot_complete),'pilot_log_sha256':file_sha256(a.pilot_log),
        'pilot_actual_elapsed_s':pilot_seconds,'legacy_c1_identity':legacy_source.IDENTITY,
        'all_variants_logical_tensor_bytes':total,'temporary_native_shard_bytes_per_worker':32*(18*24*1024*4+18*24),
        'free_bytes_snapshot':shutil.disk_usage(output).free,'required_free_reserve_gib':100,
        'teacher_gpu_seconds_linear_pilot_estimate':sum(x['teacher_gpu_seconds_linear_pilot_estimate'] for x in roles.values()),
        'parallel_scaling_measured':False,'cache_started_by_this_script':False,'formal_training_approved':False,
        'cost_boundaries':['current-only; future teacher requests are separate',
            'file/index/receipt overhead additional to logical tensor sizes',
            '8-way speed is not proven by single-GPU pilot',
            'first reader shard checksum and shared IO can dominate random access',
            'SP1 pools the actual completed native FP32 cache; no new DINO inference',
            'SP0 keeps exact old C1 FP16 values and must not silently regenerate missing images']}
    plan['identity']=identity_hash(plan)
    atomic_json(output/'FULL_CACHE_STAGE_PLAN.json',plan,immutable=True)
    return {'status':plan['status'],'identity':plan['identity'],'role_counts':{k:(v['scenes'],v['unique_images']) for k,v in roles.items()},
            'all_variants_logical_tensor_bytes':total,'commands_file':str(output/'FULL_CACHE_STAGE_PLAN.json')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['metadata-root','output','cache-runtime','teacher-root','legacy-c1-root','pilot-complete','pilot-log','python']:
        p.add_argument('--'+name,required=True)
    p.add_argument('--world-size',type=int,default=8)
    p.add_argument('--roles',nargs='+',choices=('train','dev'),default=['train','dev'])
    p.add_argument('--wait-seconds',type=float,default=0.)
    print(json.dumps(run(p.parse_args()),sort_keys=True))


if __name__=='__main__':main()
