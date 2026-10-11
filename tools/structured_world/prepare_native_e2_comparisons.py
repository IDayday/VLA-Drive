"""Read small metadata and prepare five pinned CPU comparator commands.

No subprocess/SSH execution, tensor load, payload deletion or retry. A terminal
controller/worker failure blocks readiness even if some COMPLETE files exist.
Caller may run these commands only after all three continuous500 roles complete.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import execution_e2_provenance as p
import prepare_execution_e2_envelope as pins

COMPARATOR = pins.MODEL/'tools/structured_world/compare_execution_checkpoints.py'
OFFSETS = (1, 32, 100, 250, 500)


def residency_check(image_root, stage, alias_record):
    """Path equality is a scientific input guard in frozen dataset.py line54.

    The verified alias may resolve to shared bytes on the worker. Do not resolve
    a remote /var/tmp alias in this CPU observer's own filesystem namespace.
    """
    pins.require(Path(image_root) == Path(stage['image_root']), 'Frozen current image residency requires the registered alias path')
    pins.require(stage['scenes'] == 101592 and stage['source_identity'] and stage['image_hash_manifest'], 'Incomplete current image residency metadata')
    pins.require(alias_record, 'Actual worker shared-alias installation receipt required')
    return {'requested_image_root': image_root, 'registered_image_root': stage['image_root'],
            'comparison': 'literal Path equality, as frozen dataset', 'remote_alias_NOT_locally_resolved': True}


def command(python, runs, origin, origin_completed, offset, destination):
    pins.require(offset in OFFSETS and set(runs) == set(p.ROLES), 'Original five observations and three roles required')
    tag = f'milestone_{origin_completed+offset:07d}'
    return [str(python), '-B', str(COMPARATOR), '--reference', str(Path(runs['A'])/'checkpoints'/tag),
            '--reference-replay', str(Path(runs['AA'])/'checkpoints'/tag),
            '--candidate', str(Path(runs['B'])/'checkpoints'/tag), '--origin', str(origin),
            '--threads', '2', '--chunk-elements', '262144', '--output', str(destination)]


def status(reader, controller, case, jobs):
    root = Path(controller)
    missing, failures = [], []
    results = root/'RESULTS.json'
    if (root/'FAILURE.json').is_file():
        failure = reader.read(reader.file(root/'FAILURE.json'))
        failures.append({'controller_failure': failure.get('error'), 'completed_jobs': failure.get('completed_jobs')})
    if results.is_file():
        result = reader.read(reader.file(results))
        if result.get('error') is not None or result.get('status') in ('FAILED', 'BUDGET_REFUSED_BEFORE_NEXT_JOB', 'STOPPED_BEFORE_NEXT_JOB'):
            failures.append({'controller_status': result.get('status'), 'error': result.get('error')})
    refs = {}
    for role, job in jobs.items():
        run = Path(job['run']); records = {}
        for name in ('PROCESS.json', 'PROCESS_EXIT.json', 'TRAINING_COMPLETE.json', 'identity.json'):
            path = run/name
            if not path.is_file():
                missing.append(role+'/'+name); continue
            records[name] = reader.read(reader.file(path))
        if 'PROCESS_EXIT.json' in records:
            x = records['PROCESS_EXIT.json']
            if x['returncode'] != 0 or x.get('monitor_error') is not None:
                failures.append({'role': role, 'returncode': x['returncode'], 'monitor_error': x.get('monitor_error')})
        if 'PROCESS.json' in records:
            pins.require(records['PROCESS.json']['command'] == job['command'], 'Actual PROCESS argv differs from phase enrollment')
        if 'TRAINING_COMPLETE.json' in records:
            done = records['TRAINING_COMPLETE.json']; origin = int(job['origin']['tag'].split('_')[-1]); end = origin+500
            pins.require(done['updates'] == end and records.get('identity.json', {}).get('identity') == done['identity'], 'Actual final identity/update differs')
        capture = root/(case+'_'+role+'_CAPTURE.json')
        if not capture.is_file():
            missing.append(role+'/CAPTURE'); continue
        value = reader.read(reader.file(capture))
        pins.require(value['captures_complete'] is True and value['role'] == role and value['training_source_sha'] == pins.PINS['model'][1], 'Incomplete/foreign capture')
        checkpoints = value['checkpoints']
        origin = int(job['origin']['tag'].split('_')[-1])
        pins.require(len(checkpoints) == 5 and {c['completed'] for c in checkpoints} == {origin+n for n in OFFSETS}, 'Five full native observations required')
        refs[role] = reader.file(capture)
    if not (root/'COUNT_SCENE_HORIZON_TRIPLETS.json').is_file():
        missing.append('COUNT_SCENE_HORIZON_TRIPLETS.json')
    return {'ready_for_full_CPU_comparison': not failures and not missing, 'failures': failures,
            'missing': missing, 'captures': refs, 'scope': 'Metadata readiness only, not native payload/numeric verification'}


def prepare(enrollment_path, controller, case, python, results_root, output):
    reader = p.Reader(); enrollment_ref = reader.file(enrollment_path); enrollment = reader.read(enrollment_ref)
    pins.require(enrollment['schema'] == 'execution_E2_phase_enrollment_v2' and enrollment['kind'] == 'native'
                 and case in enrollment['selected_cases'], 'Actual selected native phase enrollment required')
    reader.file(COMPARATOR, pins.COMPARATOR_SHA)
    jobs = {j['key'][1]: j for j in enrollment['jobs'] if j['key'][0] == case}
    pins.require(set(jobs) == set(p.ROLES), 'Three actual enrolled native roles required')
    plan = reader.read(reader.file(Path(controller)/'PLAN.json'))
    pins.require(plan['source_sha'] == pins.PINS['model'][1] and plan['manifest_sha256'] == enrollment['manifest']['sha256'], 'Actual controller/source/manifest differs')
    actual_jobs = {j['role']: j for j in plan['jobs'] if j['case'] == case}
    pins.require(set(actual_jobs) == set(jobs) and all(actual_jobs[r]['command'] == jobs[r]['command'] for r in p.ROLES), 'Controller changed the prospective jobs')
    state = status(reader, controller, case, jobs)
    origins = {(j['origin']['run'], j['origin']['tag']) for j in jobs.values()}
    pins.require(len(origins) == 1, 'Different role origins')
    run, tag = origins.pop(); start = int(tag.split('_')[-1]); origin = Path(run)/'checkpoints'/tag
    routes = {r:j['run'] for r,j in jobs.items()}
    root = Path(results_root)
    pins.require(root.is_absolute() and not root.exists(), 'NEW comparison report root required; no overwrite/retry')
    cmds = [{'offset':n, 'command':command(python,routes,origin,start,n,root/f'offset_{n:04d}/COMPARISON.json'),
             'stdout':str(root/f'offset_{n:04d}/stdout.log'), 'stderr':str(root/f'offset_{n:04d}/stderr.log')} for n in OFFSETS]
    reader.recheck()
    out = {'schema':'native_E2_CPU_comparison_preparation_v1','created_UTC':datetime.now(timezone.utc).isoformat(),
        'enrollment':enrollment_ref,'controller':str(controller),'case':case,'status':state,'commands':cmds,
        'environment':{'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'},
        'comparator':pins.ref(COMPARATOR,pins.COMPARATOR_SHA),'origin':str(origin),'input_refs':reader.inputs,
        'original_strict_exit':{'0':'original strict passed','2':'completed strict scientific failure; preserve report, continue distinct observation; never retry','other':'infrastructure/incomplete; preserve partial and stop CPU observer'},
        'expected_full_banks':'Actual three roles x five17file banks; all payload hashes/RNG/model/master/moments/BN/groups/scheduler/data verified by original comparator',
        'ready_means':'metadata ready only; execute only when ready, on qualified CPU host with threads2; no concurrent GPU stage save/hash IO',
        'comparison_performed':False,'passed':False,'formal_authorized':False,'GPU_jobs_started':0,'subprocesses_started':0}
    pins.write_new(output,out);return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('enrollment','controller','results-root','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--case',required=True,choices=('G1_mid','G3_mid'))
    parser.add_argument('--python',required=True,type=Path)
    args=parser.parse_args();out=prepare(args.enrollment,args.controller,args.case,args.python,args.results_root,args.output)
    print(__import__('json').dumps({'ready':out['status']['ready_for_full_CPU_comparison'],'failures':out['status']['failures'],
                                  'comparisons':len(out['commands']),'GPU_jobs_started':0}))


if __name__ == '__main__':
    main()
