"""Bounded final evidence collector; JSON and checkpoint stat only, no payload IO."""
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path('/mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_training2_v1')
RUN = ROOT/'runs/indexed_provider_single_updates2_v1'


def main():
    read_bytes = 0
    pins = {}
    def read(path, *, lines=False):
        nonlocal read_bytes
        path = Path(path)
        if path.suffix not in ('.json','.jsonl') or path.stat().st_size > 32 << 20:
            raise ValueError('Only small JSON evidence may be read')
        body = path.read_bytes(); read_bytes += len(body)
        if read_bytes > 32 << 20:
            raise ValueError('Evidence metadata budget exceeded')
        pins[str(path)] = hashlib.sha256(body).hexdigest()
        def invalid(value):
            raise ValueError('Nonfinite JSON constant: '+value)
        return ([json.loads(x,parse_constant=invalid) for x in body.decode().splitlines() if x.strip()]
                if lines else json.loads(body,parse_constant=invalid))
    def digest(value):
        return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if not (ROOT/'EXIT_initial.json').is_file():
        raise SystemExit('STILL_RUNNING_OR_EXIT_RECEIPT_PENDING: no result is published')
    ended = read(ROOT/'EXIT_initial.json')
    config = read(ROOT/'CONFIG.json')
    registration = read(ROOT/'REGISTRATION.json')
    if ended['returncode'] != 0:
        failure = read(RUN/'FAILURE.json') if (RUN/'FAILURE.json').is_file() else None
        output = {'status':'FAILED','exit':ended,'failure':failure,'no_retry':True,'pins':pins}
    else:
        result = read(RUN/'RESULT.json')
        steps = read(RUN/'steps.jsonl',lines=True)
        costs = read(RUN/'checkpoint_costs.jsonl',lines=True)
        provider = read(RUN/'teacher_provider_rank0.json')
        initial_engine = read(RUN/'official_runtime/initial_engine_rank0.json')
        updated_engine = read(RUN/'official_runtime/updated_engine_rank0.json')
        init = read(RUN/'initialization.json')
        selected = read(registration['artifacts']['data_manifest']['path'])
        descriptor = read(registration['artifacts']['teacher_manifest']['path'])
        for name in ('config','data_manifest','teacher_manifest'):
            pin = registration['artifacts'][name]
            if pins[pin['path']] != pin['sha256']:
                raise ValueError('Registered source artifact changed: '+name)
        available = {r['token'] for r in selected['scenes']}
        if (result['completed'] != 2 or result['new_optimizer_updates'] != 2 or result['exposure'] != 4 or
                result['offset'] != 2 or result['epoch'] != 0 or result['status'] != 'PAUSED' or
                result['pause_reason'] != 'BUDGET_PAUSED' or result['formal_launch_approval'] is not False or
                result['scientific_result'] != 'NOT_EVALUATED' or [s['step'] for s in steps] != [1,2]):
            raise ValueError('Run did not terminate at the registered exact diagnostic boundary')
        totals = {}
        seen = []
        for step in steps:
            if (not step['fp32_master_update_verified'] or step['sampled_changed_elements'] <= 0 or
                    not math.isfinite(step['maximum_sampled_change']) or step['maximum_sampled_change'] <= 0 or
                    not math.isfinite(step['gradient_norm']) or step['gradient_norm'] <= 0 or
                    not all(math.isfinite(v) for v in step['losses'].values()) or
                    len(step['sample_ids_by_rank']) != 1 or len(step['sample_ids_by_rank'][0]) != 2):
                raise ValueError('Invalid loss/gradient/master/sample evidence')
            seen.extend(step['sample_ids_by_rank'][0])
            for name, count in step['counts'].items():
                if type(count) is not int or count < 0:
                    raise ValueError('Invalid optimizer denominator')
                totals[name] = totals.get(name,0)+count
        if totals != result['counts'] or len(set(seen)) != 4 or not set(seen) <= available:
            raise ValueError('Counts/exposure/manifest join mismatch')
        if (provider != result['teacher_provider_runtime'] or provider['provider_identity'] != descriptor['provider']['identity'] or
                provider['failures'] != 0 or
                provider['tensor_disk_bytes'] != 0 or provider['peak_resident_bytes'] > provider['resident_limit_bytes'] or
                provider['fixed_compute']['compute_image_slots'] != 8*provider['fixed_compute']['calls'] or
                provider['fixed_compute']['compute_image_slots'] != provider['fixed_compute']['requested_images']+
                provider['fixed_compute']['padding_image_slots']):
            raise ValueError('Provider execution counters or no-tensor-write boundary violated')
        native = []
        for p in sorted((RUN/'checkpoints').glob('*/COMPLETE.json')):
            receipt = read(p)
            if (digest({k:v for k,v in receipt.items() if k!='receipt_identity'}) != receipt['receipt_identity'] or
                    receipt['state']['identity'] != result['identity'] or receipt['world_size'] != 1):
                raise ValueError('Wrong native receipt identity')
            names = {f.name for f in p.parent.glob('*.pt')}
            if names != set(receipt['files']):
                raise ValueError('Native inventory does not match its committed receipt')
            for name, info in receipt['files'].items():
                if Path(name).name != name or (p.parent/name).stat().st_size != info['bytes']:
                    raise ValueError('Native checkpoint file missing or size changed')
            native.append({'step':receipt['state']['completed'],'tag':p.parent.name,
                'native_tensor_bytes':sum(x['bytes'] for x in receipt['files'].values()),
                'files':receipt['files'],'termination_reason':receipt['state']['termination_reason'],
                'receipt_identity':receipt['receipt_identity'],'payload_rehashed_by_observer':False})
        if [r['step'] for r in native] != [1,2] or [r['step'] for r in costs] != [1,2]:
            raise ValueError('Expected exactly two committed native saves, no duplicated terminal snapshot')
        output = {'schema':'st_indexed_student_smoke_actual_v1','status':'REAL_TWO_UPDATE_WIRING_PASS',
            'source_sha':registration['source_sha'],'registration_sha256':pins[str(ROOT/'REGISTRATION.json')],
            'config_sha256':pins[str(ROOT/'CONFIG.json')],'execution':registration['execution'],
            'scientific_result':'NOT_EVALUATED','formal_approved':False,'resume_test_run':False,
            'full_model_bitwise_claim':False,'steady_state_throughput_claim':False,
            'weights_usable_for_formal_init':False,'result':result,'steps':steps,'checkpoint_costs':costs,
            'native_checkpoints':native,'initial_engine_evidence':initial_engine,
            'updated_engine_evidence':updated_engine,'initialization_fields':sorted(init),
            'samples':{'actual_consumed_ids':seen,'all_in_exact32_manifest':True,'unique':len(set(seen))},
            'observer_boundary':'Small JSON hashes plus checkpoint inventory/stat; no tensor/model/image read or new GPU task',
            'metadata_read_bytes':read_bytes,'observed_unix':time.time(),'pins':pins}
    path = Path(__file__).parent/'RESULT.json'
    if path.exists():
        raise FileExistsError('Preserve immutable previous observation')
    path.write_text(json.dumps(output,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':output['status'],'metadata_read_bytes':read_bytes,'output':str(path)}))


if __name__ == '__main__':
    main()
