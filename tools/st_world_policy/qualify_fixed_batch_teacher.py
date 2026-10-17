"""Fixed-8 teacher qualification: bounded real data, no training or tensor disk.

`plan` reads small metadata only. `run` is a foreground single-GPU SMOKE that
requires Lead resource admission. Legacy compatibility and new-policy stability
are separate gates, neither of which is formal training approval.
"""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import resource
import time

import torch

from starVLA.dataloader.st_world_history import file_sha256
from starVLA.dataloader.st_world_ondemand import BoundedTeacherProvider
from starVLA.dataloader.st_world_schema import identity_hash
from starVLA.dataloader.st_world_targets import tensor_digest, validate_target_manifest
from starVLA.model.modules.st_world.requests import sample_requests, gather_future, gather_current
from tools.st_world_policy import qualify_ondemand_teacher as shared
from tools.st_world_policy.fixed_batch_teacher import FIXED_BATCH_POLICY, FixedBatchDINOExecutor


def make_plan(requests_path, reference_root, current_root, teacher_config, *, max_seconds=900):
    plan = shared.make_plan(requests_path, reference_root, current_root, teacher_config,
                            max_seconds=max_seconds)
    plan.pop('identity')
    plan.update(schema='st_fixed_teacher_qualification_plan_v1',
                runner_source_sha256=file_sha256(__file__), batch_policy=deepcopy(FIXED_BATCH_POLICY))
    plan['source_files'] += [shared.pin(shared.__file__),
                            shared.pin(Path(__file__).with_name('fixed_batch_teacher.py')),
                            shared.pin(Path(__file__).parents[2]/'starVLA/model/modules/foresight/dinov3_target_encoder.py'),
                            shared.pin(Path(__file__).with_name('cache_teacher.py'))]
    plan['backend'].update(deterministic_algorithms=True, deterministic_warn_only=False,
                           cublas_workspace_config=':4096:8')
    plan['cases'] = ['fixed8_all_original_order', 'current_request1', 'current_request4',
                     'current_reversed8', 'current_tail8_plus_duplicate',
                     'provider_cold_partial_lru_cold_resume']
    plan['gate'] = {
        'fixed_policy_repeatability': 'all elements bitwise equal across batch 1/4/8/tail/reorder/partial-LRU/cold-resume',
        'legacy_reference_compatibility': 'separate full tensor comparison; never infer from repeatability',
        'bitwise_masks': True, 'rng_unchanged': True, 'teacher_frozen': True,
        'numeric_tolerance': None,
        'nonzero_difference': 'NUMERICAL_REVIEW_REQUIRED; no post-hoc tolerance',
        'legacy_short_batches': 'may differ even when the new fixed policy is repeatable; retain original cache unchanged',
    }
    ref = shared.open_reader(reference_root)
    # Exactly the existing writer's shard order, but every final compute group
    # is padded to eight. The existing last short group remains a legacy value.
    original_groups = sum(math.ceil(len(ref.identity['records'][i:i+ref.identity['images_per_shard']])/8)
                          for i in range(0, len(ref.identity['records']), ref.identity['images_per_shard']))
    n = plan['current_unique_images']
    planned_batches = original_groups + n + math.ceil(n/4) + math.ceil(n/8) + math.ceil((n+1)/8)
    # Each scene has at most 15 unique current/future views; three complete
    # reads are a conservative cap for cold/partial/cold-restored execution.
    planned_batches += plan['scenes'] * 3 * 2
    record_bytes = math.prod(ref.contract.output_grid_hw)*(ref.contract.channels*4+1)
    plan['budget'].update(resident_provider_bytes=8*record_bytes,
                          max_image_forwards=8*planned_batches,
                          max_batch_calls=planned_batches,
                          accounting='image_forwards counts real compute slots including repeated padding',
                          max_scene_bytes=64*shared.MiB)
    plan['identity'] = identity_hash(plan)
    return plan


def validate_plan(plan):
    if (plan.get('schema') != 'st_fixed_teacher_qualification_plan_v1' or
            plan.get('identity') != identity_hash({k:v for k,v in plan.items() if k!='identity'}) or
            plan.get('runner_source_sha256') != file_sha256(__file__)):
        raise ValueError('Fixed teacher qualification plan/source changed')
    for entry in plan['inputs']+plan['source_files']:
        if file_sha256(entry['path']) != entry['sha256']:
            raise ValueError('Pinned qualification input changed: '+entry['path'])
    expected = make_plan(plan['requests_path'], plan['reference_root'], plan['current_root'],
                         plan['teacher_config'], max_seconds=plan['budget']['max_seconds'])
    if plan != expected:
        raise ValueError('Fixed teacher qualification protocol/budget differs from registered runner')


class GuardedFixedExecutor:
    def __init__(self, executor, budget, device):
        self.executor, self.budget, self.device = executor, budget, device
        self.execution_identity = executor.execution_identity
        self.contract_identity = executor.contract_identity
        self.rng_checks = 0

    def compute_batch(self, records, **kwargs):
        # Admit actual fixed compute, not just the non-padding labels returned.
        self.budget.admit(8)
        before = shared.rng_digest(self.device)
        try:
            return self.executor.compute_batch(records, **kwargs)
        finally:
            if before != shared.rng_digest(self.device):
                raise RuntimeError('Fixed teacher changed external RNG')
            self.rng_checks += 1
            self.budget.check()


def run(plan_path, output):
    plan = shared.read_json(plan_path)
    validate_plan(plan)  # No CUDA, weights, feature loads or output before this.
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new qualification output; preserve earlier evidence')
    output.mkdir(parents=True)
    shared.bounded_json(output/'RESOLVED_PLAN.json', plan, plan['budget']['metadata_bytes'])
    budget = shared.Budget(plan['budget'])
    result = {'schema':'st_fixed_teacher_qualification_v1', 'plan_identity':plan['identity'],
              'status':'STARTING', 'batch_policy':deepcopy(FIXED_BATCH_POLICY),
              'backend':deepcopy(plan['backend']), 'cases':{}, 'errors':[],
              'fixed_policy_repeatability':False, 'legacy_reference_compatibility':False,
              'completed_all_cases':False, 'formal_approved':False,
              'scientific_gain':'UNTESTED', 'tensor_disk_bytes':0}
    executor = None
    code = 1
    try:
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
            raise ValueError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before this qualification process starts')
        torch.set_num_threads(2)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError('Explicitly expose exactly one Lead-admitted GPU')
        device = torch.device('cuda:0')
        torch.cuda.init(); torch.cuda.get_rng_state(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        ref, current = shared.open_reader(plan['reference_root']), shared.open_reader(plan['current_root'])
        manifest = validate_target_manifest(shared.read_json(plan['requests_path']))
        contract = ref.contract
        before = shared.rng_digest(device)
        executor = FixedBatchDINOExecutor(plan['teacher_root'], contract=contract,
                                          variant=plan['variant'], device=str(device))
        if shared.rng_digest(device) != before:
            raise RuntimeError('Fixed teacher construction changed external RNG')
        checked = GuardedFixedExecutor(executor, budget, device)
        result.update(execution=executor.execution, execution_identity=executor.execution_identity,
                      construction_rng_unchanged=True, device=torch.cuda.get_device_name(device),
                      frozen_initial_versions_hash=identity_hash(executor._versions))
        records = ref.identity['records']; by_key = {r['key']:r for r in records}
        current_keys = shared._keys(manifest, contract, True)
        baseline = {}
        internal_rows, legacy_rows = [], []

        def save_progress():
            shared.bounded_json(output/'PROGRESS.json', result, plan['budget']['metadata_bytes'])

        def compare_case(name, groups, keep=False):
            tick = time.monotonic(); rows = []
            for group in groups:
                produced = checked.compute_batch(tuple(group), contract=contract,
                                                  execution_identity=checked.execution_identity)
                for i, row in enumerate(group):
                    key = row['key']; stored, mask = ref.read_image(key)
                    if not torch.equal(mask, produced.valid[i]):
                        raise ValueError('Fixed teacher changed a reference valid mask')
                    legacy_diff = shared.difference(produced.features[i], stored)
                    new_diff = shared.difference(produced.features[i], baseline[key][0]) if key in baseline else None
                    legacy_rows.append(legacy_diff)
                    if new_diff is not None:
                        internal_rows.append(new_diff)
                    rows.append({'key':key, 'legacy_reference':legacy_diff, 'fixed8_reference':new_diff,
                                 'computed_digest':tensor_digest(produced.features[i]),
                                 'legacy_digest':tensor_digest(stored)})
                    if keep:
                        with torch.inference_mode(False):
                            baseline[key] = (produced.features[i].clone(), produced.valid[i].clone())
                budget.check()
            result['cases'][name] = {'seconds':time.monotonic()-tick, 'all_masks_equal':True,
                                    'legacy_reference':shared.summarize([r['legacy_reference'] for r in rows]),
                                    'fixed8_reference':shared.summarize([r['fixed8_reference'] for r in rows if r['fixed8_reference'] is not None]),
                                    'records':rows}
            save_progress()

        def chunks(keys, size):
            return [[by_key[k] for k in keys[i:i+size]] for i in range(0, len(keys), size)]

        groups = []
        for start in range(0, len(records), ref.identity['images_per_shard']):
            shard = records[start:start+ref.identity['images_per_shard']]
            groups.extend(shard[i:i+8] for i in range(0, len(shard), 8))
        compare_case('fixed8_all_original_order', groups, keep=True)
        compare_case('current_request1', chunks(current_keys, 1))
        compare_case('current_request4', chunks(current_keys, 4))
        compare_case('current_reversed8', chunks(list(reversed(current_keys)), 8))
        compare_case('current_tail8_plus_duplicate', chunks(current_keys+[current_keys[0]], 8))

        provider = BoundedTeacherProvider(manifest, contract, [current], executor=checked,
            execution_identity=checked.execution_identity,
            max_cached_bytes=plan['budget']['resident_provider_bytes'],
            max_scene_bytes=plan['budget']['max_scene_bytes'], max_batch_images=8,
            max_manifest_bytes=8*shared.MiB)
        patch = torch.Generator().manual_seed(20261017)
        horizon = torch.Generator().manual_seed(20261018)
        tick = time.monotonic(); rows = []; partial_seen = False
        for scene in manifest['scenes']:
            token = scene['token']
            provider.load_state_dict(provider.state_dict())
            request_states = (patch.get_state(), horizon.get_state())
            cold = None; expected_ids = None
            for case in ('cold', 'partial_lru', 'cold_resume'):
                if case == 'cold_resume':
                    provider.load_state_dict(provider.state_dict())
                patch.set_state(request_states[0]); horizon.set_state(request_states[1])
                count_before = dict(provider.stats)
                rng_before = shared.rng_digest(device)
                item = provider.read_scene(token, expected_split='train')
                if shared.rng_digest(device) != rng_before:
                    raise RuntimeError('Provider read changed external RNG')
                if case == 'cold':
                    cold = item
                masks_equal = all(torch.equal(item[k], cold[k]) for k in ('current_valid','future_valid'))
                if not masks_equal:
                    raise ValueError('Partial/cold resume changed physical missing-source masks')
                for name in ('current_dense_features', 'future_dense_features'):
                    metric = shared.difference(item[name], cold[name])
                    internal_rows.append(metric)
                    rows.append({'scene':token, 'case':case, 'target':name, 'cold_reference':metric,
                                 'computed_digest':tensor_digest(item[name])})
                # Future-only comparisons check batching against the independent
                # fixed8 reference. Current values intentionally come from the
                # immutable legacy source, whose compatibility is a separate gate.
                for t, frame in enumerate(scene['frames'][1:]):
                    for v, image in enumerate(frame['views']):
                        if image is None:
                            if item['future_valid'][t,v].any():
                                raise ValueError('A missing frame became a valid padded label')
                            continue
                        from starVLA.dataloader.st_world_targets import payload_key
                        key = payload_key(image, contract)
                        z = item['future_dense_features'][t,v]
                        internal_rows.append(shared.difference(z, baseline[key][0]))
                        stored, mask = ref.read_image(key)
                        legacy_rows.append(shared.difference(z, stored))
                        if not torch.equal(item['future_valid'][t,v], mask):
                            raise ValueError('Provider future valid mask changed')
                request = sample_requests(1, len(contract.view_names), contract.output_grid_hw,
                    min(96, math.prod(contract.output_grid_hw)), future=True,
                    patch_generator=patch, horizon_generator=horizon)
                ids = (tensor_digest(request.patch_indices), tensor_digest(request.future_pair))
                if expected_ids is None:
                    expected_ids = ids
                elif ids != expected_ids:
                    raise RuntimeError('Partial LRU/cold resume changed patch/pair requests')
                gather_current(item['current_dense_features'][None], item['current_valid'][None], request)
                gather_future(item['future_dense_features'][None], item['future_valid'][None], request)
                if case == 'partial_lru':
                    hits = provider.stats['ram_hits']-count_before['ram_hits']
                    computed = provider.stats['computed_images']-count_before['computed_images']
                    partial_seen |= hits > 0 and computed > 0
            budget.check()
        if not partial_seen:
            raise RuntimeError('Qualification failed to exercise both RAM hits and compute misses in one read')
        result['cases']['provider_cold_partial_lru_cold_resume'] = {
            'seconds':time.monotonic()-tick, 'records':rows, 'partial_lru_exercised':True,
            'sampler_equal':True, 'all_masks_equal':True, 'provider_memory':provider.memory_report()}
        if executor._frozen_versions() != executor._versions:
            raise ValueError('Teacher frozen state changed')
        result.update(completed_all_cases=True,
                      fixed_policy_repeatability=all(r['bitwise_equal'] for r in internal_rows),
                      legacy_reference_compatibility=all(r['bitwise_equal'] for r in legacy_rows),
                      fixed_policy_summary=shared.summarize(internal_rows),
                      legacy_reference_summary=shared.summarize(legacy_rows),
                      frozen_final_versions_hash=identity_hash(executor._frozen_versions()),
                      rng_forward_checks=checked.rng_checks)
        if result['fixed_policy_repeatability'] and result['legacy_reference_compatibility']:
            result['status'] = 'FIXED_POLICY_AND_LEGACY_BITWISE_QUALIFIED'; code = 0
        elif result['fixed_policy_repeatability']:
            result['status'] = 'FIXED_POLICY_REPEATABLE_LEGACY_REVIEW_REQUIRED'; code = 2
        else:
            result['status'] = 'FIXED_POLICY_NUMERICAL_REVIEW_REQUIRED'; code = 2
    except Exception as error:
        result['status'] = 'PARTIAL_BUDGET' if isinstance(error, (TimeoutError,MemoryError)) else 'FAILED'
        result['errors'].append({'type':type(error).__name__, 'message':str(error)})
        code = 2 if result['status'] == 'PARTIAL_BUDGET' else 1
    finally:
        result.update(elapsed_seconds=time.monotonic()-budget.start,
                      image_forwards=budget.images, batch_calls=budget.calls,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        if executor is not None:
            result['compute_stats'] = executor.compute_stats
        if torch.cuda.is_initialized():
            result.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        result['identity'] = identity_hash(result)
        shared.bounded_json(output/'RESULT.json', result, plan['budget']['metadata_bytes'])
    return result, code


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('plan', help='CPU metadata only; no CUDA, weights or target tensors')
    for name in ('requests', 'reference-root', 'current-cache-root', 'teacher-config', 'output'):
        plan.add_argument('--'+name, required=True)
    plan.add_argument('--max-seconds', type=int, default=900)
    execute = sub.add_parser('run', help='Finite foreground one-GPU SMOKE, never formal training')
    execute.add_argument('--plan', required=True); execute.add_argument('--output', required=True)
    return p


def main():
    args = parser().parse_args()
    if args.command == 'plan':
        plan = make_plan(args.requests, args.reference_root, args.current_cache_root,
                         args.teacher_config, max_seconds=args.max_seconds)
        output = Path(args.output)
        if output.exists():
            raise FileExistsError('Do not replace an existing qualification plan')
        output.mkdir(parents=True)
        shared.bounded_json(output/'PLAN.json', plan, plan['budget']['metadata_bytes'])
        print(json.dumps({'status':'PLANNED_NOT_RUN', 'identity':plan['identity'],
                          'plan':str(output/'PLAN.json'), 'budget':plan['budget']}))
        return 0
    result, code = run(args.plan, args.output)
    print(json.dumps({k:result[k] for k in ('status','image_forwards','batch_calls','elapsed_seconds','errors')}))
    return code


if __name__ == '__main__':
    try:
        exit_code = main()
    except Exception as error:
        print(json.dumps({'status':'FAILED', 'error':type(error).__name__, 'message':str(error)}))
        exit_code = 1
    raise SystemExit(exit_code)
