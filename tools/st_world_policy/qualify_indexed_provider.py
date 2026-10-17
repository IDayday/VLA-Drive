"""Finite indexed/cache/on-demand composition qualification; never a trainer.

plan reads bounded metadata and stats payload files, without opening them. run
uses one independently admitted GPU, fixed-8 FP32 teacher and RAM references.
No new teacher tensor is written. A receipt covers only its exact <=32 scenes.
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
from starVLA.dataloader.st_world_schema import TeacherFeatureContract, identity_hash, require_hash
from starVLA.dataloader.st_world_targets import payload_key, tensor_digest
from starVLA.dataloader.st_world_indexed_provider import IndexedTeacherCacheReader, IndexedBoundedTeacherProvider
from starVLA.model.modules.st_world.requests import sample_requests, gather_current, gather_future
from tools.st_world_policy import teacher_provider as tp
from tools.st_world_policy import qualify_ondemand_teacher as shared
from tools.st_world_policy import fixed_batch_teacher as fixed
from tools.st_world_policy.qualify_fixed_batch_teacher import GuardedFixedExecutor

PLAN_SCHEMA = 'st_indexed_composition_plan_v1'
RESULT_SCHEMA = 'st_indexed_composition_qualification_v1'
CASES = ('cold', 'partial_lru', 'cold_restored', 'reordered')
GiB, MiB = 1 << 30, 1 << 20


def _signed(value):
    return value.get('identity') == identity_hash({k:v for k,v in value.items() if k != 'identity'})


def prior_evidence(result_pin, plan_pin):
    q, p = tp.pinned(result_pin), tp.pinned(plan_pin)
    if (not _signed(p) or q.get('plan_identity') != p['identity'] or q.get('errors') != [] or
            q.get('tensor_disk_bytes') != 0 or q.get('construction_rng_unchanged') is not True):
        raise ValueError('Incomplete prior actual fixed8 qualification')
    tp.validate_fixed_qualification(p, q, q['execution'])
    return q, p


def _source_files():
    from starVLA.dataloader import st_world_indexed_provider, st_world_metadata_index, st_world_ondemand, st_world_targets
    from tools.st_world_policy import qualify_fixed_batch_teacher
    return [tp.pin(m.__file__) for m in (tp, shared, fixed, qualify_fixed_batch_teacher,
            st_world_indexed_provider, st_world_metadata_index, st_world_ondemand, st_world_targets)] + [tp.pin(__file__)]


def open_pair(plan):
    req = tp.open_metadata_index(plan['request_index'])
    try:
        src = tp.open_metadata_index(plan['source_index'])
        return req, src
    except BaseException:
        req.close()
        raise


def make_plan(request_index, source_index, teacher_config, prior_result, *, max_seconds=900):
    if type(max_seconds) is not int or not 1 <= max_seconds <= 900:
        raise ValueError('Composition qualification is limited to 900 seconds')
    prior_pin = tp.pin(prior_result)
    prior_plan_pin = tp.pin(Path(prior_result).parent / 'RESOLVED_PLAN.json')
    prior, old = prior_evidence(prior_pin, prior_plan_pin)
    cfgpin = tp.pin(teacher_config); cfg = tp.pinned(cfgpin)
    if cfgpin != tp.pin(old['teacher_config']):
        raise ValueError('Reuse the exact prior qualified teacher build config')
    bindings = {'request_index':tp.index_binding(request_index), 'source_index':tp.index_binding(source_index)}
    req, src = open_pair(bindings)
    try:
        c = TeacherFeatureContract.from_dict(req.receipt['contract'])
        if (req.receipt['cache_identity'] is not None or not 1 <= req.receipt['scene_count'] <= 32 or
                req.receipt['record_count'] > 480 or req.manifest['split'] != 'train' or
                req.manifest['times_s'] != [0., .5, 1., 2., 4.]):
            raise ValueError('Exact request-only train index must contain <=32 scenes and the complete future table')
        if (c.to_dict() != old['contract'] or c.compute_dtype != 'float32' or c.storage_dtype != 'float32' or
                c.pooling['kind'] != 'none' or c.native_grid_hw != c.output_grid_hw or
                src.receipt['cache_identity'] is None or src.manifest['times_s'] != [0.]):
            raise ValueError('Require same native FP32 teacher and COMPLETE current source')
        reader = IndexedTeacherCacheReader(src, expected_identity=src.receipt['cache_identity'], expected_contract=c)
        # Constructing the real provider validates role/source lineage without any tensor/image read.
        record_bytes = math.prod(c.output_grid_hw) * (c.channels * 4 + 1)
        limits = {'max_cached_bytes':8 * record_bytes, 'max_scene_bytes':64*MiB,
                  'max_batch_images':8, 'max_manifest_bytes':32*MiB}
        from types import SimpleNamespace
        execution = prior['execution']
        placeholder = SimpleNamespace(contract_identity=c.identity, execution_identity=identity_hash(execution))
        provider = IndexedBoundedTeacherProvider(req, c, [reader], executor=placeholder,
                        execution_identity=placeholder.execution_identity, **limits)
        rows = list(req.records)  # <=480 explicitly admitted above; never enumerate the full source.
        keys = [r['key'] for r in rows]
        hits = [k for k in keys if k in src.locations]
        misses = [k for k in keys if k not in src.locations]
        if not hits or not misses:
            raise ValueError('Composition qualification must exercise both existing-cache hits and compute misses')
        shards = sorted({src.locations[k][0] for k in hits})
        shard_pins = []
        for sid in shards:
            path = reader.root / f'shard_{sid:06d}.safetensors'
            meta = tp.pin(path.with_suffix('.json')); receipt = tp.pinned(meta)
            if meta['sha256'] != src.receipt_hashes[sid] or receipt['identity'] != reader.identity['identity']:
                raise ValueError('Source shard receipt identity changed')
            shard_pins.append({'shard':sid, 'receipt':meta, 'tensor_path':str(path),
                               'tensor_sha256':receipt['sha256'], 'tensor_bytes':path.stat().st_size})
        scene_rows = list(req.manifest['scenes'])
        per_scene = [len({payload_key(v,c) for f in s['frames'] for v in f['views'] if v is not None}) for s in scene_rows]
        slot_count = sum(len(f['views']) for s in scene_rows for f in s['frames'])
        missing_slots = sum(v is None for s in scene_rows for f in s['frames'] for v in f['views'])
        # Independent baseline + reverse regrouping + four provider layouts. Provider
        # reads whole scenes; this upper bound includes cold restore and padding.
        calls = math.ceil(len(rows)/8) + math.ceil(len(rows)/4) + 4*sum(math.ceil(n/8) for n in per_scene)
        weight_path = Path(cfg['teacher_root']) / 'model.safetensors'
        weight_bytes = weight_path.stat().st_size
        metadata_bytes = sum(Path(i['complete']['path']).stat().st_size +
            Path(i['complete']['path']).with_name('metadata.sqlite').stat().st_size +
            sum(Path(d['path']).stat().st_size for d in tp.pinned(i['complete'])['dependencies'].values())
            for i in bindings.values())
        # rchar is measured; explicit tensor slices/weight mmap reads are also counted.
        # Conservative preload + validation + actual image calls; no entire-shard read per record.
        estimated_read = (3*metadata_bytes + sum(p['tensor_bytes'] for p in shard_pins) +
            (len(hits)+4*sum(per_scene))*record_bytes + 3*weight_bytes +
            calls*8*max(r['image']['bytes'] for r in rows) + 32*MiB)
        if estimated_read > 64*GiB or (len(rows)+len(hits))*record_bytes > 2*GiB:
            raise ValueError('Composition plan exceeds 64 GiB logical input or 2 GiB online reference budget')
        p = {'schema':PLAN_SCHEMA, 'scope':'SMOKE', 'formal_approved':False,
            'training_admission':{'max_scenes':32, 'max_updates':4, 'exact_request_identity':req.receipt['request_identity']},
            **bindings, 'requests':req.receipt['dependencies']['requests'],
            'current_identity':src.receipt['cache_identity'], 'contract':c.to_dict(), 'teacher_config':cfgpin,
            'teacher_root':cfg['teacher_root'], 'variant':cfg['variant'],
            'prior_result':prior_pin, 'prior_plan':prior_plan_pin,
            'execution':execution, 'execution_identity':identity_hash(execution), 'backend':prior['backend'],
            'batch_policy':fixed.FIXED_BATCH_POLICY, 'provider':provider.identity, 'limits':limits,
            'scenes':len(scene_rows), 'scene_ids':[s['token'] for s in scene_rows], 'unique_images':len(rows),
            'keys':keys, 'hit_keys':hits, 'miss_keys':misses, 'source_shards':shard_pins,
            'slots':slot_count, 'missing_slots':missing_slots, 'cases':list(CASES),
            'scene_keys':{s['token']:[payload_key(v,c) for f in s['frames'] for v in f['views'] if v is not None] for s in scene_rows},
            'source_files':_source_files(), 'runner_source_sha256':file_sha256(__file__),
            'budget':{'max_seconds':max_seconds, 'cpu_threads':2, 'visible_gpus':1,
                'ram_peak_bytes':8*GiB, 'gpu_peak_bytes':16*GiB, 'max_logical_input_bytes':64*GiB,
                'planned_input_upper_bytes':estimated_read, 'teacher_weight_mmap_bytes':weight_bytes,
                'online_reference_bytes':len(rows)*record_bytes,
                'composed_reference_extra_bytes':len(hits)*record_bytes,
                'reference_resident_limit_bytes':2*GiB, 'metadata_bytes':32*MiB,
                'resident_provider_bytes':limits['max_cached_bytes'], 'max_scene_bytes':limits['max_scene_bytes'],
                'tensor_disk_bytes':0, 'max_batch_calls':calls, 'max_image_forwards':8*calls},
            'gate':{'numerical_tolerance':None, 'all_elements_bitwise':True, 'all_masks_equal':True,
                'source_legacy_compatibility':'separate cached-FP32 vs fixed8 online comparison on every hit key',
                'prior_population_reused':False, 'partial_lru_and_mixed_path_required':True}}
        p['identity'] = identity_hash(p)
        return p
    finally:
        req.close(); src.close()


CONTROL_MIGRATION_SCHEMA = 'st_indexed_profile_control_migration_v1'
CONTROL_PATHS = ('tools/st_world_policy/teacher_provider.py',
                 'tools/st_world_policy/qualify_indexed_provider.py',
                 'tools/st_world_policy/training_config.py')
MATH_PATHS = ('tools/st_world_policy/fixed_batch_teacher.py',
    'tools/st_world_policy/ondemand_teacher.py', 'tools/st_world_policy/qualify_fixed_batch_teacher.py',
    'tools/st_world_policy/qualify_ondemand_teacher.py', 'tools/st_world_policy/provider_pca.py',
    'tools/st_world_policy/teacher_pca.py', 'tools/st_world_policy/cache_teacher.py',
    'starVLA/model/modules/foresight/dinov3_target_encoder.py', 'starVLA/dataloader/st_world_history.py',
    'starVLA/dataloader/st_world_indexed_provider.py',
    'starVLA/dataloader/st_world_metadata_index.py', 'starVLA/dataloader/st_world_ondemand.py',
    'starVLA/dataloader/st_world_targets.py', 'starVLA/dataloader/st_world_schema.py',
    'starVLA/model/modules/st_world/requests.py')


def validate_control_migration(plan, migration):
    """An explicit archived-code boundary, never a skip-source-check switch.

    Historical controls are checked in place. Only three named control modules
    may differ; every enumerated numerical/reader/PCA/sampler file must match.
    New control code remains subject to new independent review/registration.
    """
    import subprocess
    fields = {'schema','identity','parent_runtime','parent_commit','parent_descriptor',
              'parent_plan','parent_result','controls','mathematics','scope',
              'new_population_tensor_qualified','formal_approved'}
    if (set(migration)!=fields or migration['schema']!=CONTROL_MIGRATION_SCHEMA or not _signed(migration)
            or migration['scope']!='PROFILE' or migration['new_population_tensor_qualified'] is not False
            or migration['formal_approved'] is not False):
        raise ValueError('Invalid explicit PROFILE control-source migration')
    old=Path(migration['parent_runtime']); current=Path(__file__).resolve().parents[2]
    if not old.is_absolute() or old.resolve()!=old:
        raise ValueError('Archived runtime must be a canonical absolute worktree')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=old,text=True).strip()
    if commit!=migration['parent_commit'] or len(commit)!=40:
        raise ValueError('Archived runtime commit changed')
    parent=tp.pinned(migration['parent_descriptor'])
    if (parent.get('profile_admission') is not None or
            parent['qualification_plan']!=migration['parent_plan'] or parent['qualification']!=migration['parent_result']
            or tp.pinned(migration['parent_plan'])!=plan):
        raise ValueError('Migration does not bind the original exact-population descriptor/evidence')
    expected_control=[{'relative':r,'old':tp.pin(old/r),'new':tp.pin(current/r)} for r in CONTROL_PATHS]
    expected_math=[{'relative':r,'old':tp.pin(old/r),'new':tp.pin(current/r)} for r in MATH_PATHS]
    if migration['controls']!=expected_control or migration['mathematics']!=expected_math:
        raise ValueError('Migration source inventory/pins changed')
    import hashlib
    for row in expected_control+expected_math:
        committed=subprocess.check_output(['git','show',commit+':'+row['relative']],cwd=old)
        if hashlib.sha256(committed).hexdigest()!=row['old']['sha256']:
            raise ValueError('Archived runtime source differs from its actual commit')
    if any(r['old']['sha256']!=r['new']['sha256'] for r in expected_math):
        raise ValueError('PROFILE migration changed protected mathematical/reader/PCA/sampler source')
    expected_files=[]
    for pin in _source_files():
        rel=Path(pin['path']).relative_to(current)
        expected_files.append(tp.pin(old/rel))
    if (plan['source_files']!=expected_files or
            plan['runner_source_sha256']!=tp.pin(old/'tools/st_world_policy/qualify_indexed_provider.py')['sha256']):
        raise ValueError('Historical qualification source pins do not match the archived runtime')
    return parent


def prepare_control_migration(parent_descriptor, parent_runtime, output):
    """Metadata proposal only; it never creates a review or launch approval."""
    import subprocess
    parent_pin=tp.pin(parent_descriptor); parent=tp.pinned(parent_pin)
    old=Path(parent_runtime).resolve(); current=Path(__file__).resolve().parents[2]
    migration={'schema':CONTROL_MIGRATION_SCHEMA,'scope':'PROFILE',
        'parent_runtime':str(old),'parent_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=old,text=True).strip(),
        'parent_descriptor':parent_pin,'parent_plan':parent['qualification_plan'],'parent_result':parent['qualification'],
        'controls':[{'relative':r,'old':tp.pin(old/r),'new':tp.pin(current/r)} for r in CONTROL_PATHS],
        'mathematics':[{'relative':r,'old':tp.pin(old/r),'new':tp.pin(current/r)} for r in MATH_PATHS],
        'new_population_tensor_qualified':False,'formal_approved':False}
    migration['identity']=identity_hash(migration)
    plan=tp.pinned(parent['qualification_plan']); report=tp.pinned(parent['qualification'])
    validate_receipt(plan,report,parent,control_migration=migration)
    from starVLA.dataloader.st_world_targets import atomic_json
    atomic_json(output,migration,immutable=True)
    return migration


def validate_plan(plan, *, rebuild=True, control_migration=None):
    if (plan.get('schema') != PLAN_SCHEMA or not _signed(plan) or
            (control_migration is None and plan.get('runner_source_sha256') != file_sha256(__file__))):
        raise ValueError('Indexed composition plan/source changed')
    if (plan.get('scope')!='SMOKE' or plan.get('formal_approved') is not False or
            type(plan.get('scenes')) is not int or not 1 <= plan['scenes'] <= 32 or
            plan.get('training_admission') != {'max_scenes':32,'max_updates':4,
                'exact_request_identity':plan['request_index']['request_identity']} or
            plan.get('cases')!=list(CASES) or not plan.get('hit_keys') or not plan.get('miss_keys') or
            set(plan['hit_keys']) & set(plan['miss_keys']) or
            set(plan['keys']) != set(plan['hit_keys']) | set(plan['miss_keys']) or
            plan['budget']['tensor_disk_bytes']!=0 or plan['budget']['max_seconds']>900 or
            plan['budget']['ram_peak_bytes']>8*GiB or plan['budget']['gpu_peak_bytes']>16*GiB or
            plan['budget']['max_logical_input_bytes']>64*GiB):
        raise ValueError('Indexed composition scope/budget is not the bounded registered smoke')
    if control_migration is not None:
        if rebuild:
            raise ValueError('Historical migration may validate completed evidence only, never rebuild its plan')
        validate_control_migration(plan, control_migration)
    elif plan.get('source_files') != _source_files():
        raise ValueError('Composition must bind the complete active implementation source list')
    for pin in plan['source_files']:
        if tp.pin(pin['path']) != pin:
            raise ValueError('Indexed composition source changed')
    if rebuild:
        expected = make_plan(Path(plan['request_index']['complete']['path']).parent,
            Path(plan['source_index']['complete']['path']).parent, plan['teacher_config']['path'],
            plan['prior_result']['path'], max_seconds=plan['budget']['max_seconds'])
        if plan != expected:
            raise ValueError('Indexed composition plan differs from original protocol, dependencies or budget')


def _rchar():
    return int(dict(line.split(':',1) for line in Path('/proc/self/io').read_text().splitlines())['rchar'])


class Budget(shared.Budget):
    def __init__(self, plan):
        super().__init__(plan); self.start_rchar = _rchar(); self.mmap_bytes = 0
    def admit(self,count):
        self.check()
        if self.images+count>self.plan['max_image_forwards'] or self.calls+1>self.plan['max_batch_calls']:
            raise TimeoutError('Composition fixed compute budget exhausted')
        self.images+=count;self.calls+=1
    def check(self):
        super().check()
        if _rchar()-self.start_rchar+self.mmap_bytes > self.plan['max_logical_input_bytes']:
            raise TimeoutError('Composition logical input byte budget exhausted')
        if torch.cuda.is_initialized() and torch.cuda.max_memory_allocated() > self.plan['gpu_peak_bytes']:
            raise MemoryError('Composition GPU allocation exceeded 16 GiB')


def qualify_layouts(plan, request, source, checked, *, rng, budget, progress=lambda:None):
    """Real adapter path; injected executor/RNG only enable explicit CPU behavior fixtures."""
    c = TeacherFeatureContract.from_dict(plan['contract'])
    rows = list(request.records); by_key = {r['key']:r for r in rows}
    reference = {}; composed = {}; repeated = []; legacy = []
    def compute(group):
        before = rng()
        out = checked.compute_batch(tuple(group), contract=c, execution_identity=plan['execution_identity'])
        if before != rng(): raise RuntimeError('Composition reference changed external RNG')
        if (out.keys != tuple(r['key'] for r in group) or out.contract_identity != c.identity or
                out.execution_identity != plan['execution_identity'] or out.features.dtype != torch.float32 or
                out.features.requires_grad or out.features.device.type != 'cpu' or
                out.features.shape != (len(group), *c.output_grid_hw, c.channels) or
                out.valid.dtype != torch.bool or out.valid.shape != out.features.shape[:-1] or
                not out.valid.all() or not torch.isfinite(out.features).all()):
            raise ValueError('Independent fixed8 reference violated key/shape/mask/FP32 contract')
        return out
    for start in range(0,len(rows),8):
        group = rows[start:start+8]; out = compute(group)
        for i,row in enumerate(group):
            reference[row['key']] = (out.features[i].clone(), out.valid[i].clone())
        budget.check()
    composed.update(reference)
    reference_digests = {k:{'features':tensor_digest(z), 'valid':tensor_digest(m)} for k,(z,m) in reference.items()}
    # Distinct grouping and order checks all new keys, not only prior qualification keys.
    reverse = list(reversed(rows))
    for start in range(0,len(reverse),4):
        group = reverse[start:start+4]; out = compute(group)
        for i,row in enumerate(group):
            z,m = reference[row['key']]
            if not torch.equal(m,out.valid[i]): raise ValueError('Reference regrouping changed masks')
            repeated.append({'key':row['key'], 'difference':shared.difference(out.features[i],z)})
    record_bytes = math.prod(c.output_grid_hw)*(4*c.channels+1)
    for key in plan['hit_keys']:
        budget.mmap_bytes += record_bytes; budget.check()
        z,m = source.read_image(key); expected,mask = reference[key]
        if not torch.equal(m,mask): raise ValueError('Legacy source mask differs from online reference')
        legacy.append({'key':key, 'difference':shared.difference(z,expected)})
        composed[key]=(z.clone(),m.clone())
    provider = IndexedBoundedTeacherProvider(request,c,[source],executor=checked,
        execution_identity=plan['execution_identity'], **plan['limits'])
    if provider.identity != plan['provider']: raise ValueError('Composition provider identity changed')
    case_rows = {case:[] for case in CASES}; partial=False; mixed=False
    patch=torch.Generator().manual_seed(20261017); horizon=torch.Generator().manual_seed(20261018)
    sampler_ids={}; sampler_states={}
    def compare_scene(token,case):
        nonlocal mixed,partial
        before=dict(provider.stats); rng_before=rng()
        item=provider.read_scene(token, expected_split='train')
        budget.mmap_bytes += (provider.stats['immutable_hits']-before['immutable_hits'])*record_bytes
        budget.check()
        if rng()!=rng_before: raise RuntimeError('Indexed provider changed external RNG')
        scene=request.scenes[token]; z=torch.cat((item['current_dense_features'][None],item['future_dense_features']))
        mask=torch.cat((item['current_valid'][None],item['future_valid']))
        comparisons=[]
        for ti,frame in enumerate(scene['frames']):
            for vi,image in enumerate(frame['views']):
                if image is None:
                    if mask[ti,vi].any() or z[ti,vi].count_nonzero():
                        raise ValueError('Missing input became a real label')
                    continue
                key=payload_key(image,c); ref,valid=composed[key]
                if not torch.equal(mask[ti,vi],valid): raise ValueError('Provider changed valid masks')
                comparisons.append({'key':key,'difference':shared.difference(z[ti,vi],ref),
                    'online_difference':shared.difference(z[ti,vi],reference[key][0])})
        patch.set_state(sampler_states[token][0]); horizon.set_state(sampler_states[token][1])
        sample=sample_requests(1,len(c.view_names),c.output_grid_hw,min(96,math.prod(c.output_grid_hw)),future=True,
                              patch_generator=patch,horizon_generator=horizon)
        ids=(tensor_digest(sample.patch_indices),tensor_digest(sample.future_pair))
        if token in sampler_ids and sampler_ids[token]!=ids: raise RuntimeError('Restored sample IDs changed')
        sampler_ids[token]=ids
        gather_current(item['current_dense_features'][None],item['current_valid'][None],sample)
        gather_future(item['future_dense_features'][None],item['future_valid'][None],sample)
        delta={k:provider.stats[k]-before[k] for k in ('ram_hits','immutable_hits','computed_images','evictions')}
        mixed |= delta['immutable_hits']>0 and delta['computed_images']>0
        if case=='partial_lru':partial |= delta['ram_hits']>0 and delta['computed_images']>0
        case_rows[case].append({'scene':token,'comparisons':comparisons,'all_masks_equal':True,
            'sampler_ids':list(ids),'counts':delta})
        progress()
    for token in plan['scene_ids']:
        provider.load_state_dict(provider.state_dict()); sampler_states[token]=(patch.get_state(),horizon.get_state())
        compare_scene(token,'cold'); compare_scene(token,'partial_lru')
        state=provider.state_dict(); provider.load_state_dict(state)
        if provider.memory_report()['resident_bytes']!=0: raise RuntimeError('Cold restoration retained tensors')
        compare_scene(token,'cold_restored')
    provider.load_state_dict(provider.state_dict())
    for token in reversed(plan['scene_ids']):compare_scene(token,'reordered')
    internal=[r['difference'] for r in repeated]+[x['difference'] for rows in case_rows.values() for r in rows for x in r['comparisons']]
    return {'reference_digests':reference_digests, 'reference_reordered4':repeated,
            'legacy_cached_reference':legacy,'cases':case_rows, 'all_masks_equal':True,'sampler_equal':True,
            'partial_lru_exercised':partial,'mixed_cache_compute_exercised':mixed,
            'fixed_policy_summary':shared.summarize(internal),
            'legacy_reference_summary':shared.summarize([r['difference'] for r in legacy]),
            'provider_memory':provider.memory_report()}



def _uint(value, name, *, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError('Composition requires an exact nonnegative integer: '+name)
    return value


def _full_difference(metric, expected_elements):
    fields = {'elements','bitwise_equal','max_abs','sum_abs','sum_squared',
              'reference_squared','rms','l2_relative','max_elementwise_relative'}
    if (not isinstance(metric,dict) or set(metric)!=fields or
            _uint(metric.get('elements'),'comparison elements',positive=True)!=expected_elements or
            metric.get('bitwise_equal') is not True):
        raise ValueError('Composition comparison must cover every expected tensor element')
    for name in fields-{'elements','bitwise_equal'}:
        value=metric[name]
        if type(value) not in (int,float) or not math.isfinite(value) or value<0:
            raise ValueError('Composition difference statistic is not finite nonnegative: '+name)
        if name!='reference_squared' and value!=0:
            raise ValueError('Composition comparison contradicts exact bitwise equality')


def _checked_summary(actual, metrics):
    expected=shared.summarize(metrics)
    if not isinstance(actual,dict) or set(actual)!=set(expected):
        raise ValueError('Composition summary schema differs from full comparisons')
    for name,value in actual.items():
        if name in ('comparisons','elements'):
            _uint(value,'summary '+name,positive=True)
        elif name=='bitwise_equal':
            if type(value) is not bool:raise ValueError('Composition summary bitwise flag must be boolean')
        elif type(value) not in (int,float) or not math.isfinite(value) or value<0:
            raise ValueError('Composition summary scalar is invalid: '+name)
    if actual!=expected:
        raise ValueError('Composition summary differs from recomputed full-tensor comparisons')


def validate_receipt(plan, report, descriptor, *, control_migration=None):
    """Admission requires measured composition evidence, never path rebinding alone."""
    validate_plan(plan, rebuild=False, control_migration=control_migration)
    prior, _ = prior_evidence(plan['prior_result'],plan['prior_plan'])
    if (plan['requests'] != descriptor['requests'] or plan['request_index'] != descriptor['request_index'] or
            descriptor['sources'] != [{'index':plan['source_index'],'cache_identity':plan['current_identity']}] or
            descriptor['teacher_config'] != plan['teacher_config'] or descriptor['execution'] != plan['execution'] or
            descriptor['limits'] != plan['limits'] or descriptor['provider'] != plan['provider'] or
            plan['execution'] != prior['execution']):
        raise ValueError('New population/source/teacher differs from actual indexed composition qualification')
    if (report.get('schema')!=RESULT_SCHEMA or not _signed(report) or report.get('plan_identity')!=plan['identity'] or
            report.get('status')!='INDEXED_COMPOSITION_BITWISE_QUALIFIED' or report.get('errors')!=[] or
            report.get('tensor_disk_bytes')!=0 or report.get('execution')!=plan['execution'] or
            report.get('execution_identity')!=plan['execution_identity'] or report.get('construction_rng_unchanged') is not True or
            report.get('completed_all_cases') is not True or report.get('fixed_policy_repeatability') is not True or
            report.get('legacy_reference_compatibility') is not True or report.get('all_masks_equal') is not True or
            report.get('sampler_equal') is not True or report.get('partial_lru_exercised') is not True or
            report.get('mixed_cache_compute_exercised') is not True or
            report.get('frozen_initial_versions_hash')!=report.get('frozen_final_versions_hash') or
            not report.get('frozen_initial_versions_hash')):
        raise ValueError('Indexed composition is incomplete, changed or numerically unqualified')
    expected_elements=math.prod(plan['contract']['output_grid_hw'])*plan['contract']['channels']
    if (set(report['reference_digests'])!=set(plan['keys']) or
            [r['key'] for r in report['reference_reordered4']]!=list(reversed(plan['keys']))):
        raise ValueError('Independent reference did not cover all exact new keys in the registered order')
    for row in report['reference_digests'].values():
        if set(row)!={'features','valid'}:raise ValueError('Independent reference digest schema changed')
        for value in row.values():require_hash(value,'independent reference tensor digest')
    legacy=report['legacy_cached_reference']
    if [r['key'] for r in legacy]!=plan['hit_keys']:
        raise ValueError('Legacy cached comparison is incomplete or reordered')
    if set(report['cases'])!=set(CASES):raise ValueError('Missing composition layout')
    totals={name:0 for name in ('ram_hits','immutable_hits','computed_images','evictions')}
    provider_calls=0
    for name,rows in report['cases'].items():
        expected=plan['scene_ids'][::-1] if name=='reordered' else plan['scene_ids']
        if [r['scene'] for r in rows]!=expected or sum(len(r['comparisons']) for r in rows)!=plan['slots']-plan['missing_slots']:
            raise ValueError('Incomplete scene/slot comparison')
        for row in rows:
            keys=plan['scene_keys'][row['scene']]
            if row['all_masks_equal'] is not True or [v['key'] for v in row['comparisons']]!=keys:
                raise ValueError('Unqualified masks/scene keys')
            if not isinstance(row['sampler_ids'],list) or len(row['sampler_ids'])!=2:
                raise ValueError('Missing patch/future-pair sample identities')
            for value in row['sampler_ids']:require_hash(value,'sample tensor digest')
            if not isinstance(row['counts'],dict) or set(row['counts'])!=set(totals):
                raise ValueError('Per-layout counter schema changed')
            counts=row['counts']
            for key,value in counts.items():
                totals[key]+=_uint(value,'layout '+key)
            hits=len(set(keys)&set(plan['hit_keys'])); misses=len(set(keys))-hits
            if (sum(counts[k] for k in ('ram_hits','immutable_hits','computed_images'))!=len(set(keys)) or
                    counts['computed_images']>misses or counts['immutable_hits']>hits or
                    counts['evictions']>len(set(keys))):
                raise ValueError('Per-layout counters do not cover the actual scene keys')
            if name in ('cold','cold_restored') and (counts['ram_hits']!=0 or
                    counts['computed_images']!=misses or counts['immutable_hits']!=hits):
                raise ValueError('Cold/restored counters contradict empty RAM and actual source coverage')
            provider_calls+=math.ceil(counts['computed_images']/8)
            for comparison in row['comparisons']:
                _full_difference(comparison['online_difference'],expected_elements)
    comparisons=legacy+report['reference_reordered4']+[c for rows in report['cases'].values() for r in rows for c in r['comparisons']]
    for row in comparisons:_full_difference(row['difference'],expected_elements)
    _checked_summary(report['legacy_reference_summary'],[r['difference'] for r in legacy])
    fixed_metrics=[r['difference'] for r in report['reference_reordered4']]+[
        c['difference'] for rows in report['cases'].values() for r in rows for c in r['comparisons']]
    _checked_summary(report['fixed_policy_summary'],fixed_metrics)
    for token in plan['scene_ids']:
        rows=[next(r for r in report['cases'][case] if r['scene']==token) for case in CASES]
        if any(r['sampler_ids']!=rows[0]['sampler_ids'] for r in rows):
            raise ValueError('Sampling changed across cached/cold/restored layouts')
    if (not any(r['counts']['ram_hits']>0 and r['counts']['computed_images']>0 for r in report['cases']['partial_lru']) or
            not any(r['counts']['immutable_hits']>0 and r['counts']['computed_images']>0 for rows in report['cases'].values() for r in rows)):
        raise ValueError('Claimed partial/mixed layout was not actually exercised')
    for field,limit in [('peak_rss_bytes',plan['budget']['ram_peak_bytes']),
                        ('cuda_peak_allocated_bytes',plan['budget']['gpu_peak_bytes'])]:
        if _uint(report.get(field),field)>limit:
            raise ValueError('Composition exceeded its measured resource budget: '+field)
    seconds=report.get('elapsed_seconds')
    if type(seconds) not in (int,float) or not math.isfinite(seconds) or not 0<=seconds<=plan['budget']['max_seconds']:
        raise ValueError('Composition elapsed time is invalid or outside budget')
    reads=_uint(report.get('process_rchar_bytes'),'process rchar bytes')
    mmap=_uint(report.get('tensor_mmap_logical_bytes'),'tensor mmap logical bytes')
    if reads+mmap>plan['budget']['max_logical_input_bytes']:
        raise ValueError('Exceeded logical input accounting')
    if _uint(report.get('tensor_disk_bytes'),'tensor disk bytes')!=0:
        raise ValueError('Composition may not write teacher tensors')
    calls=_uint(report.get('batch_calls'),'batch calls',positive=True)
    stats=report.get('compute_stats',{})
    if not isinstance(stats,dict) or set(stats)!={'calls','requested_images','compute_image_slots','padding_image_slots'}:
        raise ValueError('Fixed compute counter schema changed')
    for field,value in stats.items():_uint(value,'fixed compute '+field,positive=field!='padding_image_slots')
    expected_calls=math.ceil(len(plan['keys'])/8)+math.ceil(len(plan['keys'])/4)+provider_calls
    expected_requests=2*len(plan['keys'])+totals['computed_images']
    if (calls!=expected_calls or _uint(report.get('rng_forward_checks'),'RNG checks',positive=True)!=calls or
            stats['calls']!=calls or _uint(report.get('image_forwards'),'image forwards',positive=True)!=8*calls or
            stats['compute_image_slots']!=8*calls or stats['requested_images']!=expected_requests or
            stats['padding_image_slots']!=8*calls-expected_requests or not calls<=expected_requests<=8*calls or
            calls>plan['budget']['max_batch_calls'] or 8*calls>plan['budget']['max_image_forwards']):
        raise ValueError('Incomplete padded compute/RNG accounting')
    memory=report.get('provider_memory',{})
    for field,value in {**totals,'batches':provider_calls,'failures':0,'tensor_disk_bytes':0}.items():
        if _uint(memory.get(field),'provider '+field)!=value:
            raise ValueError('Provider totals differ from actual per-layout counters')
    limit=plan['limits']['max_cached_bytes']
    resident=_uint(memory.get('resident_bytes'),'provider resident bytes')
    peak=_uint(memory.get('peak_resident_bytes'),'provider peak resident bytes')
    if not resident<=peak<=limit or _uint(memory.get('resident_limit_bytes'),'provider resident limit')!=limit:
        raise ValueError('Provider RAM accounting exceeds its bound')


def run(plan_path, output):
    plan=tp.read_small(plan_path); validate_plan(plan,rebuild=False)
    budget=Budget(plan['budget'])
    validate_plan(plan);budget.check()
    output=Path(output)
    if output.exists():raise FileExistsError('Preserve prior attempts; choose a fresh output')
    output.mkdir(parents=True); shared.bounded_json(output/'RESOLVED_PLAN.json',plan,plan['budget']['metadata_bytes'])
    executor=None; indices=[];code=1
    result={'schema':RESULT_SCHEMA,'plan_identity':plan['identity'],'status':'STARTING','errors':[],
        'tensor_disk_bytes':0,'completed_all_cases':False,'fixed_policy_repeatability':False,
        'legacy_reference_compatibility':False,'formal_approved':False,'scientific_gain':'UNTESTED'}
    try:
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8')
        torch.set_num_threads(2)
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('Expose exactly one Lead-admitted GPU')
        device=torch.device('cuda:0');torch.cuda.init();torch.cuda.get_rng_state(device)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
        torch.use_deterministic_algorithms(True,warn_only=False)
        torch.backends.cuda.enable_flash_sdp(True);torch.backends.cuda.enable_mem_efficient_sdp(True);torch.backends.cuda.enable_math_sdp(True)
        req,src=open_pair(plan);indices=[req,src]
        c=TeacherFeatureContract.from_dict(plan['contract']);before=shared.rng_digest(device)
        budget.mmap_bytes+=plan['budget']['teacher_weight_mmap_bytes']
        executor=fixed.FixedBatchDINOExecutor(plan['teacher_root'],contract=c,variant=plan['variant'],device=str(device),
            expected_execution_identity=plan['execution_identity'])
        if before!=shared.rng_digest(device):raise RuntimeError('Teacher construction changed RNG')
        budget.check()
        result.update(execution=executor.execution,execution_identity=executor.execution_identity,
            construction_rng_unchanged=True,frozen_initial_versions_hash=identity_hash(executor._versions))
        reader=IndexedTeacherCacheReader(src,expected_identity=plan['current_identity'],expected_contract=c)
        checked=GuardedFixedExecutor(executor,budget,device)
        progress_count=0
        def progress():
            nonlocal progress_count
            progress_count+=1
            shared.bounded_json(output/'PROGRESS.json',{'plan_identity':plan['identity'],
                'status':'RUNNING','completed_scene_layouts':progress_count,'elapsed_seconds':time.monotonic()-budget.start,
                'batch_calls':budget.calls,'image_forwards':budget.images},plan['budget']['metadata_bytes'])
        result.update(qualify_layouts(plan,req,reader,checked,rng=lambda:shared.rng_digest(device),budget=budget,progress=progress))
        if identity_hash(executor._frozen_versions())!=result['frozen_initial_versions_hash']:
            raise ValueError('Teacher frozen state changed')
        result.update(completed_all_cases=True,
            fixed_policy_repeatability=result['fixed_policy_summary']['bitwise_equal'],
            legacy_reference_compatibility=result['legacy_reference_summary']['bitwise_equal'],
            frozen_final_versions_hash=identity_hash(executor._frozen_versions()),rng_forward_checks=checked.rng_checks)
        budget.check()
        good=all(result[k] for k in ('fixed_policy_repeatability','legacy_reference_compatibility','partial_lru_exercised','mixed_cache_compute_exercised'))
        result['status']='INDEXED_COMPOSITION_BITWISE_QUALIFIED' if good else 'NUMERICAL_OR_LAYOUT_REVIEW_REQUIRED';code=0 if good else 2
    except Exception as e:
        result['status']='PARTIAL_BUDGET' if isinstance(e,(TimeoutError,MemoryError)) else 'FAILED'
        result['errors'].append({'type':type(e).__name__,'message':str(e)})
        code=2 if result['status']=='PARTIAL_BUDGET' else 1
    finally:
        for index in indices:index.close()
        result.update(elapsed_seconds=time.monotonic()-budget.start,image_forwards=budget.images,batch_calls=budget.calls,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            process_rchar_bytes=_rchar()-budget.start_rchar,tensor_mmap_logical_bytes=budget.mmap_bytes)
        if executor is not None:result['compute_stats']=executor.compute_stats
        if torch.cuda.is_initialized():result.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        result['identity']=identity_hash(result);shared.bounded_json(output/'RESULT.json',result,plan['budget']['metadata_bytes'])
    return result,code


def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('plan',help='CPU metadata/stat only, no image/weight/tensor reads')
    for name in ('request-index','source-index','teacher-config','prior-result','output'):prepare.add_argument('--'+name,required=True)
    prepare.add_argument('--max-seconds',type=int,default=900)
    migration=sub.add_parser('prepare-profile-migration',help='Metadata-only explicit archived/control-source proposal; no approval')
    for name in ('parent-descriptor','parent-runtime','output'):migration.add_argument('--'+name,required=True)
    execute=sub.add_parser('run',help='One GPU finite qualifier, no training/cache build')
    execute.add_argument('--plan',required=True);execute.add_argument('--output',required=True)
    return p


def main():
    a=parser().parse_args()
    if a.command=='prepare-profile-migration':
        d=prepare_control_migration(a.parent_descriptor,a.parent_runtime,a.output)
        print(json.dumps({'status':'PROPOSED_NOT_APPROVED','identity':d['identity']}));return 0
    if a.command=='plan':
        p=make_plan(a.request_index,a.source_index,a.teacher_config,a.prior_result,max_seconds=a.max_seconds)
        out=Path(a.output)
        if out.exists():raise FileExistsError('Preserve prior plan directory')
        out.mkdir(parents=True);shared.bounded_json(out/'PLAN.json',p,p['budget']['metadata_bytes'])
        print(json.dumps({'status':'PLANNED_NOT_RUN','identity':p['identity'],'budget':p['budget']}));return 0
    result,code=run(a.plan,a.output);print(json.dumps({k:result[k] for k in ('status','errors','elapsed_seconds')}));return code


if __name__=='__main__':
    try:code=main()
    except Exception as e:
        print(json.dumps({'status':'FAILED','error':type(e).__name__,'message':str(e)}));code=1
    raise SystemExit(code)
