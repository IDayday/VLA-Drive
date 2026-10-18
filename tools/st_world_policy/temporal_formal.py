"""Fixed full-population T1--T4 metadata preparation, never a launch approval.

Reuses the original indexed provider, PCA, history reader and native trainer.
The separate FORMAL descriptor does not promote or relax a PROFILE descriptor.
Actual dispatch still requires the unchanged independent formal launch gate.
"""
from copy import deepcopy
from pathlib import Path

SCHEMA = 'st_temporal_full_formal5k_admission_v1'
PROFILE_SCHEMA = 'st_temporal_full_global32_profile_admission_v1'
POPULATION = 101592
CELLS = {'T0': (False, False, False), 'T1': (True, False, False), 'T2': (False, True, False),
         'T3': (True, True, False), 'T4': (True, True, True)}
DECISION_SHA = '6aab4626386b17dc666f2452413e14007a1e50d90e06052161cb5819fc931081'
FUTURE_WEIGHT = 0.9954445849209375
DELTA_WEIGHT = 1.6396291934956382
FIELDS = {'schema', 'scope', 'cell', 'decision', 'implementation', 'parent_descriptor',
          'source_migration', 'data_manifest', 'recipe', 'reference_recipe',
          'max_scenes', 'max_updates', 'population', 'consumption', 'history',
          'new_population_tensor_qualified', 'formal_approved'}


def consumption(updates=5000, *, world_size=4, micro_batch=1, scope='FORMAL', timing=False):
    limit = 5000 if scope == 'FORMAL' else 256
    if (scope not in ('FORMAL','PROFILE') or type(updates) is not int or not 1 <= updates <= limit
            or type(world_size) is not int or world_size not in (1,2,4,8)
            or type(micro_batch) is not int or micro_batch not in (1,2,4)
            or 32 % (world_size * micro_batch)
            or (scope == 'PROFILE' and updates != (240 if timing else 4))):
        raise ValueError('Explicit global32 topology and finite FORMAL5k/PROFILE4-or-240 required')
    from tools.foresight.student_state import validate_rank_batches
    validate_rank_batches(POPULATION, 32, world_size, micro_batch)
    epochs, remainder = divmod(updates, (POPULATION + 31) // 32)
    tail = min(remainder * 32, POPULATION)
    return dict(schema='st_temporal_formal5k_consumption_v1', population=POPULATION,
        initial_completed_updates=0, initial_epoch=0, initial_offset=0,
        cumulative_optimizer_updates=updates, cumulative_scene_exposures=epochs * POPULATION + tail,
        world_size=world_size, micro_batch=micro_batch, global_batch=32,
        gradient_accumulation_steps=32 // (world_size * micro_batch),
        per_rank_scene_exposures=[epochs * len(range(r, POPULATION, world_size)) + len(range(r, tail, world_size)) for r in range(world_size)],
        resume_budget='fresh_complete_windows_no_resume' if timing else 'same_run_cumulative_not_per_invocation',
        diagnostic_weights_allowed=False)


def selected_recipe(reference, cell, *, output_root, history_root, max_seconds=28800,
                    scope='FORMAL', world_size=4, micro_batch=1, profile_mode='none'):
    """The only permitted changes from the original T2 numerical fixture."""
    from starVLA.model.modules.st_world.config import SpaceTimeConfig
    from tools.st_world_policy.official_runtime import REFERENCE_RECIPE
    if (cell not in CELLS or type(max_seconds) is not int
            or not 1 <= max_seconds <= (28800 if scope == 'FORMAL' else 3600)
            or profile_mode not in ('none','timing') or (scope == 'FORMAL' and profile_mode != 'none')):
        raise ValueError('Explicit T1--T4 cell and finite <=8h invocation required')
    st = SpaceTimeConfig.from_dict(reference['space_time'])
    if (st.cell != 'T2' or st.spatial_parent != 'S-P2' or st.teacher.channels != 384
            or st.world_queries_per_view != 48 or st.sampled_positions != 96
            or st.teacher.image_hw != (288, 384) or st.teacher.output_grid_hw != (18, 24)
            or reference['execution']['max_updates'] != 20000
            or reference['optimizer']['warmup'] != 1000
            or reference['execution']['seed'] != 42 or 'continuation' in reference):
        raise ValueError('Reference differs from fixed SP2/PCA384/original 20k recipe')
    updates = 5000 if scope == 'FORMAL' else (240 if profile_mode == 'timing' else 4)
    consumption(updates, world_size=world_size, micro_batch=micro_batch, scope=scope, timing=profile_mode=='timing')
    history, future, delta = CELLS[cell]
    if history != (history_root is not None) or not Path(output_root).is_absolute():
        raise ValueError('History role or isolated output is unresolved')
    cfg = deepcopy(reference)
    cfg['execution'].update(scope=scope, family='temporal', sample_limit=0 if scope=='FORMAL' else POPULATION,
        stop_after=updates, world_size=world_size, micro_batch=micro_batch, global_batch=32)
    cfg['space_time'] = st.to_dict()
    cfg['space_time'].update(cell=cell, history=history, future=future, delta=delta,
        future_weight=FUTURE_WEIGHT if future else 0., delta_weight=DELTA_WEIGHT if delta else 0.)
    cfg['data'].pop('teacher_provider', None)
    cfg['data'].update(teacher_root=None, history_root=history_root)
    cfg['runtime'].update(output_root=str(output_root), loader_workers=0,
        save_every=1000 if scope=='FORMAL' else updates, max_seconds=max_seconds, profile_mode=profile_mode)
    effective = {**{k:cfg['execution'][k] for k in ('world_size','micro_batch','global_batch','max_updates')},
                 **cfg['optimizer'], 'gradient_checkpointing':cfg['runtime']['gradient_checkpointing']}
    cfg['runtime']['official_execution'].update(
        declared_recipe_overrides=sorted(k for k,v in REFERENCE_RECIPE.items() if effective[k] != v),
        adaptation_reason='Lead fixed T1-T4 full-train 5k screen: shared SP2/PCA384/W48/K96, '
            'explicit reviewed topology/global32, original20k/warmup1000, bounded synchronous '
            'fixed8 FP32 teacher and unchanged BF16/FA2/FP32-master/ZeRO2 student. '
            'Fresh common Base initialization; no diagnostic checkpoint. Independent launch reviews required.')
    return cfg


def validate_recipe(a, cfg):
    from tools.st_world_policy import teacher_provider as tp
    from tools.st_world_policy.temporal_composition import history_binding
    from tools.st_world_policy.training_config import official_runtime_policy
    if (not isinstance(a, dict) or set(a) != FIELDS
            or (a['schema'],a['scope']) not in ((SCHEMA,'FORMAL'),(PROFILE_SCHEMA,'PROFILE'))
            or a['cell'] not in CELLS or a['max_scenes'] != POPULATION
            or a['max_updates'] != (5000 if a['scope']=='FORMAL' else 256)
            or a['new_population_tensor_qualified'] is not False or a['formal_approved'] is not False
            or a['implementation'] != tp.pin(__file__) or a['decision']['sha256'] != DECISION_SHA):
        raise ValueError('Unknown/stale explicit full temporal FORMAL admission')
    decision = tp.pinned(a['decision'])
    if (decision.get('status') != 'RECIPE_SELECTED_NOT_FORMAL_DISPATCH'
            or decision.get('weights') != dict(current=1., future=FUTURE_WEIGHT, delta=DELTA_WEIGHT)):
        raise ValueError('Lead common temporal recipe changed')
    original = tp.pinned(a['reference_recipe'])
    if (original['execution']['scope'] != 'SMOKE' or original['execution']['stop_after'] > 4
            or original['data'].get('teacher_provider', {}).get('descriptor') != a['parent_descriptor']):
        raise ValueError('Formal input must preserve original exact32 qualification reference')
    need_history = CELLS[a['cell']][0]
    history = a['history']; history_root = None
    if need_history:
        if not isinstance(history, dict) or set(history) != {'history_complete','history_identity','history_manifest_sha256'}:
            raise ValueError('History cell requires its actual immutable full-population receipt')
        done = history_binding(history['history_complete'])
        if (done['scenes'] != POPULATION or done['history_identity'] != history['history_identity']
                or done['history_manifest_sha256'] != history['history_manifest_sha256']):
            raise ValueError('History COMPLETE does not cover the fixed full train population')
        history_root = str(Path(history['history_complete']['path']).parent)
    elif history is not None:
        raise ValueError('T0/T2 cannot introduce causal history')
    expected = selected_recipe(original, a['cell'], output_root=cfg['runtime']['output_root'],
        history_root=history_root, max_seconds=cfg['runtime']['max_seconds'], scope=a['scope'],
        world_size=a['consumption']['world_size'], micro_batch=a['consumption']['micro_batch'],
        profile_mode=cfg['runtime'].get('profile_mode','none'))
    # Fresh registration can stop earlier only by a runtime safety pause. Recipe
    # remains exactly 5k; resume carries the consumed cursor inside that ceiling.
    for value in (cfg, tp.pinned(a['recipe'])):
        normalized = deepcopy(value); normalized['data'].pop('teacher_provider', None)
        if normalized != expected:
            raise ValueError('FORMAL changed unregistered recipe, cell, topology, precision or 5k/20k budget')
    if (a['consumption'] != consumption(expected['execution']['stop_after'], scope=a['scope'],
            world_size=expected['execution']['world_size'], micro_batch=expected['execution']['micro_batch'],
            timing=expected['runtime']['profile_mode']=='timing') or cfg['runtime']['deterministic'] is not True
            or cfg['runtime']['gradient_checkpointing'] is not False):
        raise ValueError('FORMAL consumption/backend changed')
    official_runtime_policy(cfg)
    return expected


def qualified_execution(d, manifest, scene_count):
    from tools.st_world_policy import teacher_provider as tp
    from starVLA.dataloader.st_world_schema import TeacherFeatureContract
    if d.get('schema') != tp.INDEXED_SCHEMA or 'profile_admission' in d:
        raise ValueError('FORMAL cannot reuse a diagnostic admission')
    a = d.get('formal_admission', d.get('temporal_profile_admission'))
    if ('formal_admission' in d) != (a['scope']=='FORMAL'):
        raise ValueError('Temporal descriptor field and explicit scope differ')
    validate_recipe(a, tp.pinned(a['recipe']))
    parent, plan = tp._qualified_parent_execution(d, a)
    original = tp.pinned(parent['request_index']['complete'])['manifest_header']
    for key in ('split','source_student_identity','source_index_sha256','partition_sha256','times_s','view_names'):
        if manifest.get(key) != original.get(key):
            raise ValueError('FORMAL changed original source/partition/time/view lineage: ' + key)
    if scene_count != POPULATION or manifest['split'] != 'train':
        raise ValueError('FORMAL requires the complete original train population')
    return tp.pinned(d['teacher_config']), TeacherFeatureContract.from_dict(d['provider']['contract']), plan


def _prepare_staged(args, out):
    from tools.st_world_policy import teacher_provider as tp
    from tools.st_world_policy.temporal_composition import history_binding, validate_dataset_attachment
    from tools.st_world_policy.provider_pca import register_provider_scope
    from tools.st_world_policy.training_config import validate_config
    from starVLA.dataloader.st_world_history import HistoryReader
    from starVLA.dataloader.st_world_targets import atomic_json
    template = tp.pinned(tp.pin(args.full_profile_descriptor))
    if (template.get('identity') != tp.identity_hash({k:v for k,v in template.items() if k != 'identity'})
            or template.get('schema') != tp.INDEXED_SCHEMA or 'formal_admission' in template
            or 'temporal_profile_admission' in template
            or not tp.is_full_profile(template.get('profile_admission'))):
        raise ValueError('Supply the existing complete full-index descriptor, not a subset/SMOKE')
    prior = template['profile_admission']
    need_history = CELLS[args.cell][0]; history = None
    if need_history:
        if not args.history_complete: raise ValueError('Missing real full history COMPLETE')
        hp = tp.pin(args.history_complete); done = history_binding(hp)
        history = dict(history_complete=hp, history_identity=done['history_identity'],
                       history_manifest_sha256=done['history_manifest_sha256'])
    elif args.history_complete:
        raise ValueError('T2 must not attach history metadata')
    original = tp.pinned(prior['reference_recipe'])
    cfg = selected_recipe(original, args.cell, output_root=args.training_output_root,
        history_root=str(Path(args.history_complete).parent) if need_history else None,
        max_seconds=args.max_seconds, scope=args.scope, world_size=args.world_size,
        micro_batch=args.micro_batch, profile_mode=args.profile_mode)
    counts=consumption(cfg['execution']['stop_after'], scope=args.scope, world_size=args.world_size,
        micro_batch=args.micro_batch, timing=args.profile_mode=='timing')
    recipe_path = out / 'RECIPE.json'; atomic_json(recipe_path, cfg, immutable=True)
    a = dict(schema=SCHEMA if args.scope=='FORMAL' else PROFILE_SCHEMA, scope=args.scope, cell=args.cell, decision=tp.pin(args.decision),
        implementation=tp.pin(__file__), parent_descriptor=prior['parent_descriptor'],
        source_migration=tp.pin(args.source_migration), data_manifest=prior['data_manifest'],
        recipe=tp.pin(recipe_path), reference_recipe=prior['reference_recipe'],
        max_scenes=POPULATION, max_updates=5000 if args.scope=='FORMAL' else 256, population=prior['population'],
        consumption=counts, history=history, new_population_tensor_qualified=False, formal_approved=False)
    validate_recipe(a, cfg)
    d = deepcopy(template); del d['profile_admission']
    d['formal_admission' if args.scope=='FORMAL' else 'temporal_profile_admission'] = a
    d['identity'] = tp.identity_hash({k:v for k,v in d.items() if k != 'identity'})
    _, _, preview = tp.validate_descriptor(d)
    try:
        from starVLA.dataloader.st_world_dataset import projected_teacher_contract
        from starVLA.model.modules.st_world.config import SpaceTimeConfig
        pca = tp.read_small(Path(cfg['data']['pca_path']) / 'PCA.json')
        if (384 not in pca['dimensions'] or
                projected_teacher_contract(preview.contract, pca_identity=pca['identity'], dimension=384)
                != SpaceTimeConfig.from_dict(cfg['space_time']).teacher):
            raise ValueError('Formal target differs from the same frozen PCA384 basis')
        if need_history:
            reader = HistoryReader(cfg['data']['history_root'], history['history_identity'])
            source = dict(split=preview.manifest['split'], identity=preview.manifest['source_student_identity'],
                index_sha256=preview.manifest['source_index_sha256'], partition_sha256=preview.manifest['partition_sha256'])
            validate_dataset_attachment(reader, source, tp.pinned(a['data_manifest'])['scenes'], preview)
    finally:
        tp.close_provider_metadata(preview)
    descriptor = out / 'PROVIDER.json'; atomic_json(descriptor, d, immutable=True)
    scope = out / 'PROVIDER_CHANNEL_SCOPE.json'
    register_provider_scope(cfg['data']['pca_path'], args.pca_fit_cache_root, tp.pin(descriptor), scope,
                            fit_cache_index=args.pca_fit_cache_index)
    cfg['data']['teacher_provider'] = dict(descriptor=tp.pin(descriptor), pca_channel_scope=tp.pin(scope))
    validate_config(cfg)
    atomic_json(out / 'training_config.json', cfg, immutable=True)
    atomic_json(out / 'TARGET_CONTRACT.json', cfg['space_time']['teacher'], immutable=True)
    return dict(config=tp.pin(out / 'training_config.json'), scope=args.scope, updates=cfg['execution']['stop_after'],
        consumption=counts, data_manifest=a['data_manifest'], teacher_manifest=tp.pin(descriptor),
        target_contract=tp.pin(out / 'TARGET_CONTRACT.json'), training_started=False, formal_approved=False,
        new_teacher_tensor_bytes=0, diagnostic_weights_allowed=False)


def prepare(args):
    """Single immutable metadata version, reusing the existing atomic pattern."""
    import os
    import tempfile
    from tools.st_world_policy import teacher_provider as tp
    from starVLA.dataloader.st_world_targets import atomic_json, file_lock
    out = Path(args.output).absolute(); out.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(out.with_name('.' + out.name + '.prepare.lock'), blocking=False):
        inputs = {key:tp.pin(getattr(args, key)) for key in
            ('full_profile_descriptor','source_migration','decision')}
        inputs.update(history_complete=tp.pin(args.history_complete) if args.history_complete else None,
            cell=args.cell, scope=args.scope, world_size=args.world_size, micro_batch=args.micro_batch,
            profile_mode=args.profile_mode, max_seconds=args.max_seconds, training_output_root=args.training_output_root,
            pca_fit_cache_root=args.pca_fit_cache_root, pca_fit_cache_index=args.pca_fit_cache_index,
            implementation=tp.pin(__file__))
        if out.exists() or out.is_symlink():
            done = tp.pinned(tp.pin(out / 'PREPARATION_COMPLETE.json'))
            if done.get('schema') != 'st_temporal_formal5k_preparation_v1' or done.get('inputs') != inputs:
                raise ValueError('Existing immutable preparation has different inputs')
            for item in done['artifacts']: tp.pinned(item)
            from tools.st_world_policy.training_config import validate_config
            validate_config(tp.pinned(done['result']['config']))
            return done['result']
        stage = Path(tempfile.mkdtemp(prefix='.' + out.name + '.staging-', dir=out.parent))
        published = False
        try:
            atomic_json(stage / 'PREPARATION_INPUTS.json', inputs, immutable=True)
            result = _prepare_staged(args, stage)
            artifacts = [tp.pin(stage / name) for name in ('RECIPE.json', 'PROVIDER.json',
                'PROVIDER_CHANNEL_SCOPE.json', 'TARGET_CONTRACT.json', 'training_config.json')]
            atomic_json(stage / 'PREPARATION_COMPLETE.json', dict(
                schema='st_temporal_formal5k_preparation_v1', inputs=inputs, artifacts=artifacts, result=result), immutable=True)
            fd = os.open(stage, os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)
            os.symlink(stage.name, out); published = True
            fd = os.open(out.parent, os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)
            return result
        except BaseException as error:
            if not published:
                atomic_json(stage / 'PREPARATION_FAILED.json', dict(status='FAILED_NOT_PUBLISHED',
                    error_type=type(error).__name__, error=str(error), training_started=False), immutable=True)
            raise


def main():
    import argparse
    import json
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('full-profile-descriptor','source-migration','decision','pca-fit-cache-root',
                 'pca-fit-cache-index','output','training-output-root'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--history-complete')
    p.add_argument('--scope',choices=('FORMAL','PROFILE'),required=True)
    p.add_argument('--world-size',type=int,choices=(1,2,4,8),required=True)
    p.add_argument('--micro-batch',type=int,choices=(1,2,4),required=True)
    p.add_argument('--profile-mode',choices=('none','timing'),default='none')
    p.add_argument('--cell', choices=tuple(CELLS), required=True)
    p.add_argument('--max-seconds', type=int, choices=range(1,28801), metavar='1..28800', default=28800)
    args = p.parse_args()
    for key, value in vars(args).items():
        if key not in ('cell','max_seconds','scope','world_size','micro_batch','profile_mode') and value is not None and not Path(value).is_absolute():
            p.error('All dependency/output paths must be absolute')
    if Path(args.output).resolve() == Path(args.training_output_root).resolve():
        p.error('Preparation and training checkpoint namespaces must be distinct')
    print(json.dumps(prepare(args), sort_keys=True))


if __name__ == '__main__':
    main()
