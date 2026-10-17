"""Explicit, <=4-update T3/T4 integration on the existing indexed provider.

This is a diagnostic composition, not a formal training or scientific-weight
admission. T2 validation is called unchanged after reversing exactly the
declared history/cell/delta differences. No new encoder, target or trainer.
"""
from copy import deepcopy
from pathlib import Path

SCHEMA = 'st_indexed_temporal_composition_diagnostic_v1'
FIELDS = {'schema', 'cell', 'parent_recipe', 'history_complete',
          'history_identity', 'history_manifest_sha256', 'delta_weight'}


def _validate_exact_population(admission, parent):
    """Composition retains the original qualified <=32 ordered population."""
    from tools.st_world_policy import teacher_provider as tp
    n = parent['execution']['sample_limit']
    if (admission.get('schema') != tp.PROFILE_SCHEMA or type(n) is not int
            or not 1 <= n <= 32):
        raise ValueError('Temporal composition requires non-full exact population <=32')
    original = tp.pinned(admission['reference_recipe'])
    d = tp.pinned(admission['parent_descriptor'])
    if (original['execution']['sample_limit'] != n or d.get('schema') != tp.INDEXED_SCHEMA
            or 'profile_admission' in d):
        raise ValueError('Composition population differs from original exact qualification')
    plan = tp.pinned(d['qualification_plan'])
    if (plan.get('schema') != 'st_indexed_composition_plan_v1'
            or plan.get('scenes') != n
            or plan.get('request_index') != d['request_index']
            or plan.get('requests') != d['requests']
            or plan.get('training_admission') != {'max_scenes':32, 'max_updates':4,
                'exact_request_identity':d['request_index']['request_identity']}):
        raise ValueError('Composition must retain its qualified request identity and count')
    request = tp.pinned(d['requests'])
    selected = tp.pinned(admission['data_manifest'])
    rows = selected.get('scenes', [])
    request_rows = request.get('scenes', [])
    if (request.get('identity') != d['request_index']['request_identity']
            or selected.get('schema') != 'st_fixed_scene_list_v1'
            or selected.get('split') != request.get('split') or selected.get('split') != 'train'
            or selected.get('source_index_hash') != request.get('source_index_sha256')
            or len(rows) != n or len(request_rows) != n
            or [r['token'] for r in rows] != plan.get('scene_ids')
            or len({r['token'] for r in rows}) != n
            or [(r['token'], r['log']) for r in rows] !=
                [(r['token'], r['log']) for r in request_rows]):
        raise ValueError('Composition cannot replace, reorder or expand the qualified population')


def history_binding(complete_pin):
    from tools.st_world_policy import teacher_provider as tp
    done = tp.pinned(complete_pin)
    if Path(complete_pin['path']).name != 'COMPLETE.json' or done.get('status') != 'COMPLETE':
        raise ValueError('Temporal composition requires actual history COMPLETE')
    for key in ('history_identity', 'history_manifest_sha256'):
        tp.require_hash(done.get(key), key)
    if type(done.get('scenes')) is not int or done['scenes'] < 1:
        raise ValueError('History COMPLETE lacks its real population')
    return done


def validate_composition_recipe(admission, cfg):
    from tools.st_world_policy import teacher_provider as tp
    c = admission['temporal_composition']
    if (not isinstance(c, dict) or set(c) != FIELDS or c['schema'] != SCHEMA
            or c['cell'] not in ('T3', 'T4')):
        raise ValueError('Unknown explicit temporal composition')
    parent_admission = deepcopy(admission)
    parent_admission.pop('temporal_composition')
    parent_admission['recipe'] = c['parent_recipe']
    parent = tp.pinned(c['parent_recipe'])
    _validate_exact_population(parent_admission, parent)
    # This retains the old current-only guard, source migration, numerical
    # recipe, full population and consumption checks. No recursive exception.
    tp.validate_profile_recipe(parent_admission, parent)
    if (parent['space_time']['cell'] != 'T2' or parent['space_time']['history']
            or not parent['space_time']['future'] or parent['space_time']['delta']
            or parent['execution']['stop_after'] > 4
            or parent['runtime'].get('profile_mode', 'none') != 'none'
            or 'continuation' in parent):
        raise ValueError('Composition requires fresh T2 diagnostic <=4 updates')
    done = history_binding(c['history_complete'])
    if (done['history_identity'] != c['history_identity']
            or done['history_manifest_sha256'] != c['history_manifest_sha256']
            or done['scenes'] < parent['execution']['sample_limit']):
        raise ValueError('History receipt identity/population differs')
    # Unit delta weight is an explicit engineering diagnostic, never a claimed
    # calibrated scientific coefficient. Future weight/teacher width stay fixed.
    delta = c['cell'] == 'T4'
    if type(c['delta_weight']) not in (int, float) or c['delta_weight'] != float(delta):
        raise ValueError('Diagnostic delta coefficient must be explicit zero/one')
    expected = deepcopy(parent)
    expected['space_time'].update(cell=c['cell'], history=True, delta=delta,
                                  delta_weight=c['delta_weight'])
    expected['data']['history_root'] = str(Path(c['history_complete']['path']).parent)
    reference = tp.pinned(admission['recipe'])
    if (tp.profile_recipe_identity(reference) != tp.profile_recipe_identity(expected)
            or tp.profile_recipe_identity(cfg) != tp.profile_recipe_identity(expected)):
        raise ValueError('Temporal composition changed undeclared recipe/input/budget')
    return reference


def validate_dataset_attachment(history, source, rows, provider):
    """Check the already loaded HistoryReader; no extra record/image/DB reads."""
    c = getattr(provider, 'temporal_composition', None)
    if c is None:
        return  # Original T2 and legacy SMOKE semantics remain unchanged.
    if history is None:
        raise ValueError('Temporal provider requires its bound causal history')
    done = history_binding(c['history_complete'])
    manifest = history.manifest
    if (history.root.resolve() != Path(c['history_complete']['path']).parent.resolve()
            or done['history_identity'] != c['history_identity']
            or done['history_manifest_sha256'] != c['history_manifest_sha256']
            or manifest['identity'] != c['history_identity']
            or done['scenes'] != len(manifest['scenes'])
            or manifest.get('schema') != 'st_history_manifest_v1'
            or manifest.get('view_names') != ['CAM_F0', 'CAM_L0', 'CAM_R0']):
        raise ValueError('Loaded history differs from temporal input binding')
    for key, source_key in (('split', 'split'), ('source_student_identity', 'identity'),
                            ('source_index_sha256', 'index_sha256'),
                            ('partition_sha256', 'partition_sha256')):
        if manifest.get(key) != source[source_key] or manifest.get(key) != provider.manifest.get(key):
            raise ValueError('History/current/future source or role differs: ' + key)
    for row in rows:
        past = history.index.get(row['token'])
        if (past is None or past['log'] != row['log']
                or past['current_record_sha256'] != row['current_record_sha256']):
            raise ValueError('History scene/log/current identity differs from selected population')
    # Existing HistoryReader.read validates t<0, camera/time, same-log record
    # hashes and exact current-record identity per consumed scene.


def _prepare_staged(args, out):
    from tools.st_world_policy import teacher_provider as tp
    from tools.st_world_policy.provider_pca import register_provider_scope
    from tools.st_world_policy.training_config import validate_config
    from starVLA.dataloader.st_world_targets import atomic_json
    from starVLA.dataloader.st_world_history import HistoryReader
    parent = tp.pinned(tp.pin(args.parent_descriptor))
    if parent.get('schema') != tp.INDEXED_SCHEMA or 'profile_admission' in parent:
        raise ValueError('Initial composition requires original exact32 indexed qualification')
    cfg = deepcopy(tp.pinned(tp.pin(args.reference_config)))
    if (cfg['execution']['scope'] != 'SMOKE' or cfg['execution']['sample_limit'] > 32
            or cfg['data'].get('teacher_provider', {}).get('descriptor') != tp.pin(args.parent_descriptor)):
        raise ValueError('Reference must be exact32 parent SMOKE, not a different Base/population')
    cfg['execution'].update(scope='PROFILE', stop_after=4)
    cfg['runtime'].update(output_root=str(Path(args.training_output_root).resolve()),
                          max_seconds=args.max_seconds, save_every=4, profile_mode='none', loader_workers=0)
    cfg['data'].pop('teacher_provider', None)
    cfg['data']['teacher_root'] = None
    base_recipe = out / 'T2_REFERENCE.json'
    atomic_json(base_recipe, cfg, immutable=True)
    a = dict(schema=tp.PROFILE_SCHEMA, scope='PROFILE', parent_descriptor=tp.pin(args.parent_descriptor),
             source_migration=tp.pin(args.source_migration), data_manifest=tp.pin(args.data_manifest),
             recipe=tp.pin(base_recipe), reference_recipe=tp.pin(args.reference_config),
             max_scenes=256, max_updates=256, new_population_tensor_qualified=False, formal_approved=False)
    _validate_exact_population(a, cfg)
    tp.validate_profile_recipe(a, cfg)
    hp = tp.pin(args.history_complete); done = history_binding(hp)
    c = dict(schema=SCHEMA, cell=args.cell, parent_recipe=tp.pin(base_recipe), history_complete=hp,
             history_identity=done['history_identity'], history_manifest_sha256=done['history_manifest_sha256'],
             delta_weight=float(args.cell == 'T4'))
    cfg['space_time'].update(cell=args.cell, history=True, delta=args.cell == 'T4', delta_weight=c['delta_weight'])
    cfg['data']['history_root'] = str(Path(hp['path']).parent)
    recipe = out / (args.cell + '_RECIPE.json'); atomic_json(recipe, cfg, immutable=True)
    a.update(recipe=tp.pin(recipe), temporal_composition=c)
    d = deepcopy(parent); d['profile_admission'] = a
    d['identity'] = tp.identity_hash({k:v for k,v in d.items() if k != 'identity'})
    _, _, preview = tp.validate_descriptor(d)
    try:
        history = HistoryReader(Path(hp['path']).parent, c['history_identity'])
        rows = tp.pinned(a['data_manifest'])['scenes']
        source = {'split':preview.manifest['split'], 'identity':preview.manifest['source_student_identity'],
                  'index_sha256':preview.manifest['source_index_sha256'], 'partition_sha256':preview.manifest['partition_sha256']}
        validate_dataset_attachment(history, source, rows, preview)
    finally:
        tp.close_provider_metadata(preview)
    descriptor = out / 'PROVIDER.json'; atomic_json(descriptor, d, immutable=True)
    scope = out / 'PROVIDER_CHANNEL_SCOPE.json'
    register_provider_scope(cfg['data']['pca_path'], args.pca_fit_cache_root, tp.pin(descriptor), scope,
                            fit_cache_index=args.pca_fit_cache_index)
    cfg['data']['teacher_provider'] = {'descriptor':tp.pin(descriptor), 'pca_channel_scope':tp.pin(scope)}
    validate_config(cfg)
    atomic_json(out / 'training_config.json', cfg, immutable=True)
    return {'config':tp.pin(out / 'training_config.json'), 'scope':'PROFILE', 'updates':4,
            'training_started':False, 'formal_approved':False, 'scientific_weights_calibrated':False}


def prepare(args):
    """Publish one validated immutable version; failed staging remains inspectable.

    Pins refer to the stable version directory, so the single atomic output
    symlink publishes all artifacts without rewriting any dependent hash.
    """
    import os
    import tempfile
    from tools.st_world_policy import teacher_provider as tp
    from starVLA.dataloader.st_world_targets import atomic_json, file_lock
    from tools.st_world_policy.training_config import validate_config
    out = Path(args.output).absolute()  # Do not resolve an already published link.
    out.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(out.with_name('.' + out.name + '.prepare.lock'), blocking=False):
        def inputs():
            return {**{key:tp.pin(getattr(args, key)) for key in
                ('parent_descriptor', 'reference_config', 'data_manifest',
                 'history_complete', 'source_migration')},
                **{key:str(Path(getattr(args, key)).resolve()) for key in
                   ('pca_fit_cache_root', 'pca_fit_cache_index', 'training_output_root')},
                'cell':args.cell, 'max_seconds':args.max_seconds}
        if out.exists() or out.is_symlink():
            done = tp.pinned(tp.pin(out / 'PREPARATION_COMPLETE.json'))
            if (done.get('schema') != 'st_temporal_composition_preparation_v1'
                    or done.get('inputs') != inputs()):
                raise ValueError('Existing temporal preparation has different immutable inputs')
            for item in done['artifacts']:
                tp.pinned(item)
            validate_config(tp.pinned(done['result']['config']))
            return done['result']
        stage = Path(tempfile.mkdtemp(prefix='.' + out.name + '.staging-', dir=out.parent))
        published = False
        try:
            request = inputs()
            atomic_json(stage / 'PREPARATION_INPUTS.json', request, immutable=True)
            result = _prepare_staged(args, stage)
            artifacts = [tp.pin(stage / name) for name in ('T2_REFERENCE.json',
                args.cell + '_RECIPE.json', 'PROVIDER.json', 'PROVIDER_CHANNEL_SCOPE.json',
                'training_config.json')]
            atomic_json(stage / 'PREPARATION_COMPLETE.json', dict(
                schema='st_temporal_composition_preparation_v1', inputs=request,
                artifacts=artifacts, result=result), immutable=True)
            fd = os.open(stage, os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)
            os.symlink(stage.name, out)  # Exclusive atomic publication; never replaces out.
            published = True
            fd = os.open(out.parent, os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)
            return result
        except BaseException as error:
            if not published:
                atomic_json(stage / 'PREPARATION_FAILED.json', dict(
                    status='FAILED_NOT_PUBLISHED', error_type=type(error).__name__,
                    error=str(error), training_started=False), immutable=True)
            raise


def main():
    import argparse
    import json
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('parent-descriptor', 'reference-config', 'data-manifest', 'history-complete',
                 'source-migration', 'pca-fit-cache-root', 'pca-fit-cache-index', 'output', 'training-output-root'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--cell', choices=('T3', 'T4'), required=True)
    p.add_argument('--max-seconds', type=int, choices=range(1, 3601), metavar='1..3600', required=True)
    args = p.parse_args()
    if not all(Path(getattr(args, key)).is_absolute() for key in vars(args) if key not in ('cell', 'max_seconds')):
        p.error('All dependencies and output namespaces must be explicit absolute paths')
    if Path(args.output).resolve() == Path(args.training_output_root).resolve():
        p.error('Metadata output and training checkpoint namespace must be separate')
    print(json.dumps(prepare(args), sort_keys=True))


if __name__ == '__main__':
    main()
