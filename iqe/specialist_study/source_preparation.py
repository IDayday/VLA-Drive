"""Cold preparation for another Base, reusing the existing IQE context pipeline."""
from dataclasses import asdict
from pathlib import Path
from iqe.contracts import require
from iqe.io import atomic_json, read_json, file_hash


def source_config(config):
    from iqe.internal_baseline.config import Config
    value = Config(experiment_id=config.campaign_id + '_context_source', output_root=config.reuse_root,
                   contract=config.contract, base_run=config.base_run, checkpoint=config.base_checkpoint,
                   scene_manifest=config.scene_manifest, fit_limit='all_fit', val_limit=1024, cal_limit=1024,
                   research_protocol='two_candidate_v1', scorer_dropout=0., diagnostic_updates=200,
                   selection_seed=config.selection_seed, partition_seed=config.partition_seed,
                   scoring_workers=config.scoring_workers, scoring_attempts=config.scoring_attempts,
                   scoring_python=config.scoring_python, navsim_root=config.navsim_root,
                   cache_batch=config.cache_batch)
    require(value.root.resolve() != config.root.resolve(), 'source and campaign outputs must be distinct')
    atomic_json(value.root / 'resolved_config.json', asdict(value), immutable=True)
    return value


def prepare(config, action, device='cuda:0', shard=0, shards=1, max_samples=None):
    require(action in ('bind', 'cache', 'score'), 'unknown source preparation stage')
    source = source_config(config)
    receipt = config.root / 'SOURCE_PREPARATION.json'
    if action == 'bind':
        from iqe.internal_baseline.binding import bind
        binding = bind(source)
        record = {'campaign_config_hash': config.identity, 'source_config_hash': source.identity,
                  'source_root': str(source.root), 'source_binding_hash': file_hash(source.root / 'BASE_BINDING.json'),
                  'checkpoint': binding['checkpoint'], 'checkpoint_hash': binding['checkpoint_hash'],
                  'latest_resolved_once_here': True, 'no_training_or_navtest_started': True}
        atomic_json(receipt, record, immutable=True)
        return record
    recorded = read_json(receipt)
    require(recorded['campaign_config_hash'] == config.identity and
            recorded['source_binding_hash'] == file_hash(source.root / 'BASE_BINDING.json'), 'source preparation identity changed')
    if action == 'cache':
        require(0 <= shard < shards <= 3, 'at most three bounded source cache workers')
        require(max_samples is None or 0 < max_samples <= 256, 'source cache pilot maximum256')
        from iqe.internal_baseline.features import build_cache
        return build_cache(source, shard, shards, device, max_samples)
    from iqe.internal_baseline.bank import cache_candidates, score_candidates
    from iqe.data.manifests import load_scenes
    scenes = [s for s in load_scenes(source.root / 'scenes.json') if s.split_role in ('incremental_fit', 'stage_val')]
    candidates = cache_candidates(source, scenes, 'base')
    labels = score_candidates(source, scenes, candidates, 'study_source_Base')
    atomic_json(source.root / 'BASE_FIT_VAL_SCORES.json', {s.scene_id: asdict(v) for s, v in zip(scenes, labels)}, immutable=True)
    return {'status': 'COMPLETE', 'Base_candidates': len(labels), 'source_root': str(source.root),
            'expert_training_started': False, 'Scorer_training_started': False}
