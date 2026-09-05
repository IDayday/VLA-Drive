#!/usr/bin/env python3
"""Create official trainval labels bound to full frozen 64-candidate groups."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navsim.agents.EpisodeDrive.scorer_replay import validate_train_tokens, group_identity, categorize_group
from navsim.agents.EpisodeDrive.formal_initialization import sha256_file, combined_file_sha256


def build(args):
    if args.output.exists():
        raise FileExistsError(f'Replay bank is immutable: {args.output}')
    manifest = json.loads(args.train_manifest.read_text())
    provenance = json.loads(args.candidate_bank.with_suffix('.npz.manifest.json').read_text())
    checkpoint_sha = sha256_file(args.checkpoint)
    if provenance.get('checkpoint_sha256') != checkpoint_sha or provenance.get('split') != 'trainval':
        raise ValueError('Candidate bank needs matching checkpoint SHA and trainval provenance')
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.agents.EpisodeDrive.score_module.compute_navsim_score import get_sub_score
    loader = MetricCacheLoader(args.metric_cache)
    evaluator_files = [ROOT / 'navsim/agents/EpisodeDrive/score_module/compute_navsim_score.py',
                       ROOT / 'navsim/agents/EpisodeDrive/score_module/train_pdm_scorer.py']
    evaluator_sha = combined_file_sha256(evaluator_files)
    with np.load(args.candidate_bank, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    tokens = arrays['tokens'].tolist()
    validate_train_tokens(tokens, manifest)
    identities, scores, categories, group_tests = [], [], [], []
    for index, token in enumerate(tokens):
        proposals = arrays['proposals'][index]
        scene_path = Path(loader.metric_cache_paths[token])
        identity = group_identity(token, proposals, evaluator_sha, checkpoint_sha, sha256_file(scene_path))
        labels = np.asarray(get_sub_score(str(scene_path), proposals, True)[0], dtype=np.float32)
        if labels.shape != (64, 7) or not np.isfinite(labels).all():
            raise RuntimeError(f'Invalid official labels for {token}')
        if index < args.group_dependency_checks:
            singleton = np.asarray(get_sub_score(str(scene_path), proposals[:1], True)[0])
            group_tests.append({'token': token, 'max_abs_diff_singleton_vs_group': float(abs(singleton[0]-labels[0]).max())})
        identities.append(identity)
        scores.append(labels)
        categories.append(categorize_group(arrays['predicted_pdms'][index], labels, arrays['component_probabilities'][index]))
    arrays['candidate_scores'] = np.stack(scores)
    arrays['group_keys'] = np.asarray([item['group_key'] for item in identities])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    report = {'schema_version': 1, 'split': 'trainval', 'run_type': 'scorer_replay_bank',
              'source_bank_sha256': sha256_file(args.candidate_bank), 'checkpoint_sha256': checkpoint_sha,
              'train_manifest_sha256': sha256_file(args.train_manifest), 'bank_sha256': sha256_file(args.output),
              'groups': identities, 'categories': categories, 'group_dependency_checks': group_tests,
              'label_cache_always_binds_full_64_group': True,
              'sampling_default': 'uniform_full_data', 'formal_replay_enabled': False}
    args.output.with_suffix('.npz.manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('candidate-bank', 'train-manifest', 'checkpoint', 'metric-cache', 'output'):
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--group-dependency-checks', type=int, default=4)
    print(json.dumps(build(parser.parse_args()), indent=2))
