import json

import pytest
import torch

from scripts.analysis.flow_safe_improvement.evaluate_policy import load_scores
from starVLA.rl.safe_flow_improvement.contracts import file_sha, fingerprint
from starVLA.rl.safe_flow_improvement.evaluation_override import evaluation_seeds


def test_override_preserves_population_and_reference(tmp_path):
    population = dict(identity_sha256='p', evaluation_seeds=[2101, 2102, 2103, 2104, 2105], reference_seeds=[1101, 1102, 1103, 1104])
    record = dict(population_identity='p', evaluation_seeds=[2101], training_seeds=[42],
                  reference_seeds=population['reference_seeds'], preserve_training_model_optimizer_bank_and_population=True,
                  user_instruction='不用做多seed实验')
    record['sha256'] = fingerprint(record)
    path = tmp_path/'override.json'; path.write_text(json.dumps(record))
    assert evaluation_seeds(population, path)[0] == [2101]
    assert population['evaluation_seeds'] == [2101, 2102, 2103, 2104, 2105]
    record['evaluation_seeds'] = [2102]; path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='checksum'):
        evaluation_seeds(population, path)


def sealed_scores(root, population, seeds, role):
    token = 'scene1'
    sample = dict(mode='ode', role=role, population_identity='p', seeds=seeds,
                  root_precision='original_bfloat16', cohort='dev_tokens', tokens_sha256=fingerprint([token]))
    manifest = dict(sample_manifest=sample); manifest['sha256'] = fingerprint(manifest)
    root.mkdir(); (root/'manifest.json').write_text(json.dumps(manifest))
    path = root/'scenes'/token[:2]/(token+'.pt'); path.parent.mkdir(parents=True)
    feedback = [dict(status='OK', metrics=dict(score=seed/10000, NC=1., DAC=1., TTC=1.)) for seed in seeds]
    torch.save(dict(feedback=feedback), path)
    path.with_suffix('.json').write_text(json.dumps(dict(identity_sha256=manifest['sha256'], payload_sha256=file_sha(path))))
    return path


def test_select_seed_by_identity_not_position_or_best_score(tmp_path):
    population = dict(identity_sha256='p', evaluation_seeds=[2101, 2102, 2103, 2104, 2105],
                      reference_seeds=[1101, 1102, 1103, 1104], dev_tokens=['scene1'])
    root = tmp_path/'old'; sealed_scores(root, population, [2105, 2101, 2103, 2102, 2104], 'evaluation')
    _, rows, _ = load_scores(root, population, 'evaluation', [2101])
    assert rows['scene1']['score'].tolist() == [0.2101]
    with pytest.raises(ValueError, match='seeds'):
        load_scores(root, population, 'evaluation', [2200])
    reference = tmp_path/'ref'; sealed_scores(reference, population, population['reference_seeds'], 'reference')
    _, rows, _ = load_scores(reference, population, 'reference')
    assert len(rows['scene1']['score']) == 4


def test_selected_seed_cannot_hide_infrastructure_errors_or_missing_scenes(tmp_path):
    population = dict(identity_sha256='p', evaluation_seeds=[2101, 2102], reference_seeds=[1101], dev_tokens=['scene1'])
    root = tmp_path/'old'; path = sealed_scores(root, population, [2101, 2102], 'evaluation')
    row = torch.load(path, weights_only=True); row['feedback'][1]['status'] = 'INFRA_ERROR'; torch.save(row, path)
    seal = json.loads(path.with_suffix('.json').read_text()); seal['payload_sha256'] = file_sha(path)
    path.with_suffix('.json').write_text(json.dumps(seal))
    with pytest.raises(ValueError, match='Failed scoring'):
        load_scores(root, population, 'evaluation', [2101])
    path.with_suffix('.json').unlink()
    with pytest.raises(FileNotFoundError):
        load_scores(root, population, 'evaluation', [2101])
