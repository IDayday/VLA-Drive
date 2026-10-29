"""Shared canonical scene validation, full denominator and invocation ledger."""
import json

import pytest
import numpy as np

from tools.qwen_world_compare import evaluate
from starVLA.rl.qwen_world_compare.contracts import fingerprint


def population():
    return dict(dev_tokens=['s0', 's1', 's2'], navtest_tokens=['t0'],
                token_logs={'s0':'log0_1_2', 's1':'log0_1_2', 's2':'log1_1_2', 't0':'test_1_2'})


def write_eval(root, rewards=(.8, .6, 1.)):
    root.mkdir();(root/'scenes').mkdir()
    pop = population();tokens = pop['dev_tokens']
    identity = dict(schema=evaluate.EVALUATION_SCHEMA, architecture_id='A', qualification=False,
        seed=2101, cohort='DEV', tokens=tokens, count=3, token_sha256=fingerprint(tokens),
        model_sha256='test_model', parent_model_sha256='test_IL', protocol={'sha256':'canonical_v1.1'},
        data_sha256='same_data', numerics={'same':True}, inference='single_candidate_cleanx_K10_shift5')
    identity['sha256'] = fingerprint(identity)
    (root/'identity.json').write_text(json.dumps(identity))
    request = np.zeros((8, 3)).tolist()
    for token, reward in zip(tokens, rewards):
        # All safety/comfort pass. Canonical score=(5*EP+7)/12.
        progress = (reward*12-7)/5
        metrics = dict(score=reward, no_at_fault_collisions=1., drivable_area_compliance=1.,
            time_to_collision_within_bound=1., ego_progress=progress, comfort=1.,
            driving_direction_compliance=1.)
        feedback = dict(status='OK', token=token, log=pop['token_logs'][token],
            protocol_sha256='canonical_v1.1', requested_trajectory=request,
            metrics=metrics, cache={'sha256':token+'_cache'})
        row = dict(token=token, log=pop['token_logs'][token], status='OK', identity=identity['sha256'],
                   requested_trajectory=request, feedback=feedback)
        row['sha256'] = fingerprint(row)
        (root/'scenes'/f'{token}.json').write_text(json.dumps(row))
    return pop


def test_complete_new_schema_summary_and_paired_bootstrap(tmp_path):
    left, right = tmp_path/'left', tmp_path/'right'
    pop = write_eval(left, (.8, .7, 1.));write_eval(right, (.75, .65, 1.))
    summary = evaluate.summarize(left, right, population=pop)
    assert summary['scenes'] == summary['scene_denominator'] == 3
    assert summary['invalid_scenes'] == 0
    assert summary['PDMS'] == pytest.approx(100*2.5/3)
    assert summary['paired']['physical_groups'] == 2
    assert summary['paired']['mean_delta'] == pytest.approx(.1/3)


def test_missing_scene_or_invalid_metric_is_not_filtered(tmp_path):
    root = tmp_path/'data';pop = write_eval(root, (.8, .7, 1.))
    (root/'scenes/s2.json').unlink()
    with pytest.raises(ValueError, match='incomplete'):
        evaluate.summarize(root, population=pop)
    p = root/'scenes/s0.json';row = json.loads(p.read_text())
    row['feedback']['metrics']['time_to_collision_within_bound'] = float('nan')
    row.pop('sha256');row['sha256'] = fingerprint(row);p.write_text(json.dumps(row))
    # Restore coverage, then the invalid scene is rejected rather than dropped.
    saved = tmp_path/'saved';write_eval(saved, (.8, .7, 1.))
    (root/'scenes/s2.json').write_text((saved/'scenes/s2.json').read_text())
    with pytest.raises(ValueError, match='Invalid official metric'):
        evaluate.summarize(root, population=pop)


def test_invocation_cost_does_not_disappear_when_rank_summary_is_rewritten(tmp_path):
    (tmp_path/'invocations').mkdir()
    first = dict(invocation='old', new=2, reused=0, physical_inference_attempts=3,
                 wall_seconds=10., forward_seconds=6., scoring_seconds=2., model_load_seconds=1.)
    (tmp_path/'invocations/rank0_old.json').write_text(json.dumps(first))
    current = dict(invocation='new', new=1, reused=2, physical_inference_attempts=1,
                   wall_seconds=4., forward_seconds=2., scoring_seconds=1., model_load_seconds=.5)
    (tmp_path/'invocations/rank0_new.json').write_text(json.dumps(current))
    total = evaluate._cumulative_cost(tmp_path, 0, current)
    assert total['wall_seconds'] == 14.
    assert total['new'] == 3
    assert total['physical_inference_attempts'] == 4
    assert total['model_load_seconds'] == 1.5
