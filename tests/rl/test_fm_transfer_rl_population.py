import copy
import json

import pytest
import torch

from scripts.analysis.fm_transfer.analyze_rl_population import analyze_rows, read_prefix
from starVLA.rl.simwam_recipe import data_stream


def fixture(tmp_path, monkeypatch, world=8, count=9, collections=3):
    tokens = [f'train-{i}' for i in range(count)]
    pop = tmp_path / 'population.json'
    pop.write_text(json.dumps(dict(train_tokens=tokens, identity_sha256='population',
        token_logs={t: str(i // 2) for i, t in enumerate(tokens)})))
    monkeypatch.setattr(data_stream, 'POPULATION', pop)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(tokens=tokens, population_identity='population',
        status='COMPLETE_FIXED_SFT_SINGLE_NATIVE_ODE_HARD_SUBSET')))
    stream = data_stream.SceneStream(manifest, world)
    rows = [[] for _ in range(world)]
    for c in range(1, collections + 1):
        for rank in range(world):
            token, _ = stream.at((c - 1) * world, rank)
            for inner in range(1, 5):
                rows[rank].append(dict(updates=(c-1)*4+inner, collection=c,
                    inner_completed=inner, token=token, run='run', invocation='attempt',
                    raw_scene_exposures=c*world, new_candidates=c*world*8,
                    behavior=dict(run='run', update=(c-1)*4, adapter_sha256=str(c)),
                    chain_identity=dict(chain=f'chain-{c}-{rank}', old_logp=f'prob-{c}-{rank}')))
    return stream, rows


def test_padding_revisit_and_inner_reuse_have_different_denominators(tmp_path, monkeypatch):
    stream, rows = fixture(tmp_path, monkeypatch)
    before = torch.random.get_rng_state().clone()
    summary, probes = analyze_rows(rows, stream, 'run', 12)
    assert torch.equal(before, torch.random.get_rng_state())
    assert summary['scene_exposures'] == 24 and summary['fresh_candidates'] == 192
    assert summary['unique_scene_count'] == 9
    assert summary['scene_revisits_after_first_exposure'] == 15
    assert summary['epoch_padding_exposures'] == 7
    assert summary['nonpadding_revisits_after_first_exposure'] == 8
    assert summary['distinct_chain_digests_in_journal'] == 24
    assert summary['epochs'][0]['scene_exposures'] == 16
    assert summary['epochs'][1]['scene_exposures'] == 8
    assert probes == [1, 2, 3]


@pytest.mark.parametrize('world,count,collections', [(1, 3, 5), (2, 3, 3), (8, 8, 2)])
def test_complete_and_partial_epochs(tmp_path, monkeypatch, world, count, collections):
    stream, rows = fixture(tmp_path, monkeypatch, world, count, collections)
    result, _ = analyze_rows(rows, stream, 'run', collections*4)
    assert result['scene_exposures'] == collections*world
    assert result['unique_scene_count'] == count
    assert result['epoch_padding_exposures'] == (1 if world == 2 else 0)


@pytest.mark.parametrize('failure', ['permutation', 'inner_chain', 'old_probability',
    'foreign_run', 'budget', 'cross_rank_behavior', 'reused_chain', 'duplicate', 'missing_rank'])
def test_refuse_false_fresh_or_committed_population(tmp_path, monkeypatch, failure):
    stream, rows = fixture(tmp_path, monkeypatch)
    if failure == 'permutation': rows[0][0]['token'] = 'dev-or-wrong-train'
    elif failure == 'inner_chain': rows[0][1]['chain_identity']['chain'] = 'other'
    elif failure == 'old_probability': rows[0][1]['chain_identity']['old_logp'] = 'other'
    elif failure == 'foreign_run': rows[0][0]['run'] = 'other'
    elif failure == 'budget': rows[0][0]['new_candidates'] *= 4
    elif failure == 'cross_rank_behavior':
        for row in rows[0][:4]: row['behavior']['adapter_sha256'] = 'other'
    elif failure == 'reused_chain':
        for row in rows[1][:4]: row['chain_identity']['chain'] = rows[0][0]['chain_identity']['chain']
    elif failure == 'duplicate': rows[0][1] = copy.deepcopy(rows[0][0])
    elif failure == 'missing_rank': rows.pop()
    with pytest.raises(ValueError): analyze_rows(rows, stream, 'run', 12)


def test_prefix_ignores_uncommitted_tail_but_does_not_deduplicate(tmp_path, monkeypatch):
    _, rows = fixture(tmp_path, monkeypatch)
    path = tmp_path / 'journal.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows[0][:4]) + '{"updates":5')
    prefix, sha = read_prefix(path, 4)
    assert prefix == rows[0][:4] and len(sha) == 64
    with pytest.raises(ValueError): read_prefix(path, 8)
    path.write_text(''.join(json.dumps(rows[0][0])+'\n' for _ in range(4)))
    with pytest.raises(ValueError): read_prefix(path, 4)
