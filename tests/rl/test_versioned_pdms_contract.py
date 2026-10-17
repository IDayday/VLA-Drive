"""Version/cache failure gates; official numeric parity is a real-cache CLI."""
import json
from pathlib import Path
import pytest
from starVLA.rl.fm_transfer import versioned_pdms as v


def test_release_pins_distinct():
    assert v.RELEASES['v1.0'][0] != v.RELEASES['v1.1'][0]
    assert v.RELEASES['v1.0'][1] != v.RELEASES['v1.1'][1]


def test_wrong_release_refused(tmp_path):
    (tmp_path / 'SOURCE_COMPLETE.json').write_text(json.dumps(dict(commit=v.RELEASES['v1.1'][0], tag='v1.1')))
    with pytest.raises(ValueError, match='Wrong official'):
        v.upstream_identity(tmp_path, 'v1.0')


@pytest.mark.parametrize('failure', ['wrong_version', 'changed_payload', 'wrong_token', 'wrong_log', 'missing_seal'])
def test_cache_contract(tmp_path, monkeypatch, failure):
    path = tmp_path / 'metric_cache.pkl'; path.write_bytes(b'qualification payload')
    spec = dict(cache_identity='v1.0-fixed-reference')
    monkeypatch.setattr(v, 'SPEC', spec)
    seal = dict(token='token', log='log', cache_identity=spec['cache_identity'], sha256=v.file_sha(path))
    if failure == 'wrong_version': seal['cache_identity'] = 'v1.1-reference'
    if failure == 'changed_payload': seal['sha256'] = 'bad'
    if failure == 'wrong_token': seal['token'] = 'different'
    if failure == 'wrong_log': seal['log'] = 'different'
    if failure != 'missing_seal': path.with_suffix('.json').write_text(json.dumps(seal))
    with pytest.raises((ValueError, FileNotFoundError)):
        v.validate_cache(path, 'token', 'log')


def test_valid_cache_seal(tmp_path, monkeypatch):
    path = tmp_path / 'metric_cache.pkl'; path.write_bytes(b'qualification payload')
    monkeypatch.setattr(v, 'SPEC', dict(cache_identity='pinned'))
    seal = dict(token='token', log='log', cache_identity='pinned', sha256=v.file_sha(path))
    path.with_suffix('.json').write_text(json.dumps(seal))
    assert v.validate_cache(path, 'token', 'log') == seal
