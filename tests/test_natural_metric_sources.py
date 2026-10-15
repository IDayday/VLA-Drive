from pathlib import Path

import pytest

from tools.decision_local.verify_nuplan_runtime_sources import compare


def trees(tmp_path):
    reference, installed = tmp_path / 'reference', tmp_path / 'installed'
    for root in (reference, installed):
        root.mkdir()
        (root / 'runtime.py').write_text('VALUE = 1\n')
    (reference / 'test').mkdir()
    (reference / 'test' / 'test_runtime.py').write_text('def test_value(): pass\n')
    return reference, installed


def test_test_only_package_omission(tmp_path):
    result = compare(*trees(tmp_path))
    assert result['shared_files'] == 1
    assert result['omitted_test_files'] == ['test/test_runtime.py']


def test_runtime_change_is_rejected(tmp_path):
    reference, installed = trees(tmp_path)
    (installed / 'runtime.py').write_text('VALUE = 2\n')
    with pytest.raises(ValueError, match='runtime file differs'):
        compare(reference, installed)


def test_non_test_omission_is_rejected(tmp_path):
    reference, installed = trees(tmp_path)
    (reference / 'other_runtime.py').write_text('VALUE = 3\n')
    with pytest.raises(ValueError, match='non-test reference file'):
        compare(reference, installed)
