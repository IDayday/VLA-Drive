import hashlib
import json

import pytest

from starVLA.rl.simwam_recipe.execution_sources import snapshot_roots, checked_module_path, OVERLAY_SCHEMA


def overlay(tmp_path):
    base=tmp_path/'base';(base/'starVLA').mkdir(parents=True)
    module=base/'starVLA/example.py';module.write_text('frozen = True\n')
    raw=b'{"commit":"pinned-base"}\n';(base/'FROZEN_SOURCE.json').write_bytes(raw)
    root=tmp_path/'overlay';root.mkdir();(root/'starVLA').symlink_to(base/'starVLA',target_is_directory=True)
    body=dict(schema=OVERLAY_SCHEMA,base=dict(path=str(base),frozen_source_sha256=hashlib.sha256(raw).hexdigest()))
    body['sha256']=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (root/'FROZEN_SOURCE.json').write_text(json.dumps(body))
    return root,base,module


def test_import_resolves_to_exact_pinned_logical_module(tmp_path):
    root,base,module=overlay(tmp_path);roots=snapshot_roots(root)
    assert checked_module_path('starVLA.example',module,roots)==module
    foreign=tmp_path/'foreign.py';foreign.write_bytes(module.read_bytes())
    with pytest.raises(ValueError,match='outside the execution snapshot'):
        checked_module_path('starVLA.example',foreign,roots)
    (root/'starVLA').unlink();(root/'starVLA').mkdir()
    (root/'starVLA/example.py').write_text('new_overlay = True\n')
    with pytest.raises(ValueError,match='outside the execution snapshot'):
        checked_module_path('starVLA.example',module,roots)


def test_tampered_base_and_unsealed_external_import_remain_rejected(tmp_path):
    root,base,module=overlay(tmp_path)
    (base/'FROZEN_SOURCE.json').write_text('{}')
    with pytest.raises(ValueError,match='Pinned frozen base metadata changed'):snapshot_roots(root)
    plain=tmp_path/'plain';plain.mkdir()
    with pytest.raises(ValueError,match='outside the execution snapshot'):
        checked_module_path('starVLA.example',module,snapshot_roots(plain))
