import pytest
from tools.foresight.flux_targets import verify_flux_source,FLUX_REPOSITORY,FLUX_REVISION,FLUX_WEIGHT,FLUX_WEIGHT_SHA256,FLUX_CONFIG


def test_vae_registration_rejects_empty_inventory_unknown_revision_and_self_attested_weights(tmp_path):
    base={'repository':FLUX_REPOSITORY,'revision':FLUX_REVISION,'files':{}}
    with pytest.raises(ValueError,match='Both exact'):verify_flux_source(tmp_path,base)
    other=dict(base,revision='a'*40)
    with pytest.raises(ValueError,match='source/revision'):verify_flux_source(tmp_path,other)
    (tmp_path/'vae').mkdir();(tmp_path/FLUX_CONFIG).write_text('{}')
    claimed=dict(base,files={FLUX_WEIGHT:'b'*64,FLUX_CONFIG:'c'*64})
    with pytest.raises(ValueError,match='Both exact'):verify_flux_source(tmp_path,claimed)
    claimed['files'][FLUX_WEIGHT]=FLUX_WEIGHT_SHA256
    with pytest.raises(ValueError,match='public Git blob'):verify_flux_source(tmp_path,claimed)
