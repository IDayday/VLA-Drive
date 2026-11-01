"""Two bounded CPU checks: variance controls/FP32 and exact saved-source/shards."""
import json

import numpy as np
import pytest

from tools.qwen_world_safety_exploration import probe_noise_profiles as C
from starVLA.model.modules.qwen_world_postcot.candidate_bank import trajectory_hash


def test_H_same_horizon_variance_as_G_I_equal_trace_and_FP32_contract():
    z=np.zeros((16,8,3),dtype=np.float64)
    for i in range(8):z[i,i,:]=1
    base=np.zeros((16,8,3),dtype=np.float32)
    requested,arrays,audit=C.propose(base,z)
    old_noise,_=C.P.noise_from_innovations(z)
    for axis in range(3):
        covH=arrays['noise'][0,:8,:,axis].T@arrays['noise'][0,:8,:,axis]
        covI=arrays['noise'][1,:8,:,axis].T@arrays['noise'][1,:8,:,axis]
        covG=old_noise[2,:8,:,axis].T@old_noise[2,:8,:,axis]
        np.testing.assert_allclose(np.diag(covH),np.diag(covG),atol=1e-15,rtol=0)
        np.testing.assert_allclose(np.trace(covI),np.trace(covG),atol=3e-15,rtol=0)
        np.testing.assert_allclose(covH,np.diag(np.diag(covH)),atol=0,rtol=0)
    assert C.ENERGY_RATIO==.3984375 and requested.dtype==np.float32 and requested.shape==(32,8,3)
    np.testing.assert_array_equal(requested,arrays['raw_requested'].astype(np.float32))
    np.testing.assert_array_equal(arrays['innovations'],z)
    for submitted in requested:assert trajectory_hash(submitted)==trajectory_hash(submitted.tolist())
    raw=requested[0].astype(np.float64);raw[0,0]=.123456789012345
    with pytest.raises(ValueError,match='not exact FP32'):trajectory_hash(raw)
    assert audit['new_independent_draws']==0 and audit['reused_independent_coordinates']==384


def test_exact_saved_z_root_join_and_only_all8_complete_shards(tmp_path):
    folder=tmp_path/'v2';(folder/'scenes').mkdir(parents=True)
    token='fixture';base=np.zeros((16,8,3),dtype=np.float32);roots=np.ones_like(base)
    z=np.random.Generator(np.random.PCG64(42)).standard_normal((16,8,3),dtype=np.float64)
    path=folder/'scenes/fixture.npz';np.savez_compressed(path,source_ODE_requested=base,source_roots=roots,innovations=z)
    row=C.sealed(dict(schema=C.P.SCHEMA,status='OK',identity='fixtureidentity',token=token,log='fixturelog',role='historical_failure',pair_token='pair',source_root_hash=C.P.array_hash(roots),arrays=dict(path='scenes/fixture.npz'),noise_audit=dict(independent_innovations_hash=C.P.array_hash(z))))
    C.atomic(folder/'scenes/fixture.json',row)
    case=dict(token=token,log='fixturelog',role='historical_failure',pair_token='pair',source_root_hash=C.P.array_hash(roots),source_ODE_requested_hash=C.P.array_hash(base),source_innovations_hash=C.P.array_hash(z),profile_scene_sha256=C.sha(folder/'scenes/fixture.json'),profile_arrays_sha256=C.sha(path))
    manifest=dict(source_profile=dict(output=str(folder),identity_sha256='fixtureidentity'))
    _,gotbase,gotroot,gotz=C.load_source(case,manifest)
    np.testing.assert_array_equal(gotbase,base);np.testing.assert_array_equal(gotroot,roots);np.testing.assert_array_equal(gotz,z)
    bad=dict(case,source_innovations_hash=C.P.array_hash(z+1))
    with pytest.raises(ValueError):C.load_source(bad,manifest)
    output=tmp_path/'ranks';output.mkdir();tokens=[f'fixture{i}' for i in range(312)]
    identity=dict(cases=[dict(token=t) for t in tokens],world=8,sha256='shardfixture')
    for rank in range(8):C.atomic(output/f'rank{rank}.json',dict(status='COMPLETE',world=8,identity='shardfixture',expected_tokens=tokens[rank::8],completed_tokens=tokens[rank::8]))
    assert C.P.validate_shards(output,identity)==tokens
    node=json.loads((output/'rank7.json').read_text());node['completed_tokens'].pop();C.atomic(output/'rank7.json',node)
    with pytest.raises(ValueError):C.P.validate_shards(output,identity)
