"""Two finite CPU checks: covariance/RNG pairing and complete312 shard gate."""
import json

import numpy as np
import pytest

from tools.qwen_world_safety_exploration import probe_structured as P


def test_stationary_covariance_same_independent_budget_ramp_and_private_RNG():
    # Exact innovation basis proves stationary AR1 covariance without Monte Carlo.
    z=np.zeros((16,8,3),dtype=np.float64)
    for i in range(8):z[i,i,:]=1.
    noise,ar=P.noise_from_innovations(z)
    L=ar[:8,:,0].T
    np.testing.assert_allclose(L@L.T,P.RHO**np.abs(np.arange(8)[:,None]-np.arange(8)[None]),atol=3e-16,rtol=0)
    for axis in range(3):
        cov=noise[1,:8,:,axis].T@noise[1,:8,:,axis]
        np.testing.assert_allclose(np.diag(cov),np.full(8,P.STD[axis]**2),atol=2e-15,rtol=0)
        np.testing.assert_allclose(np.diag(noise[2,:8,:,axis].T@noise[2,:8,:,axis]),
            P.STD[axis]**2*(P.TIMES/4)**2,atol=2e-15,rtol=0)
    np.testing.assert_array_equal(noise[0],z*P.STD)
    np.testing.assert_array_equal(noise[2],noise[1]*(P.TIMES/4)[None,:,None])
    state=np.random.get_state();base=np.arange(16*8*3,dtype=np.float32).reshape(16,8,3)
    first,arrays,audit=P.propose(base,'TRAIN_fixture_a');second,_,again=P.propose(base,'TRAIN_fixture_a')
    np.testing.assert_array_equal(first,second);assert audit==again
    after=np.random.get_state();assert state[0]==after[0] and np.array_equal(state[1],after[1]) and state[2:]==after[2:]
    assert first.shape==(48,8,3) and first.dtype==np.float64 and audit['independent_draw_count']==384
    np.testing.assert_array_equal(first.reshape(3,16,8,3),base[None].astype(np.float64)+arrays['noise'])
    assert not np.array_equal(first,P.propose(base,'TRAIN_fixture_b')[0])


def test_only_complete_eight_exact39_case_shards_allow_aggregation(tmp_path):
    identity=dict(cases=[dict(token=f'fixture{i:03d}') for i in range(312)],world=8,sha256='test-only-fixture')
    tokens=[c['token'] for c in identity['cases']]
    for rank in range(8):
        node=dict(status='COMPLETE',world=8,identity=identity['sha256'],expected_tokens=tokens[rank::8],completed_tokens=tokens[rank::8])
        (tmp_path/f'rank{rank}.json').write_text(json.dumps(node))
    assert P.validate_shards(tmp_path,identity)==tokens
    broken=json.loads((tmp_path/'rank3.json').read_text());broken['completed_tokens'].pop()
    (tmp_path/'rank3.json').write_text(json.dumps(broken))
    with pytest.raises(ValueError):P.validate_shards(tmp_path,identity)
    bad=dict(identity,world=16)
    with pytest.raises(ValueError):P.validate_shards(tmp_path,bad)
