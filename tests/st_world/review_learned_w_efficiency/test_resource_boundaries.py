"""Independent CPU resource counterexamples; no real images/checkpoints/fit."""
import copy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tools.st_world_policy import spatial_probe as p
from tools.st_world_policy import training_state as ts
from tools.st_world_policy import metadata_storage as m
from tools.st_world_policy import cache_storage_budget as b


def test_eight_cache_claims_include_masks_staging_and_metadata():
    claims = [p.cache_requirements(n, t, 2048)
              for t in (144, 144, 144, 288) for n in (64, 16)]
    assert sum(sum(c.values()) for c in claims) == 546104992
    assert sum(c['output_bytes'] for c in claims) == 474538240
    assert 546104992 < (1 << 30)
    # 450 MiB is W alone, not the reserve or real serialized size.
    assert sum(80*t*2048*4 for t in (144, 144, 144, 288)) == 450*(1 << 20)


@pytest.mark.parametrize('tokens', [144, 288])
def test_fp32_record_actual_cpu_serialization_fits_declared_write_bound(tmp_path, tokens):
    path = tmp_path/'synthetic.safetensors'
    tensors = {'W': torch.zeros(tokens, 2048, dtype=torch.float32),
               'memory_valid': torch.ones(tokens, dtype=torch.bool)}
    save_file(tensors, str(path))
    allocated = path.stat().st_blocks * 512
    bound = tokens*2048*4 + tokens + 8192
    assert max(path.stat().st_size, allocated) <= bound


def test_new_spatial_tensor_claim_is_not_a_metadata_claim(tmp_path):
    core = {'legacy_reservations': [], 'metadata_package': {
        'schema': m.PACKAGE_SCHEMA, 'maximum_bytes': 4*(1 << 30),
        'baseline_claims': {}, 'baseline_legacy_reservations': [],
        'baseline_policy_sha256': 'a'*64, 'baseline_ledger_sha256': 'b'*64}}
    claim = {'claim_id': 'frozen-W', 'root': str(tmp_path/'W'),
             'cache_identity': 'c'*64, **p.cache_requirements(64, 144, 2048)}
    with pytest.raises(b.StorageBudgetError, match='metadata claim must bind'):
        m.validate_package(core, {claim['root']: claim})
    # Existing baseline reservations remain valid; do not edit a real baseline
    # after migration to exploit this distinction.
    baseline = copy.deepcopy(core)
    baseline['metadata_package']['baseline_claims'][claim['root']] = claim
    assert m.validate_package(baseline, {claim['root']: claim}) == 0


def test_native_verification_reads_all_payloads_even_for_identity_plan(tmp_path, monkeypatch):
    folder = tmp_path/'step_005000_fixture'; folder.mkdir()
    names = ['mp_rank_00_model_states.pt', 'bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt', 'rng_rank0.pt']
    for i, name in enumerate(names):
        (folder/name).write_bytes(bytes([i])*17)
    state = {'identity':'fixture', 'tag':folder.name, 'completed':5000,
             'termination_reason':'OPERATOR_PAUSED'}
    ts.checkpoint_receipt(folder, state, world_size=1)
    reads = []
    original = ts.file_sha256
    def counted(path):
        reads.append(Path(path).name)
        return original(path)
    monkeypatch.setattr(ts, 'file_sha256', counted)
    ts.verify_checkpoint(folder, identity='fixture', world_size=1)
    assert set(reads) == set(names) and len(reads) == len(names)


def test_head_memory_and_parameter_count_are_small_without_model_load():
    model = p.make_probe(2048)
    count = sum(t.numel() for t in model.parameters())
    assert count == 495362
    assert all(t.device.type == 'cpu' and t.dtype == torch.float32 for t in model.parameters())
    assert count*4*4 == 7925792  # weights, gradients, two Adam moment arrays
    assert count*4*12 + (4 << 20) < (64 << 20)  # production's conservative slot allowance


def test_profile_does_not_silently_change_registered_exposures():
    assert p.RECIPE['updates'] == 300
    assert p.RECIPE['scenes_per_update'] == 2
    assert p.RECIPE['patches_per_view'] == 32
    assert p.RECIPE['scenes_per_update'] * 3 * p.RECIPE['patches_per_view'] == 192
    assert p.RECIPE['updates'] * 192 == 57600
    # No invented max-scenes/fit-steps CLI: the registered populations/endpoints
    # are fixed. Profile wall bounds belong in registrations and the supervisor.
    sub = next(x for x in p.parser()._actions if hasattr(x, 'choices') and isinstance(x.choices, dict))
    fit_options = {o for a in sub.choices['fit']._actions for o in a.option_strings}
    assert '--max-steps' not in fit_options and '--max-scenes' not in fit_options
