import pytest
import torch

from tools.foresight.export_interaction_targets import save_target


@pytest.mark.parametrize('batch_size', [1, 8, 128])
def test_scene_target_preserves_values_without_serializing_other_scenes(tmp_path, batch_size):
    batch = torch.arange(batch_size * 8 * 512, dtype=torch.float32).reshape(batch_size, 8, 512)
    for index in {0, batch_size - 1}:
        latent = batch[index]
        payload = {'identity': 'fixed-export-identity', 'token': str(index),
                   'latent': latent, 'interaction_target_valid': bool(index % 2),
                   'known_peer_points': index + 1}
        destination = tmp_path / f'{index}.pt'
        save_target(destination, payload)
        restored = torch.load(destination, weights_only=True)
        assert restored.keys() == payload.keys()
        assert all(restored[key] == value for key, value in payload.items() if key != 'latent')
        assert torch.equal(restored['latent'], latent)
        assert restored['latent'].untyped_storage().nbytes() == 8 * 512 * 4
        assert destination.stat().st_size < 32768
        assert not destination.with_suffix('.tmp').exists()
        assert torch.equal(payload['latent'], batch[index])
