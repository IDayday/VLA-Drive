"""Read-only, paired PCA error diagnostics; never changes the training loss.

The caller must join predictions/targets by the exact scene/view/patch request
identity first. This function deliberately cannot load a model or teacher.
"""
import numpy as np


def paired_channel_errors(pred256, pred384, target384, valid, *, request_ids256,
                          request_ids384, target_request_ids):
    arrays = [np.asarray(x) for x in (pred256, pred384, target384)]
    p256, p384, target = arrays
    valid = np.asarray(valid)
    ids = [np.asarray(x) for x in (request_ids256, request_ids384, target_request_ids)]
    if (target.ndim < 2 or target.shape[-1] != 384 or p384.shape != target.shape
            or p256.shape != (*target.shape[:-1], 256) or valid.shape != target.shape[:-1]
            or valid.dtype != np.bool_ or any(a.dtype.kind != 'f' for a in arrays)):
        raise ValueError('Expected matched floating [...,256/384] predictions and Boolean request mask')
    if (any(x.shape != valid.shape or x.dtype.kind not in 'US' for x in ids)
            or not np.array_equal(ids[0], ids[1]) or not np.array_equal(ids[0], ids[2])):
        raise ValueError('Scene/view/patch request identities differ')
    if any(not np.isfinite(x[valid]).all() for x in arrays):
        raise ValueError('Nonfinite prediction/target at a valid request')
    n = int(valid.sum())
    if n == 0:
        return {'status': 'NO_VALID_REQUESTS', 'valid_requests': 0, 'metrics': None}
    a, b, t = (x[valid].astype(np.float64) for x in arrays)

    def block(pred, truth):
        error = np.square(pred - truth)
        energy = np.square(truth)
        total_energy = float(energy.sum())
        return {'channels': truth.shape[-1], 'elements': int(truth.size),
                'sse': float(error.sum()), 'mse': float(error.mean()),
                'target_energy': total_energy,
                'relative_sse_to_zero': float(error.sum() / total_energy) if total_energy > 0 else None,
                'per_component_mse': error.mean(0).tolist(),
                'per_component_target_mean': truth.mean(0).tolist(),
                'per_component_target_variance': truth.var(0).tolist()}

    common256 = block(a, t[:, :256])
    common384 = block(b[:, :256], t[:, :256])
    tail384 = block(b[:, 256:], t[:, 256:])
    full384 = block(b, t)
    return {'status': 'COMPLETE_DIAGNOSTIC_ONLY', 'valid_requests': n,
            'common_first256': {'arm256': common256, 'arm384': common384},
            'extra_128': {'arm384': tail384,
                          'zero_prediction_mse': float(np.square(t[:, 256:]).mean()),
                          'arm256': 'NOT_PREDICTED_NOT_ZERO_ERROR'},
            'arm384_full_mse': full384['mse'],
            'arm384_weighted_block_mse': (256 * common384['mse'] + 128 * tail384['mse']) / 384,
            'training_loss_or_weights_changed': False,
            'relative_sse_is_diagnostic_not_whitening': True}
