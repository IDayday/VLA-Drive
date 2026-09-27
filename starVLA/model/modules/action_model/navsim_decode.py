"""Original ver1225 NAVSIM decoder, independent of model/video/distributed imports."""
from typing import Optional
import numpy as np

_X_MEAN, _X_STD = 10.172484, 8.805105
_Y_MEAN, _Y_STD = 0.360762, 2.277741


def deal_action_1225(pred: np.ndarray, scale_x: float = 4.5912, act_norm: int = 0,
                     scale_y: Optional[float] = None) -> np.ndarray:
    pred = np.asarray(pred)
    dxdy = pred[..., :2].copy()
    if act_norm == 0:
        dxdy[..., 0] *= scale_x
    else:
        dxdy[..., 0] = dxdy[..., 0] * _X_STD + _X_MEAN
        dxdy[..., 1] = dxdy[..., 1] * _Y_STD + _Y_MEAN
    if scale_y is not None:
        dxdy[..., 1] *= scale_y
    theta = (np.arctan2(pred[..., 2], pred[..., 3]) + np.pi) % (2 * np.pi) - np.pi
    return np.concatenate([dxdy, theta[..., None]], axis=-1)
