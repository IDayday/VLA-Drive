"""Physical candidate diversity and reward signal, without candidate selection."""
import numpy as np


def trajectory_diversity(physical, rewards, chain=None):
    poses = np.asarray(physical, dtype=np.float64)
    values = np.asarray(rewards, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[-1] != 3 or len(poses) < 2:
        raise ValueError("expected at least two [G,H,3] physical trajectories")
    if values.shape != (len(poses),) or not np.isfinite(poses).all() or not np.isfinite(values).all():
        raise ValueError("nonfinite or mismatched candidates/rewards")
    xy = poses[..., :2]
    row, col = np.triu_indices(len(poses), 1)
    distance = np.linalg.norm(xy[row]-xy[col], axis=-1)
    ade = distance.mean(-1)
    fde = distance[:, -1]
    speed = np.linalg.norm(np.diff(np.concatenate([np.zeros((len(poses), 1, 2)), xy], axis=1), axis=1), axis=-1)/.5
    headings = poses[..., 2]
    resultant = np.hypot(np.cos(headings).mean(0), np.sin(headings).mean(0))
    centered = (xy-xy.mean(0)).reshape(len(poses), -1)
    eigenvalues = np.linalg.svd(centered, compute_uv=False)**2
    rank = float(eigenvalues.sum()**2 / (eigenvalues**2).sum()) if eigenvalues.sum() > 1e-20 else 0.
    def quantiles(x):
        return {str(q): float(np.quantile(x, q)) for q in [0., .1, .5, .9, 1.]}
    result = {"group_size": len(poses), "pairs": len(row),
        "pair_ade_m": quantiles(ade), "pair_fde_m": quantiles(fde),
        "pair_ade_above_m": {str(t): float((ade > t).mean()) for t in [.01, .05, .1, .25, .5, 1.]},
        "endpoint_std_xy_m": xy[:, -1].std(0).tolist(),
        "waypoint_std_xy_m": xy.std(0).tolist(),
        "speed_std_mean_mps": float(speed.std(0).mean()),
        "heading_circular_std_mean_rad": float(np.sqrt(-2*np.log(np.clip(resultant, 1e-15, 1))).mean()),
        "xy_covariance_effective_rank": rank,
        "reward_mean": float(values.mean()), "reward_std": float(values.std()),
        "reward_span": float(np.ptp(values)), "distinct_rewards": len(np.unique(values)),
        "all_equal_rewards": bool(np.ptp(values) == 0),
        "reward_zero_fraction": float((values == 0).mean()),
        "reward_pair_spans": quantiles(np.abs(values[row]-values[col])),
        "diagnostic_candidate_max_reward": float(values.max())}
    if chain is not None:
        x = np.asarray(chain, dtype=np.float64)
        if x.ndim != 4 or len(x) != len(poses) or not np.isfinite(x).all():
            raise ValueError("invalid [G,K+1,H,D] chain")
        result["normalized_chain_rms_std"] = np.sqrt(x.var(0).mean((1, 2))).tolist()
    return result
