"""Candidate behavior summaries and scene/log-aware policy comparisons."""
import numpy as np

from .contracts import SAFETY_METRICS


def candidate_summary(requests, metrics):
    request = np.asarray(requests, dtype=np.float64)
    reward = np.array([m['score'] for m in metrics], dtype=np.float64)
    if request.shape != (len(metrics), 8, 3) or not np.isfinite(request).all() or not np.isfinite(reward).all():
        raise ValueError('Invalid candidate population; retain failure denominator')
    n = len(metrics)
    upper = np.triu_indices(n, 1)
    distance = np.linalg.norm(request[:, None, :, :2]-request[None, :, :, :2], axis=-1)
    centered = request[:, :, :2].reshape(n, -1)
    singular = np.linalg.svd(centered-centered.mean(0), compute_uv=False)
    power = singular**2
    if power.sum() > 0:
        q = power[power > 0]/power.sum()
        rank = float(np.exp(-np.sum(q*np.log(q))))
    else:
        rank = 0.
    components = sorted(set.intersection(*[set(m) for m in metrics]) - {'score', 'pdm_score', 'ego_progress', 'two_frame_extended_comfort'})
    varying = [k for k in components if all(m[k] is not None for m in metrics) and np.ptp([m[k] for m in metrics]) > 0]
    safety = np.array([[m[k] == 1 for k in SAFETY_METRICS] for m in metrics])
    xy = np.concatenate([np.zeros((n, 1, 2)), request[:, :, :2]], axis=1)
    velocity = np.gradient(xy, .5, axis=1)
    acceleration = np.gradient(velocity, .5, axis=1)
    speed = np.linalg.norm(velocity, axis=-1)
    curvature = np.abs(velocity[:, :, 0]*acceleration[:, :, 1]-velocity[:, :, 1]*acceleration[:, :, 0])/np.maximum(speed, .5)**3
    terminal_distance = np.linalg.norm(request[:, -1, :2], axis=-1)
    return {'candidates': n, 'reward_mean': float(reward.mean()), 'reward_range': float(np.ptp(reward)),
            'reward_std': float(reward.std()), 'all_equal': bool(np.ptp(reward) == 0), 'all_zero': bool(np.all(reward == 0)),
            'progress_only_effective': bool(np.ptp(reward) > 0 and not varying),
            'varying_nonprogress_metrics': varying,
            'safety_component_variation': {k: bool(np.ptp([m[k] for m in metrics]) > 0) for k in SAFETY_METRICS},
            'joint_safe_candidate_count': int(safety.all(1).sum()),
            'pairwise_ADE_mean_m': float(distance.mean(2)[upper].mean()) if n > 1 else 0.,
            'pairwise_FDE_mean_m': float(distance[:, :, -1][upper].mean()) if n > 1 else 0.,
            'XY_covariance_entropy_effective_rank': rank,
            'candidate_mean_speed_std_mps': float(speed.mean(1).std()),
            'candidate_curvature_mean_std_per_m': float(curvature.mean(1).std()),
            'candidate_terminal_distance_std_m': float(terminal_distance.std()),
            'finite_difference_motion_metrics_are_descriptive': True,
            'geometric_variance_is_not_behavioral_usefulness': True}


def log_bootstrap_difference(scene_differences, logs, *, repeats=2000, seed=42):
    values = np.asarray(scene_differences, dtype=np.float64)
    logs = np.asarray(logs)
    if values.ndim != 1 or len(values) != len(logs) or not np.isfinite(values).all():
        raise ValueError('Aggregate inference seeds per scene before log bootstrap')
    names = sorted(set(logs.tolist()))
    sums = np.array([values[logs == log].sum() for log in names])
    counts = np.array([(logs == log).sum() for log in names])
    draws = np.random.default_rng(seed).integers(0, len(names), (repeats, len(names)))
    means = sums[draws].sum(1)/counts[draws].sum(1)
    return {'mean': float(values.mean()), 'CI95': np.quantile(means, [.025, .975]).tolist(),
            'scenes': len(values), 'logs': len(names), 'bootstrap_repeats': repeats, 'seed': seed,
            'training_seed_uncertainty_included': False}
