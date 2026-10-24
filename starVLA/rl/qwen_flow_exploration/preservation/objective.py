"""Positive native-source FM rehearsal and explicit projected expected costs."""
import math
import torch
from starVLA.rl.flow_grpo.math import centered_group_rewards
from starVLA.model.modules.qwen_sourceflow.vendor.scheduler_continuous import WanContinuousFlowMatchScheduler


@torch.no_grad()
def official_costs(subscores, metrics):
    values = torch.as_tensor([[float(row[k]) for k in metrics] for row in subscores], dtype=torch.float64)
    if not values.isfinite().all() or (values < 0).any() or (values > 1).any():
        raise ValueError('Finite official 0-1 submetrics required')
    return 1-values  # NC=0.5 is cost0.5 in this independently named version.


class ProjectedCostDual:
    """Plain projected gradient ascent; updated once per fresh raw collection."""
    def __init__(self, budgets, *, learning_rate=.01, initial=1., max_multiplier=1000.):
        self.budgets = torch.as_tensor(budgets, dtype=torch.float64).clone()
        if self.budgets.ndim != 1 or not self.budgets.isfinite().all() or (self.budgets < 0).any() or (self.budgets > 1).any():
            raise ValueError('Immutable finite TRAIN cost budgets required')
        if learning_rate <= 0 or initial < 0 or max_multiplier <= initial:
            raise ValueError('Invalid projected dual configuration')
        self.learning_rate, self.max_multiplier = float(learning_rate), float(max_multiplier)
        self._fixed_budgets = tuple(self.budgets.tolist())
        self.values = torch.full_like(self.budgets, float(initial))
        self.collections = 0

    def snapshot(self, device='cpu'):
        return self.values.detach().clone().to(device)

    def update(self, raw_means, collection):
        if tuple(self.budgets.tolist()) != self._fixed_budgets:
            raise ValueError('Frozen reference budgets changed')
        means = torch.as_tensor(raw_means, dtype=torch.float64).detach().cpu()
        if collection != self.collections or means.shape != self.budgets.shape or not means.isfinite().all():
            raise ValueError('One unfiltered cost update per fresh collection required')
        if (means < 0).any() or (means > 1).any():
            raise ValueError('Costs must be true bounded feedback, not standardized advantages')
        self.values = (self.values+self.learning_rate*(means-self.budgets)).clamp_min(0)
        self.collections += 1
        if (self.values > self.max_multiplier).any():
            raise FloatingPointError('Dual upper alarm; no silent weakening or capped-success claim')

    def state_dict(self):
        return dict(budgets=self.budgets.tolist(), values=self.values.tolist(), collections=self.collections,
                    learning_rate=self.learning_rate, max_multiplier=self.max_multiplier)

    def load_state_dict(self, state):
        if state['budgets'] != self.budgets.tolist() or state['learning_rate'] != self.learning_rate or state['max_multiplier'] != self.max_multiplier:
            raise ValueError('Changed fixed budgets or dual update rule')
        values = torch.tensor(state['values'], dtype=torch.float64)
        if values.shape != self.values.shape or not values.isfinite().all() or (values < 0).any():
            raise ValueError('Invalid restored cost multipliers')
        self.values, self.collections = values, int(state['collections'])


@torch.no_grad()
def cost_scalarized_advantage(reward_advantage, costs, multipliers, *, common_scale):
    """Use one supplied, lambda-independent scale; no post scalarization z-score.

    common_scale is locked by the main protocol (e.g. fresh RAW global reward
    std plus its numeric epsilon). This proxy is not an exact CMDP gradient.
    """
    if reward_advantage.ndim != 2 or costs.shape[:2] != reward_advantage.shape or costs.ndim != 3:
        raise ValueError('Scene, candidate and cost axes must remain separate')
    if not costs.isfinite().all() or (costs < 0).any() or (costs > 1).any():
        raise ValueError('Costs must be raw official 1-metric values, not standardized signals')
    multipliers = torch.as_tensor(multipliers, dtype=torch.float64, device=costs.device).detach()
    scale = float(common_scale)
    if not math.isfinite(scale) or scale <= 0 or multipliers.shape != costs.shape[-1:] or (multipliers < 0).any():
        raise ValueError('Positive common scale and detached nonnegative multipliers required')
    centered = torch.stack([centered_group_rewards(costs[..., k].double())[0] for k in range(costs.shape[-1])], -1)
    result = reward_advantage.double()-(centered*multipliers).sum(-1)/scale
    if not result.isfinite().all():
        raise FloatingPointError('Nonfinite cost advantage; no silent replacement')
    return result.float().detach()


def source_fm_replay_loss(actor, condition, targets, *, generator=None, noise=None, timestep=None):
    """One independent source FM draw per scene; never root-noise coupling.

    The original scheduler and action_velocity are called directly. q_target
    sampling belongs to FrozenArchive.sample; there is no negative MSE weight.
    """
    if targets.ndim != 3 or targets.shape[1:] != (8, 3) or not targets.isfinite().all():
        raise ValueError('Source-native normalized requested actions [B,8,3] required')
    targets = targets.detach().float()
    scheduler = actor.infer_action_scheduler
    if not isinstance(scheduler, WanContinuousFlowMatchScheduler):
        # The recipe's identical vendored class may have a distinct import
        # identity; assert actual source API/time convention, never replace it.
        if scheduler.num_train_timesteps != 1000 or scheduler.shift != 5.:
            raise ValueError('Wrong source FM scheduler')
    if noise is None:
        if generator is None:
            raise ValueError('Independent replay RNG is required')
        noise = torch.randn(targets.shape, generator=generator, dtype=torch.float32).to(targets.device)
    if timestep is None:
        if generator is None:
            raise ValueError('Independent replay RNG is required')
        seed = int(torch.randint(0, 2**62, (1,), generator=generator))
        devices = [targets.device.index or 0] if targets.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if targets.device.type == 'cuda':
                torch.cuda.default_generators[targets.device.index or 0].manual_seed(seed)
            timestep = scheduler.sample_training_t(len(targets), targets.device, torch.float32)
    noise, timestep = noise.detach().to(targets).float(), timestep.detach().to(targets.device).float()
    if noise.shape != targets.shape or timestep.shape != (len(targets),) or not noise.isfinite().all() or not timestep.isfinite().all():
        raise ValueError('Independent source FM draws have invalid shape/value')
    interpolated = scheduler.add_noise(targets, noise, timestep)
    target_velocity = scheduler.training_target(targets, noise, timestep)
    prediction = actor.action_velocity(interpolated, timestep, condition)
    per_scene = (prediction.float()-target_velocity).square().mean((-1, -2))
    weights = scheduler.training_weight(timestep)
    loss = (per_scene*weights).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite source archive FM loss')
    return loss, dict(per_scene_mse=per_scene.detach(), training_weight=weights.detach(),
                      timestep=timestep, noise_is_independent_of_candidate_root=True,
                      supervised_scene_exposures=len(targets), velocity_forward_calls=1)


def gradient_comparison(loss_a, loss_b, parameters):
    parameters = tuple(p for p in parameters if p.requires_grad)
    left = torch.autograd.grad(loss_a, parameters, retain_graph=True, allow_unused=True)
    right = torch.autograd.grad(loss_b, parameters, retain_graph=True, allow_unused=True)
    a2 = sum(float(g.detach().double().square().sum()) for g in left if g is not None)
    b2 = sum(float(g.detach().double().square().sum()) for g in right if g is not None)
    dot = sum(float((a.detach().double()*b.detach().double()).sum()) for a, b in zip(left, right) if a is not None and b is not None)
    return dict(norm_a=a2**.5, norm_b=b2**.5, cosine=dot/(a2*b2)**.5 if a2 > 0 and b2 > 0 else None,
                zero_gradient_angle='undefined', not_a_policy_KL=True)


def calibrate_replay_coefficient(pg_norm, replay_norm, *, target_ratio=.25, max_coefficient=1.):
    if not (math.isfinite(target_ratio) and math.isfinite(max_coefficient) and target_ratio > 0 and max_coefficient > 0):
        raise ValueError('Positive nonnegative-supervision calibration bounds required')
    if not (math.isfinite(pg_norm) and math.isfinite(replay_norm) and pg_norm >= 0 and replay_norm >= 0):
        raise ValueError('Finite measured TRAIN gradient norms required')
    if pg_norm == 0 or replay_norm == 0:
        return dict(status='UNDEFINED_ZERO_GRADIENT', coefficient=None)
    raw = target_ratio*pg_norm/replay_norm
    return dict(status='LOCKED_FROM_TRAIN', coefficient=min(raw, max_coefficient), raw_coefficient=raw,
                max_bound_hit=raw > max_coefficient, target_replay_to_PG_ratio=target_ratio)
