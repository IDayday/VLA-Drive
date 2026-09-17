from dataclasses import dataclass, asdict
import torch
from .contracts import digest
from .math import transition, reduce_dimensions
from .observation import PolicyObservation


@dataclass(frozen=True)
class SamplingSpec:
    group_size: int = 8
    num_steps: int = 10
    noise_level: float = 0.1
    reduction: str = "flow_grpo_dimension_mean"
    candidate_chunk_size: int = 1
    transition_chunk_size: int = 1

    def __post_init__(self):
        if (
            min(self.group_size, self.candidate_chunk_size, self.transition_chunk_size)
            < 1
            or self.num_steps < 2
        ):
            raise ValueError("invalid sampling counts")
        if self.noise_level < 0:
            raise ValueError("negative noise")
        if self.transition_chunk_size != 1:
            raise ValueError("transition chunks currently contain exactly one step")

    @property
    def hash(self):
        return digest(asdict(self))


@dataclass
class RolloutBatch:
    observation: PolicyObservation
    group_ids: tuple[str, ...]
    candidate_ids: torch.Tensor
    policy_version: int
    noise_seed: int
    times: torch.Tensor
    chain: torch.Tensor  # B,G,K+1,H,D, FP32, unmodified samples
    old_elementwise_logprob: torch.Tensor
    dimension_mask: torch.Tensor
    transition_mask: torch.Tensor
    spec: SamplingSpec
    provenance: dict
    physical_trajectories: object = None
    rewards: torch.Tensor | None = None
    advantages: torch.Tensor | None = None
    score_records: object = None
    reference_mean: torch.Tensor | None = None
    reference_std: torch.Tensor | None = None

    @property
    def old_logprob(self):
        return reduce_dimensions(
            self.old_elementwise_logprob, self.dimension_mask, self.spec.reduction
        )

    @property
    def raw_final_action(self):
        return self.chain[:, :, -1]

    def validate(self, version=None):
        b, g, kp, h, d = self.chain.shape
        if (g, kp) != (self.spec.group_size, self.spec.num_steps + 1):
            raise ValueError("chain/spec mismatch")
        if (
            not torch.isfinite(self.chain).all()
            or not torch.isfinite(self.old_elementwise_logprob).all()
        ):
            raise FloatingPointError("nonfinite rollout chain/logprob")
        if self.chain.requires_grad or self.old_elementwise_logprob.requires_grad:
            raise ValueError("behavior chain and old probabilities must be fixed data")
        if version is not None and self.policy_version != version:
            raise ValueError("stale rollout policy version")
        if len(set(self.group_ids)) != b:
            raise ValueError("duplicate group ids")
        if self.spec.noise_level == 0:
            raise ValueError("ODE rollout cannot enter a policy update")


def velocity(policy, x, bucket, condition, checkpoint=False):
    head = policy.action_model
    # Follow the head parameter dtype. Probability tensors stay FP32 outside the network.
    fn = head.predict_velocity
    # PyTorch 2.5 CUDA autocast accepts float32 and casts eligible linear ops
    # (including BF16 parameter values) to FP32. This is the inherited action
    # kernel profile, NOT equivalent to enabled=False with BF16 weights.
    # CPU FP32 diagnostics already use actual FP32 parameters.
    with torch.autocast(
        x.device.type, dtype=torch.float32, enabled=x.device.type == "cuda"
    ):
        if checkpoint and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint as checkpoint_fn

            return checkpoint_fn(fn, x, bucket, condition, use_reentrant=False).float()
        return fn(x, bucket, condition).float()


@torch.no_grad()
def sample_chain(policy, observation, spec, policy_version, seed, provenance):
    if spec.noise_level == 0:
        raise ValueError("use original ODE inference for noise_level=0")
    condition = policy.encode_policy_condition(observation)
    b = len(observation.tokens)
    g = spec.group_size
    k = spec.num_steps
    generator = torch.Generator(device=condition.device).manual_seed(seed)
    x = torch.randn(
        (b, g, policy.action_model.action_horizon, policy.action_model.action_dim),
        device=condition.device,
        generator=generator,
    )
    times = torch.arange(k + 1, device=x.device, dtype=torch.float32) / k
    chain = [x]
    logs = []
    for step in range(k):
        means = []
        std = None
        for start in range(0, g, spec.candidate_chunk_size):
            end = min(g, start + spec.candidate_chunk_size)
            n = end - start
            cond = condition[:, None].expand(-1, n, -1, -1).flatten(0, 1)
            xt = x[:, start:end].flatten(0, 1)
            bucket = torch.full(
                (b * n,),
                int(step / k * policy.action_model.num_timestep_buckets),
                device=x.device,
                dtype=torch.long,
            )
            v = velocity(policy, xt, bucket, cond)
            dist = transition(
                xt,
                v,
                times[step],
                times[step + 1] - times[step],
                noise_level=spec.noise_level,
                first_dt=times[1],
            )
            means.append(dist.mean.reshape(b, n, *xt.shape[1:]))
            std = dist.std
        mean = torch.cat(means, 1)
        noise = torch.randn(x.shape, device=x.device, generator=generator)
        x = mean + std * noise
        from .math import gaussian_logprob

        logs.append(gaussian_logprob(x, mean, std))
        chain.append(x)
    result = RolloutBatch(
        observation,
        tuple(
            f"{policy_version}:{i}:{token}"
            for i, token in enumerate(observation.tokens)
        ),
        torch.arange(g),
        policy_version,
        seed,
        times,
        torch.stack(chain, 2),
        torch.stack(logs, 2),
        torch.ones((b, g, k, 1, 1), device=x.device, dtype=torch.bool),
        torch.ones((b, g, k), device=x.device, dtype=torch.bool),
        spec,
        provenance,
    )
    result.validate()
    return result


def evaluate_transitions(policy, observation, rollout, checkpoint=False):
    """One fresh VLM graph per scene, shared by chunked fixed G*K transitions."""
    rollout.validate()
    condition = policy.encode_policy_condition(observation)
    if (
        torch.is_grad_enabled()
        and any(p.requires_grad for p in policy.qwen_vl_interface.parameters())
        and not condition.requires_grad
    ):
        raise RuntimeError("current condition is detached from trainable VLM")
    b, g, kp, h, d = rollout.chain.shape
    k = kp - 1
    means = []
    stds = []
    logps = []
    velocities = []
    # Keep chunk order identical during no-grad rollout and recomputation.
    for step in range(k):
        step_means = []
        step_logs = []
        step_stds = []
        step_velocities = []
        for start in range(0, g, rollout.spec.candidate_chunk_size):
            end = min(g, start + rollout.spec.candidate_chunk_size)
            n = end - start
            cond = condition[:, None].expand(-1, n, -1, -1).flatten(0, 1)
            xt = rollout.chain[:, start:end, step].detach().flatten(0, 1)
            xn = rollout.chain[:, start:end, step + 1].detach().flatten(0, 1)
            bucket = torch.full(
                (b * n,),
                int(step / k * policy.action_model.num_timestep_buckets),
                device=xt.device,
                dtype=torch.long,
            )
            v = velocity(policy, xt, bucket, cond, checkpoint)
            step_velocities.append(v.reshape(b, n, h, d))
            dist = transition(
                xt,
                v,
                rollout.times[step],
                rollout.times[step + 1] - rollout.times[step],
                noise_level=rollout.spec.noise_level,
                first_dt=rollout.times[1],
            )
            step_means.append(dist.mean.reshape(b, n, h, d))
            step_logs.append(dist.logprob(xn).reshape(b, n, h, d))
            step_stds.append(torch.broadcast_to(dist.std, xt.shape).reshape(b, n, h, d))
        velocities.append(torch.cat(step_velocities, 1))
        means.append(torch.cat(step_means, 1))
        logps.append(torch.cat(step_logs, 1))
        stds.append(torch.cat(step_stds, 1))
    return {
        "velocity": torch.stack(velocities, 2),
        "mean": torch.stack(means, 2),
        "std": torch.stack(stds, 2),
        "elementwise_logprob": torch.stack(logps, 2),
    }
