# Copyright (c) 2026 The SimWAM Authors. MIT; see LICENSE.SimWAM.
# Source: H-EmbodVis/SimWAM@1433ee7e06a62bd8b89da74455df0d82bda7b726 / src/simwam/trainer_grpo.py
# Extracted methods are unmodified; only imports/class composition are isolated.
from __future__ import annotations
import torch
from torch.optim.lr_scheduler import ConstantLR,CosineAnnealingLR,LinearLR,SequentialLR


class SimWAMGRPOTrainer:
    def _build_scheduler(
        self, scheduler_type, total_train_steps: int, warmup_steps: int = 0
    ):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), max(total_train_steps - 1, 0))
        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer, T_max=remaining_steps, eta_min=self.learning_rate * 0.01
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(
                self.optimizer, factor=1.0, total_iters=remaining_steps
            )
        else:
            raise ValueError(f"Unsupported lr_scheduler_type: {scheduler_type}.")
        if warmup_steps <= 0:
            return main_scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )


    @torch.no_grad()
    def _pg_group_mask(self, raw_std_r, rewards_mat, advantages):
        """Exclude tied groups from the local policy mean; keep the IL anchor unmasked."""
        if not self.drop_zero_adv:
            return (torch.ones_like(advantages), {})
        tied = raw_std_r.reshape(-1) <= self.adv_eps
        return ((~tied).unsqueeze(1).expand_as(rewards_mat).reshape(-1).float(), {})


    @torch.no_grad()
    def _compute_advantages(self, rewards_mat):
        """Default scalar-reward normalization; constrained training overrides this hook."""
        mean_r = rewards_mat.mean(dim=1, keepdim=True)
        raw_std_r = rewards_mat.std(dim=1, unbiased=False, keepdim=True)
        if self.adv_global_std:
            all_rewards = self.accelerator.gather(rewards_mat.reshape(-1).detach())
            std_r = all_rewards.std(unbiased=False) + self.adv_eps
        else:
            std_r = raw_std_r + self.adv_eps
        advantages = ((rewards_mat - mean_r) / std_r).reshape(-1)
        adv_diag = {}
        zero_std_frac = float(
            (raw_std_r.reshape(-1) <= self.adv_eps).float().mean().item()
        )
        adv_mask, mask_diag = self._pg_group_mask(raw_std_r, rewards_mat, advantages)
        adv_diag.update(mask_diag)
        if self.adv_q_lo > 0.0 or self.adv_q_hi < 1.0:
            lo = torch.quantile(advantages, self.adv_q_lo)
            hi = torch.quantile(advantages, self.adv_q_hi)
            advantages = advantages.clamp(min=lo, max=hi)
        advantages = advantages.detach()
        return {"advantages": advantages, "adv_mask": adv_mask.detach(),
                "adv_diag": adv_diag, "zero_std_frac": zero_std_frac,
                "nonzero_adv_frac": float((advantages.abs() > 0).float().mean().item())}


    def _effective_kl_beta(self, model):
        return float(model.grpo_kl_beta)


    def _policy_loss(self, model, rollout):
        """PPO clipping, or the optional old-policy KL gate, plus the frozen IL mean anchor.

        Reward ties mask only policy gradients. The IL anchor remains active on every group.
        """
        chain = rollout["chain"]
        timesteps = rollout["timesteps"]
        deltas = rollout["deltas"]
        advantages = rollout["advantages"]
        stoch_idx = rollout.get("stoch_idx")
        num_steps = int(timesteps.shape[0])
        kl_beta = self._effective_kl_beta(model)
        need_terms = kl_beta > 0.0 or self.kl_gate_enabled
        if need_terms:
            new_logp, terms = model.action_chain_logprobs(
                rollout["cond_bg"],
                chain,
                timesteps,
                deltas,
                stoch_idx=stoch_idx,
                return_terms=True,
            )
        else:
            terms = None
            new_logp = model.action_chain_logprobs(
                rollout["cond_bg"], chain, timesteps, deltas, stoch_idx=stoch_idx
            )
        kl_gate_div = None
        if self.kl_gate_enabled:
            old_mean, old_std = (rollout.get("old_mean"), rollout.get("old_std"))
            if old_mean is None:
                raise RuntimeError(
                    "kl_gate is on but the rollout carries no `old_mean`."
                )
            with torch.no_grad():
                d2 = (terms["mean"] - old_mean).pow(2).mean(dim=(2, 3))
                kl_gate_div = d2 / (2.0 * old_std.pow(2))[None, :]
            rollout = dict(rollout, kl_gate_div=kl_gate_div)
        if stoch_idx is None:
            denoise_idx = torch.arange(
                num_steps, device=new_logp.device, dtype=torch.float32
            )
        else:
            denoise_idx = stoch_idx.to(device=new_logp.device, dtype=torch.float32)
        discount = model.grpo_denoising_discount ** (num_steps - 1 - denoise_idx)
        adv_weighted = advantages[:, None] * discount[None, :]
        if self.adv_clip_max is not None:
            adv_weighted = adv_weighted.clamp(-self.adv_clip_max, self.adv_clip_max)
        extra = {}
        mask = rollout.get("adv_mask")
        if mask is None:
            mask = torch.ones(
                new_logp.shape[0], device=new_logp.device, dtype=new_logp.dtype
            )
        mask = mask.to(device=new_logp.device, dtype=new_logp.dtype)
        denom = mask.sum().clamp(min=1.0) * float(new_logp.shape[1])

        def _masked_mean(per_elem):
            return (per_elem * mask[:, None]).sum() / denom

        if self.ppo_clip_range is None and (not self.kl_gate_enabled):
            policy_loss = -_masked_mean(new_logp * adv_weighted)
        else:
            old_logp = rollout["old_logp"]
            ratio = torch.exp(new_logp - old_logp)
            unclipped = -adv_weighted * ratio
            if self.kl_gate_enabled:
                d_gate = rollout.get("kl_gate_div")
                if d_gate is None:
                    raise RuntimeError(
                        "grpo.train.kl_gate.enabled=true needs the rollout-time mean; `_rollout` must store `kl_gate_div` (it requires ppo machinery to be active so `old_logp` and the old mean are captured)."
                    )
                keep = d_gate <= self.kl_gate_delta
                if self.kl_gate_asymmetric:
                    moving_away = adv_weighted * (ratio - 1.0) > 0
                    keep = keep | ~moving_away
                policy_loss = _masked_mean(unclipped * keep.to(unclipped.dtype))
                with torch.no_grad():
                    extra["kl_gate_keep_frac"] = keep.float().mean().detach()
                    extra["kl_gate_div_mean"] = d_gate.mean().detach()
                    extra["kl_gate_div_max"] = d_gate.max().detach()
            else:
                clipped = -adv_weighted * ratio.clamp(
                    1.0 - self.ppo_clip_range, 1.0 + self.ppo_clip_range
                )
                policy_loss = _masked_mean(torch.max(unclipped, clipped))
            with torch.no_grad():
                extra["ratio_mean"] = ratio.mean().detach()
                if self.ppo_clip_range is not None:
                    extra["clipfrac"] = (
                        (ratio.sub(1.0).abs() > self.ppo_clip_range)
                        .float()
                        .mean()
                        .detach()
                    )
                extra["approx_kl"] = (old_logp - new_logp).mean().detach()
        with torch.no_grad():
            extra["adv_kept_frac"] = mask.mean().detach()
        bc_loss = torch.zeros((), device=policy_loss.device, dtype=policy_loss.dtype)
        if model.grpo_use_bc_loss:
            with torch.no_grad():
                ref_chain, ref_ts, ref_deltas, ref_stoch = model.sample_action_chain(
                    rollout["cond_b"],
                    rollout["cond_b"]["action_horizon"],
                    deterministic=False,
                    velocity_use_ref=True,
                )
            # Same restriction as the policy term: the BC anchor is a log-density of the reference
            # policy's samples, which only exists on the stochastic steps.
            bc_logp = model.action_chain_logprobs(
                rollout["cond_b"], ref_chain, ref_ts, ref_deltas, stoch_idx=ref_stoch
            )
            bc_loss = -bc_logp.mean()

        kl_loss = torch.zeros((), device=policy_loss.device, dtype=policy_loss.dtype)
        if terms is not None:
            with torch.no_grad():
                _, ref_terms = model.action_chain_logprobs(
                    rollout["cond_bg"],
                    chain,
                    timesteps,
                    deltas,
                    stoch_idx=stoch_idx,
                    use_ref=True,
                    return_terms=True,
                )
            d_mean_raw = (terms["mean"] - ref_terms["mean"]).pow(2)
            d_vel_raw = (terms["vel"] - ref_terms["vel"]).pow(2)
            kl_w = model.kl_weight_matrix(
                horizon=d_mean_raw.shape[2],
                action_dim=d_mean_raw.shape[3],
                device=d_mean_raw.device,
                dtype=d_mean_raw.dtype,
            )
            if kl_w is None:
                d_mean_sq = d_mean_raw.mean(dim=(2, 3))
                d_vel_sq = d_vel_raw.mean(dim=(2, 3))
            else:
                denom_w = kl_w.sum().clamp(min=1e-12)
                d_mean_sq = (d_mean_raw * kl_w).sum(dim=(2, 3)) / denom_w
                d_vel_sq = (d_vel_raw * kl_w).sum(dim=(2, 3)) / denom_w
            kl_norm = d_mean_sq / (2.0 * terms["std"].pow(2))[None, :]
            if model.grpo_kl_type == "x_kl":
                kl_loss = kl_norm.mean()
            elif model.grpo_kl_type == "x":
                kl_loss = d_mean_sq.mean()
            else:
                kl_loss = d_vel_sq.mean()
            with torch.no_grad():
                extra["kl_div"] = kl_loss.detach()
                extra["kl_x_norm"] = kl_norm.mean().detach()
                extra["kl_x"] = d_mean_sq.mean().detach()
                extra["kl_v"] = d_vel_sq.mean().detach()
                per_dim = d_mean_raw.mean(dim=(0, 1, 2))
                for i, value in enumerate(per_dim.tolist()):
                    extra[f"kl_x_dim{i}"] = torch.as_tensor(
                        value, device=policy_loss.device
                    )
        total = policy_loss + model.grpo_bc_coeff * bc_loss + kl_beta * kl_loss
        metrics = {
            "policy_loss": policy_loss.detach(),
            "bc_loss": bc_loss.detach(),
            "kl_loss": (kl_beta * kl_loss).detach(),
        }
        metrics.update(extra)
        return (total, metrics)

