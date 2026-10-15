# Copyright (c) 2026 The SimWAM Authors. MIT; see LICENSE.SimWAM.
# Source: H-EmbodVis/SimWAM@1433ee7e06a62bd8b89da74455df0d82bda7b726 / src/simwam/models/wan22/simwam_grpo.py
# Extracted methods are unmodified; only imports/class composition are isolated.
from __future__ import annotations
import copy
import math
import logging
from typing import Any, Dict, Optional
import torch
from torch.distributions import Normal
logger=logging.getLogger(__name__)
# Video is deliberately not a supported entry; action math is unchanged.
class VideoGRPOMixin: pass


class ActionGRPOMixin:
    def configure_grpo(self, grpo_cfg: Dict[str, Any]) -> None:
        """Configure the selected target; action mode also snapshots a frozen IL reference.

        Must be called *after* the IL checkpoint is loaded so the reference == IL policy.
        """
        target = str(grpo_cfg.get("target", "action"))
        if target == "video":
            if not isinstance(self, VideoGRPOMixin):
                raise ValueError("This backbone supports only grpo.target=action.")
            self.grpo_target = "video"
            self.action_lora_enabled = False
            self.video_lora_enabled = False
            self._configure_video_grpo(dict(grpo_cfg.get("video", {})))
            self.lora_enabled = self.video_lora_enabled
            return
        if target != "action":
            raise ValueError(f"grpo.target must be 'action' or 'video', got {target!r}")
        sample = dict(grpo_cfg.get("sample", {}))
        train = dict(grpo_cfg.get("train", {}))
        self.grpo_group_size = int(sample.get("group_size", 8))
        self.grpo_num_inference_steps = int(sample.get("num_inference_steps", 10))
        infer_shift = sample.get("infer_shift", None)
        self.grpo_infer_shift = None if infer_shift is None else float(infer_shift)
        self.grpo_sample_noise_std = float(sample.get("sample_noise_std", 0.1))
        logprob_std = sample.get("logprob_noise_std", None)
        self.grpo_logprob_noise_std = (
            self.grpo_sample_noise_std if logprob_std is None else float(logprob_std)
        )
        self.grpo_anneal_noise = bool(sample.get("anneal_noise", True))
        self.grpo_min_std = float(sample.get("min_std", 0.001))
        self.grpo_randn_clip = float(sample.get("randn_clip", 5.0))
        self.grpo_logprob_clamp_mode = str(sample.get("logprob_clamp_mode", "absolute"))
        if self.grpo_logprob_clamp_mode not in ("absolute", "relative"):
            raise ValueError(
                f"grpo.sample.logprob_clamp_mode must be 'absolute'|'relative', got {self.grpo_logprob_clamp_mode!r}"
            )
        self.grpo_logprob_clamp_width = float(sample.get("logprob_clamp_width", 7.0))
        if self.grpo_logprob_clamp_width <= 0.0:
            raise ValueError(
                f"grpo.sample.logprob_clamp_width must be > 0, got {self.grpo_logprob_clamp_width}"
            )
        final_clip = sample.get("final_action_clip", None)
        self.grpo_final_action_clip = None if final_clip is None else float(final_clip)
        self.grpo_sde_mode = str(sample.get("sde_mode", "bridge"))
        if self.grpo_sde_mode not in ("bridge", "rigorous"):
            raise ValueError(
                f"grpo.sample.sde_mode must be 'bridge'|'rigorous', got {self.grpo_sde_mode!r}"
            )
        self.grpo_noise_level = float(sample.get("noise_level", 0.1))
        sigma_steps = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=self.grpo_num_inference_steps,
            device="cpu",
            dtype=torch.float32,
            shift_override=self.grpo_infer_shift,
        )[0] / float(self.infer_action_scheduler.num_train_timesteps)
        self.grpo_sigma_max_guard = (
            float(sigma_steps[1].item()) if sigma_steps.numel() > 1 else 0.999
        )
        if self.grpo_sde_mode == "rigorous" and self.grpo_noise_level > 0.3:
            logger.warning(
                "rigorous SDE noise_level=%.2f may be too large for the %d-dim action space (std ~ noise_level*sqrt(sigma/(1-sigma)) explodes at high sigma -> reward may collapse to ~0); consider ~0.1 (flow_grpo's 0.7 is tuned for image latents). A large noise_level is only safe when the noise is confined to a FEW steps (grpo.sample.train_sde_steps) so the remaining deterministic steps can refine.",
                self.grpo_noise_level,
                int(self.action_expert.action_dim),
            )
        w = sample.get("train_sde_steps", None)
        self.grpo_train_sde_steps = None if w is None else int(w)
        if self.grpo_train_sde_steps is not None and self.grpo_train_sde_steps < 1:
            raise ValueError(
                f"grpo.sample.train_sde_steps must be >= 1 (or null for all steps), got {self.grpo_train_sde_steps}"
            )
        sde_idx = sample.get("sde_step_indices", None)
        self.grpo_sde_step_indices = (
            None if sde_idx is None else [int(i) for i in sde_idx]
        )
        self.grpo_sde_exclude_last_step = bool(
            sample.get("sde_exclude_last_step", False)
        )
        self.grpo_merge_sde_window = bool(sample.get("merge_sde_window", False))
        stoch_steps = self._stoch_step_indices(self.grpo_num_inference_steps)
        self.grpo_num_sde_steps = len(stoch_steps)
        self.grpo_merge_span = None
        if self.grpo_merge_sde_window:
            if stoch_steps != list(range(stoch_steps[0], stoch_steps[-1] + 1)):
                raise ValueError(
                    f"grpo.sample.merge_sde_window=true requires a CONTIGUOUS SDE window, got {stoch_steps}. A merged transition spans [l, r) as one Gaussian, so a gap would inject noise across steps it does not cover."
                )
            self.grpo_merge_span = (stoch_steps[0], stoch_steps[-1] + 1)
            logger.info(
                "MERGED SDE window: steps %s become ONE Gaussian transition spanning [%d, %d) -- one log-prob instead of %d, so the intra-window gradient imbalance vanishes by construction.",
                stoch_steps,
                self.grpo_merge_span[0],
                self.grpo_merge_span[1],
                len(stoch_steps),
            )
        if self.grpo_num_sde_steps < self.grpo_num_inference_steps:
            logger.info(
                "Partial ODE->SDE: %d/%d denoising steps are stochastic (indices=%s); the others run as deterministic Euler steps and carry no log-prob / policy gradient.",
                self.grpo_num_sde_steps,
                self.grpo_num_inference_steps,
                stoch_steps,
            )
        if self.grpo_sde_mode == "rigorous":
            sig_list = [float(s) for s in sigma_steps]
            worst_k, worst_coef = (None, float("inf"))
            for k in stoch_steps:
                s_k = self.grpo_sigma_max_guard if sig_list[k] >= 1.0 else sig_list[k]
                d_k = abs(
                    (sig_list[k + 1] if k + 1 < len(sig_list) else 0.0) - sig_list[k]
                )
                coef = 1.0 - self.grpo_noise_level**2 * d_k / (2.0 * (1.0 - s_k))
                if coef < worst_coef:
                    worst_k, worst_coef = (k, coef)
            if worst_coef <= 0.0:
                a_max = math.sqrt(
                    2.0
                    * (
                        1.0
                        - (
                            self.grpo_sigma_max_guard
                            if sig_list[worst_k] >= 1.0
                            else sig_list[worst_k]
                        )
                    )
                    / max(
                        abs(
                            (
                                sig_list[worst_k + 1]
                                if worst_k + 1 < len(sig_list)
                                else 0.0
                            )
                            - sig_list[worst_k]
                        ),
                        1e-12,
                    )
                )
                logger.warning(
                    "grpo.sample.noise_level=%.3f makes the SDE mean's x-coefficient %.2f (<= 0) at step k=%d: the mean INVERTS x there, so that step is no longer a perturbation of the ODE. Samples will still be produced (the remaining deterministic steps re-project them), so this fails SILENTLY. Keep noise_level < %.2f for this window, or move the window mid-schedule where a_max is largest.",
                    self.grpo_noise_level,
                    worst_coef,
                    worst_k,
                    a_max,
                )
            else:
                logger.info(
                    "SDE mean validity: min x-coefficient %.3f at k=%d (window %s, noise_level=%.3f).",
                    worst_coef,
                    worst_k,
                    stoch_steps,
                    self.grpo_noise_level,
                )
        last_step = self.grpo_num_inference_steps - 1
        if self.grpo_final_action_clip is not None and last_step in set(stoch_steps):
            logger.warning(
                "grpo.sample.final_action_clip=%.3f is applied at the LAST denoising step (k=%d), which IS a stochastic step under the current window %s -> the clip BIASES that step's log-prob. Either set final_action_clip=null, or shift the window so the last step is deterministic (e.g. sde_exclude_last_step=true).",
                self.grpo_final_action_clip,
                last_step,
                stoch_steps,
            )
        elif self.grpo_final_action_clip is not None:
            logger.info(
                "grpo.sample.final_action_clip=%.3f at k=%d (a DETERMINISTIC step under window %s) -> unbiased: that step carries no log-prob.",
                self.grpo_final_action_clip,
                last_step,
                stoch_steps,
            )
        self.grpo_denoising_discount = float(train.get("denoising_discount", 0.6))
        self.grpo_adv_q_lo = float(train.get("adv_clip_lower_quantile", 0.0))
        self.grpo_adv_q_hi = float(train.get("adv_clip_upper_quantile", 1.0))
        self.grpo_adv_eps = float(train.get("adv_eps", 1e-08))
        self.grpo_use_bc_loss = bool(train.get("use_bc_loss", False))
        self.grpo_bc_coeff = float(train.get("bc_coeff", 0.1))
        self.grpo_kl_beta = float(train.get("kl_beta", 0.0))
        if self.grpo_kl_beta < 0.0:
            raise ValueError(
                f"grpo.train.kl_beta must be >= 0, got {self.grpo_kl_beta}"
            )
        self.grpo_kl_type = str(train.get("kl_type", "x_kl"))
        if self.grpo_kl_type not in ("x_kl", "x", "v"):
            raise ValueError(
                f"grpo.train.kl_type must be 'x_kl'|'x'|'v', got {self.grpo_kl_type!r}"
            )
        self.grpo_kl_dim_weights = self._as_weight_list(
            train.get("kl_dim_weights", None), "grpo.train.kl_dim_weights"
        )
        self.grpo_kl_horizon_weights = self._as_weight_list(
            train.get("kl_horizon_weights", None), "grpo.train.kl_horizon_weights"
        )
        if self.grpo_kl_dim_weights is not None:
            action_dim = int(self.action_expert.action_dim)
            if len(self.grpo_kl_dim_weights) != action_dim:
                raise ValueError(
                    f"grpo.train.kl_dim_weights must have one entry per action dim ({action_dim}), got {len(self.grpo_kl_dim_weights)}: {self.grpo_kl_dim_weights}"
                )
        if (
            self.grpo_kl_dim_weights is not None
            or self.grpo_kl_horizon_weights is not None
        ) and self.grpo_kl_beta <= 0.0:
            logger.warning(
                "grpo.train.kl_dim_weights / kl_horizon_weights are set but kl_beta=%.4g -> the KL anchor is OFF and the weights do nothing.",
                self.grpo_kl_beta,
            )
        if str(grpo_cfg.get("target", "action")) != "action":
            raise ValueError("Only grpo.target=action is supported.")
        self.grpo_target = "action"
        self._configure_action_grpo(grpo_cfg)
        self.lora_enabled = self.action_lora_enabled


    def _configure_action_grpo(self, grpo_cfg: Dict[str, Any]) -> None:
        """Action-DiT GRPO setup: frozen IL reference + action LoRA."""
        # Freeze the entire IL policy before adding adapters. This also protects
        # alternate backbones' condition/proprio encoders outside the video DiT.
        self.eval()
        self.requires_grad_(False)
        self.ref_action_expert = copy.deepcopy(self.action_expert)
        self.ref_action_expert.eval()
        for p in self.ref_action_expert.parameters():
            p.requires_grad_(False)
        if (grpo_cfg.get("finetune") or "lora") != "lora":
            raise ValueError("Action GRPO requires grpo.finetune=lora.")
        lora = dict(grpo_cfg.get("lora", {}))
        if not lora.get("enabled", True):
            raise ValueError("Action GRPO requires grpo.lora.enabled=true.")
        self.action_finetune_mode = "lora"
        self.action_lora_enabled = True
        from .lora import apply_lora_to_module

        self.lora_r = int(lora.get("r", 16))
        self.lora_alpha = float(lora.get("alpha", 32.0))
        self.lora_dropout = float(lora.get("dropout", 0.0))
        resolve_targets = getattr(self, "_resolve_action_lora_target_modules", list)
        self.lora_target_modules = resolve_targets(
            list(lora.get("target_modules", ["q", "k", "v", "o"]))
        )
        if not self.lora_target_modules:
            raise ValueError("LoRA target_modules must not be empty.")
        num_wrapped = apply_lora_to_module(
            self.action_expert,
            target_names=self.lora_target_modules,
            r=self.lora_r,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
        )
        if num_wrapped == 0:
            raise ValueError(f"No Linear layers matched {self.lora_target_modules}.")
        logger.info(
            "Action LoRA: %d layers, r=%d alpha=%.1f; KL type=%s beta=%.4g",
            num_wrapped,
            self.lora_r,
            self.lora_alpha,
            self.grpo_kl_type,
            self.grpo_kl_beta,
        )


    def _resolve_action_lora_target_modules(self, target_names):
        """Translate attention projection names for a backbone's Action DiT."""
        return list(target_names)


    def _stoch_step_indices(self, num_steps: int) -> list:
        """Indices of the denoising steps that are stochastic (SDE); the rest run deterministic Euler.

        Priority: explicit `sde_step_indices` > window of `train_sde_steps` > all steps (legacy).
        The window is the **last** `train_sde_steps` steps (closest to data, matching the video
        branch), shifted one step earlier when `sde_exclude_last_step=true` -- both reference
        implementations drop the final step, whose |dsigma| is the largest and whose sigma_prev is 0
        (largest discretization error).
        """
        n = int(num_steps)
        if n < 1:
            raise ValueError(f"num_steps must be >= 1, got {n}")
        if self.grpo_sde_step_indices is not None:
            idx = sorted({int(i) for i in self.grpo_sde_step_indices})
            if not idx:
                raise ValueError(
                    "grpo.sample.sde_step_indices is empty; use null to keep all steps stochastic."
                )
            bad = [i for i in idx if not 0 <= i < n]
            if bad:
                raise ValueError(
                    f"grpo.sample.sde_step_indices {bad} out of range for num_inference_steps={n} (valid: 0..{n - 1})."
                )
            return idx
        if self.grpo_train_sde_steps is None:
            return list(range(n))
        end = n - 1 if self.grpo_sde_exclude_last_step else n
        if end < 1:
            raise ValueError(
                f"grpo.sample.sde_exclude_last_step=true requires num_inference_steps >= 2, got {n}."
            )
        w = min(self.grpo_train_sde_steps, end)
        return list(range(end - w, end))


    @staticmethod
    def _normalize_stoch_idx(stoch_idx, num_steps: int) -> list:
        """`stoch_idx` (tensor | sequence | None) -> a plain list of step indices. None -> all steps."""
        if stoch_idx is None:
            return list(range(int(num_steps)))
        if torch.is_tensor(stoch_idx):
            return [int(i) for i in stoch_idx.detach().cpu().tolist()]
        return [int(i) for i in stoch_idx]


    @staticmethod
    def _as_weight_list(value, name: str):
        """Config value -> list[float] of non-negative weights, or None (= uniform)."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            raise ValueError(
                f"{name} must be a LIST of weights (one per axis entry), got scalar {value}."
            )
        weights = [float(v) for v in value]
        if not weights:
            raise ValueError(f"{name} is an empty list; use null for uniform weights.")
        if any((w < 0.0 for w in weights)):
            raise ValueError(f"{name} must be non-negative, got {weights}")
        if sum(weights) <= 0.0:
            raise ValueError(
                f"{name} sums to 0, which would erase the KL anchor entirely: {weights}"
            )
        return weights


    def kl_weight_matrix(
        self, horizon: int, action_dim: int, device, dtype=torch.float32
    ):
        """`[H, A]` weight matrix for the KL anchor reduction, or None when both axes are uniform.

        The outer product of the per-horizon and per-dim weights. Cached per (H, A, device, dtype):
        it is rebuilt at most once per run. Callers reduce with `(d2 * w).sum(dim=(2,3)) / w.sum()`,
        which equals `d2.mean(dim=(2,3))` when this returns None (or when every weight is equal).
        """
        h_w, a_w = (self.grpo_kl_horizon_weights, self.grpo_kl_dim_weights)
        if h_w is None and a_w is None:
            return None
        h, a = (int(horizon), int(action_dim))
        key = (h, a, str(device), str(dtype))
        cache = getattr(self, "_kl_weight_cache", None)
        if cache is None:
            cache = {}
            self._kl_weight_cache = cache
        if key in cache:
            return cache[key]
        if h_w is not None and len(h_w) != h:
            raise ValueError(
                f"grpo.train.kl_horizon_weights must have one entry per action horizon ({h}), got {len(h_w)}: {h_w}"
            )
        if a_w is not None and len(a_w) != a:
            raise ValueError(
                f"grpo.train.kl_dim_weights must have one entry per action dim ({a}), got {len(a_w)}: {a_w}"
            )
        hv = (
            torch.tensor(h_w, device=device, dtype=dtype)
            if h_w is not None
            else torch.ones(h, device=device, dtype=dtype)
        )
        av = (
            torch.tensor(a_w, device=device, dtype=dtype)
            if a_w is not None
            else torch.ones(a, device=device, dtype=dtype)
        )
        w = hv[:, None] * av[None, :]
        cache[key] = w
        logger.info(
            "KL anchor reweighting active: horizon_weights=%s dim_weights=%s -> weighted average over [H=%d, A=%d] (sum(w)=%.3f; a uniform matrix reproduces the plain mean exactly).",
            h_w,
            a_w,
            h,
            a,
            float(w.sum().item()),
        )
        return w


    def _step_std(self, sigma_k: float, base_std: float) -> float:
        std = base_std * (float(sigma_k) if self.grpo_anneal_noise else 1.0)
        return max(std, self.grpo_min_std)


    def _sde_mean_std(
        self, x: torch.Tensor, v: torch.Tensor, sigma_k: float, delta_k: float
    ):
        """Rigorous score-corrected SDE step (flow_grpo `sd3_sde_with_logprob`), in SimWAM's
        sigma:1->0 / Delta-sigma<0 convention (already sign-aligned with flow_grpo).

        g = noise_level * sqrt(sigma/(1-sigma));  std = g * sqrt(-dsigma)
        mean = x*(1 + g^2/(2 sigma)*dsigma) + v*(1 + g^2 (1-sigma)/(2 sigma))*dsigma
        At g->0 (or deterministic) this reduces to the Euler step x + v*dsigma.
        """
        sig = self.grpo_sigma_max_guard if sigma_k >= 1.0 else sigma_k
        g = self.grpo_noise_level * math.sqrt(sig / (1.0 - sig))
        g2 = g * g
        xf, vf = (x.float(), v.float())
        mean = (
            xf * (1.0 + g2 / (2.0 * sig) * delta_k)
            + vf * (1.0 + g2 * (1.0 - sig) / (2.0 * sig)) * delta_k
        )
        std = g * math.sqrt(max(-float(delta_k), 0.0))
        return (mean, max(std, self.grpo_min_std))


    def _merged_span_mean_std(self, x_l, span, timesteps, deltas, cond, use_ref=False):
        """Gaussian transition for a MERGED window `[l, r)`: one step, `r-l` velocity evaluations.

        Integrate the deterministic Euler path across the span, express it as a single **average
        velocity** over the whole span, and feed that to the ordinary rigorous-SDE formula with the
        span's total sigma displacement:

            x_det   = Euler(x_l, steps l..r-1)                    (r-l velocity evaluations)
            dsig    = sigma_r - sigma_l                           (negative, the WHOLE span)
            v_eff   = (x_det - x_l) / dsig                        (Pave-GRPO's average-velocity view)
            mean,std = _sde_mean_std(x_l, v_eff, sigma_l, dsig)

        At `r - l == 1` this is bit-identical to the unmerged step, since `x_det = x_l + v*dsig` makes
        `v_eff == v` exactly -- which is the invariant the unit test pins.

        Returns `(mean_fp32, std, x_det)`. Gradient flows through all `r-l` velocity evaluations, so
        the merged transition still trains every step inside the window; what it removes is the
        SEPARATE log-prob per step, and with it the intra-window gradient imbalance.
        """
        sched = self.infer_action_scheduler
        l, r = span
        device, dtype = (x_l.device, x_l.dtype)
        batch_size = int(x_l.shape[0])
        x_det = x_l
        for j in range(l, r):
            t_j = (
                timesteps[j]
                .to(device=device, dtype=dtype)
                .reshape(1)
                .expand(batch_size)
            )
            v_j = self.action_velocity(x_det, t_j, cond, use_ref=use_ref)
            x_det = sched.step(v_j, deltas[j], x_det)
        sigma_l = float(timesteps[l].item()) / float(sched.num_train_timesteps)
        dsig = float(sum((float(deltas[j].item()) for j in range(l, r))))
        v_eff = (x_det - x_l) / dsig
        mean, std = self._sde_mean_std(x_l, v_eff, sigma_l, dsig)
        return (mean, std, x_det)


    @torch.no_grad()
    def sample_action_chain(
        self,
        cond: Dict[str, Any],
        action_horizon: int,
        deterministic: bool = False,
        init_actions: Optional[torch.Tensor] = None,
        velocity_use_ref: bool = False,
        generator: Optional[torch.Generator] = None,
    ):
        """Roll out the (partially) stochastic denoising chain.

        Steps in `_stoch_step_indices` take an SDE step (Gaussian transition, has a log-prob);
        all other steps take a deterministic Euler/ODE step (Dirac transition, no log-prob).
        With the legacy default (`train_sde_steps`/`sde_step_indices` both null) every step is
        stochastic, so this is byte-identical to the previous behavior.

        Returns:
            chain: [B, N+1, H, A] latents from x_0 (noise) to x_N (clean), detached.
            timesteps: [N] SimWAM scheduler timesteps (sigma*T, **decreasing**).
            deltas: [N] sigma deltas (negative).
            stoch_idx: [W] long tensor of the stochastic step indices (empty when
                `deterministic=True`). Pass it to `action_chain_logprobs` so the log-prob /
                policy gradient only covers the steps that actually have a density.
        """
        sched = self.infer_action_scheduler
        device = self.device
        dtype = getattr(self, "grpo_action_dtype", self.torch_dtype)
        batch_size = int(cond["context"].shape[0])
        action_dim = int(self.action_expert.action_dim)
        timesteps, deltas = sched.build_inference_schedule(
            num_inference_steps=self.grpo_num_inference_steps,
            device=device,
            dtype=dtype,
            shift_override=self.grpo_infer_shift,
        )
        num_steps = int(timesteps.shape[0])
        if init_actions is not None:
            x = init_actions.to(device=device, dtype=dtype)
        else:
            x = self._randn(
                (batch_size, action_horizon, action_dim), device, dtype, generator
            )
        stoch_idx = [] if deterministic else self._stoch_step_indices(num_steps)
        stoch_set = set(stoch_idx)
        merge_span = (
            getattr(self, "grpo_merge_span", None)
            if not deterministic and getattr(self, "grpo_merge_sde_window", False)
            else None
        )
        if merge_span is not None:
            l, r = merge_span
            chain = [x.clone()]
            for k in range(l):
                t_k = (
                    timesteps[k]
                    .to(device=device, dtype=dtype)
                    .reshape(1)
                    .expand(batch_size)
                )
                x = sched.step(
                    self.action_velocity(x, t_k, cond, use_ref=velocity_use_ref),
                    deltas[k],
                    x,
                )
                chain.append(x.clone())
            mean, std, x_det = self._merged_span_mean_std(
                x, merge_span, timesteps, deltas, cond, use_ref=velocity_use_ref
            )
            eps = self._randn(mean.shape, device, dtype, generator).clamp(
                -self.grpo_randn_clip, self.grpo_randn_clip
            )
            x = mean.to(dtype) + std * eps
            chain.append(x.clone())
            for k in range(r, num_steps):
                t_k = (
                    timesteps[k]
                    .to(device=device, dtype=dtype)
                    .reshape(1)
                    .expand(batch_size)
                )
                x = sched.step(
                    self.action_velocity(x, t_k, cond, use_ref=velocity_use_ref),
                    deltas[k],
                    x,
                )
                chain.append(x.clone())
            if self.grpo_final_action_clip is not None:
                x = x.clamp(-self.grpo_final_action_clip, self.grpo_final_action_clip)
                chain[-1] = x.clone()
            return (
                torch.stack(chain, dim=1).detach(),
                timesteps.detach(),
                deltas.detach(),
                torch.tensor([l], dtype=torch.long, device=device),
            )
        chain = [x.clone()]
        for k in range(num_steps):
            timestep_action = (
                timesteps[k]
                .to(device=device, dtype=dtype)
                .reshape(1)
                .expand(batch_size)
            )
            v = self.action_velocity(x, timestep_action, cond, use_ref=velocity_use_ref)
            sigma_k = float(timesteps[k].item()) / float(sched.num_train_timesteps)
            if k in stoch_set:
                if self.grpo_sde_mode == "rigorous":
                    mean, std = self._sde_mean_std(
                        x, v, sigma_k, float(deltas[k].item())
                    )
                    mean = mean.to(x.dtype)
                else:
                    mean = sched.step(v, deltas[k], x)
                    std = self._step_std(sigma_k, self.grpo_sample_noise_std)
                eps = self._randn(mean.shape, device, dtype, generator).clamp(
                    -self.grpo_randn_clip, self.grpo_randn_clip
                )
                x = mean + std * eps
            else:
                x = sched.step(v, deltas[k], x)
            if (
                not deterministic
                and k == num_steps - 1
                and (self.grpo_final_action_clip is not None)
            ):
                x = x.clamp(-self.grpo_final_action_clip, self.grpo_final_action_clip)
            chain.append(x.clone())
        chain = torch.stack(chain, dim=1)
        return (
            chain.detach(),
            timesteps.detach(),
            deltas.detach(),
            torch.tensor(stoch_idx, dtype=torch.long, device=device),
        )


    def action_chain_logprobs(
        self,
        cond: Dict[str, Any],
        chain: torch.Tensor,
        timesteps: torch.Tensor,
        deltas: torch.Tensor,
        stoch_idx=None,
        use_ref: bool = False,
        return_terms: bool = False,
    ) -> torch.Tensor:
        """Per-step Gaussian log-prob of a recorded chain under the (current/ref) policy.

        Recomputes each step's mean with the same SimWAM schedule used for sampling and
        evaluates ``Normal(mean, logprob_std).log_prob(x_{k+1})``, clamped and reduced over
        the (horizon, action_dim) dims -> [B, W]. Mirrors recogdrive `get_logprobs`.

        `stoch_idx` selects which steps to score -- pass the tensor returned by
        `sample_action_chain`. **Deterministic (ODE) steps MUST be excluded**: their transition is
        a Dirac delta, and evaluating a Gaussian density there yields ``-||mu_new - mu_old||^2 /
        (2 std^2)``, i.e. an advantage-weighted *divergence-from-theta_old* term that has nothing
        to do with the sampled action. `None` scores every step (legacy behavior, correct only
        when every step was stochastic).

        `return_terms=True` additionally returns ``{"mean", "vel", "std"}`` -- the per-step SDE mean
        [B, W, H, A] and velocity [B, W, H, A] in fp32 plus the per-step std [W] -- so the analytic
        KL penalty can be built from THIS pass instead of a second grad forward per step.
        """
        sched = self.infer_action_scheduler
        device = self.device
        dtype = getattr(self, "grpo_action_dtype", self.torch_dtype)
        batch_size = int(chain.shape[0])
        num_steps = int(timesteps.shape[0])
        steps = self._normalize_stoch_idx(stoch_idx, num_steps)
        if not steps:
            raise ValueError(
                "action_chain_logprobs got an empty step set: the chain has no stochastic transition, so there is no log-prob to compute (was it sampled with deterministic=True?)."
            )
        merge_span = (
            getattr(self, "grpo_merge_span", None)
            if getattr(self, "grpo_merge_sde_window", False)
            else None
        )
        if merge_span is not None:
            if steps != [merge_span[0]]:
                raise ValueError(
                    f"merge_sde_window is on, so stoch_idx must be [{merge_span[0]}], got {steps}. Pass the tensor `sample_action_chain` returned."
                )
            x_l, x_next = (chain[:, merge_span[0]], chain[:, merge_span[0] + 1])
            mean_f32, std, _ = self._merged_span_mean_std(
                x_l, merge_span, timesteps, deltas, cond, use_ref=use_ref
            )
            mean = mean_f32.to(dtype)
            dist = Normal(
                mean.float(), torch.tensor(std, device=device, dtype=torch.float32)
            )
            if self.grpo_logprob_clamp_mode == "relative":
                peak = -math.log(std) - 0.5 * math.log(2.0 * math.pi)
                lo, hi = (peak - self.grpo_logprob_clamp_width, peak)
            else:
                lo, hi = (-5.0, 2.0)
            lp = dist.log_prob(x_next.float()).clamp(min=lo, max=hi).mean(dim=(1, 2))
            logp = lp.unsqueeze(1)
            if not return_terms:
                return logp
            dsig = float(sum((float(deltas[j].item()) for j in range(*merge_span))))
            return (
                logp,
                {
                    "mean": mean_f32.unsqueeze(1),
                    "vel": ((mean_f32 - x_l.float()) / dsig).unsqueeze(1),
                    "std": torch.tensor([std], device=device, dtype=torch.float32),
                },
            )
        logps, means, vels, stds = ([], [], [], [])
        for k in steps:
            x_k = chain[:, k]
            x_next = chain[:, k + 1]
            timestep_action = (
                timesteps[k]
                .to(device=device, dtype=dtype)
                .reshape(1)
                .expand(batch_size)
            )
            v = self.action_velocity(x_k, timestep_action, cond, use_ref=use_ref)
            sigma_k = float(timesteps[k].item()) / float(sched.num_train_timesteps)
            if self.grpo_sde_mode == "rigorous":
                mean_f32, std = self._sde_mean_std(
                    x_k, v, sigma_k, float(deltas[k].item())
                )
            else:
                mean_f32 = sched.step(v, deltas[k], x_k).float()
                std = self._step_std(sigma_k, self.grpo_logprob_noise_std)
            mean = mean_f32.to(dtype)
            dist = Normal(
                mean.float(), torch.tensor(std, device=device, dtype=torch.float32)
            )
            if self.grpo_logprob_clamp_mode == "relative":
                peak = -math.log(std) - 0.5 * math.log(2.0 * math.pi)
                lo, hi = (peak - self.grpo_logprob_clamp_width, peak)
            else:
                lo, hi = (-5.0, 2.0)
            lp = dist.log_prob(x_next.float()).clamp(min=lo, max=hi).mean(dim=(1, 2))
            logps.append(lp)
            if return_terms:
                means.append(mean_f32)
                vels.append(v.float())
                stds.append(std)
        logp = torch.stack(logps, dim=1)
        if not return_terms:
            return logp
        return (
            logp,
            {
                "mean": torch.stack(means, dim=1),
                "vel": torch.stack(vels, dim=1),
                "std": torch.tensor(stds, device=device, dtype=torch.float32),
            },
        )


    def _randn(
        self, shape, device, dtype, generator: Optional[torch.Generator]
    ) -> torch.Tensor:
        if generator is not None:
            return torch.randn(
                shape, generator=generator, device="cpu", dtype=torch.float32
            ).to(device=device, dtype=dtype)
        return torch.randn(shape, device=device, dtype=dtype)

