"""
DriveDreamer-Policy: NavSim VLA Inference Script

Loads a trained checkpoint, runs batch inference over a NavSim split,
and writes per-token trajectory predictions as .npy files.
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from starVLA.dataloader.navsim_dataset import NavSimDataset, collate_fn
from starVLA.model.modules.action_model.multi_trajectory.config import (
    multi_trajectory_enabled,
)
import copy


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def to_device(batch: Any, device: torch.device) -> Any:
    """Recursively move tensors to *device*."""
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(v, device) for v in batch)
    return batch


def tensor_to_py(x: Any) -> Any:
    """Recursively convert tensors to Python lists."""
    if torch.is_tensor(x):
        return x.detach().float().cpu().tolist()
    if isinstance(x, dict):
        return {k: tensor_to_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [tensor_to_py(v) for v in x]
    return x


def apply_ddp_drs_path_overrides(
    config: Any, path_overrides: Optional[Dict[str, str]]
) -> Any:
    """Apply resolved CLI/environment artifact paths over a saved YAML."""

    for config_key, override in (path_overrides or {}).items():
        if override:
            OmegaConf.update(config, config_key, override, force_add=True)
    return config


def wrap_to_pi(a: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to the range (-π, π]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Action post-processing helpers
# ---------------------------------------------------------------------------

# Normalization statistics (computed from navtrain split)
_X_MEAN, _X_STD = 10.172484, 8.805105
_Y_MEAN, _Y_STD = 0.360762, 2.277741

# Per-channel 1st / 99th percentiles used for [-1, 1] normalization (ver_1226)
_Q01 = np.array([-0.01789146974507183, -0.19088272509455573, -0.1892357842470911], dtype=np.float64)
_Q99 = np.array([6.199554522088146,    0.24262804072441968,  0.1804889553518122],  dtype=np.float64)


def deal_action(action: np.ndarray) -> np.ndarray:
    """
    Convert raw delta predictions (legacy ver ≤ 1224) to (x, y, heading) poses.

    action: (B, H, 2)  — cumulative deltas in internal (-y, x) convention
    returns: (B, H, 3) — [x, y, heading_rad] in ego-centric frame
    """
    action = np.cumsum(action, axis=1)
    action[..., -1] *= 4.5912

    # Reorder axes: internal (-y, x) → world (x, y)
    a = action.copy()
    a[..., 0] = action[..., 1]   # x
    a[..., 1] = -action[..., 0]  # y
    xy = a

    B, H, _ = xy.shape
    heading = np.empty((B, H))
    heading[:, 0] = np.arctan2(xy[:, 0, 1], xy[:, 0, 0])
    d = xy[:, 1:, :] - xy[:, :-1, :]
    heading[:, 1:] = np.arctan2(d[..., 1], d[..., 0])

    # Re-anchor so index 0 uses the first displacement
    zeros = np.zeros((B, 1, 2), dtype=xy.dtype)
    final_xy = np.concatenate([zeros, xy], axis=1)[:, 1:]
    return np.concatenate([final_xy, heading[..., None]], axis=-1)


def smooth_pose_pred(pred: np.ndarray, alpha_xy: float = 0.4, alpha_heading: float = 0.4,
                     renorm: bool = True) -> np.ndarray:
    """
    Exponential-moving-average smoothing over the time axis.

    pred:    (B, H, 4) = [x, y, sin(heading), cos(heading)]
    returns: (B, H, 4) smoothed, sin/cos pair re-normalised if *renorm* is True
    """
    pred = np.asarray(pred, dtype=np.float64)
    out = pred.copy()
    for t in range(1, out.shape[1]):
        out[:, t, 0] = alpha_xy      * out[:, t, 0] + (1 - alpha_xy)      * out[:, t - 1, 0]
        out[:, t, 1] = alpha_xy      * out[:, t, 1] + (1 - alpha_xy)      * out[:, t - 1, 1]
        out[:, t, 2] = alpha_heading * out[:, t, 2] + (1 - alpha_heading) * out[:, t - 1, 2]
        out[:, t, 3] = alpha_heading * out[:, t, 3] + (1 - alpha_heading) * out[:, t - 1, 3]
        if renorm:
            r = np.sqrt(out[:, t, 2] ** 2 + out[:, t, 3] ** 2) + 1e-12
            out[:, t, 2] /= r
            out[:, t, 3] /= r
    return out


def deal_action_1225(pred: np.ndarray, scale_x: float = 4.5912, act_norm: int = 0,
                     scale_y: Optional[float] = None) -> np.ndarray:
    """
    Convert predictions from ver_1225 format to (x, y, heading) poses.

    pred:    (B, H, 4) = [dx, dy, sin(dθ), cos(dθ)]
    returns: (B, H, 3) = [x, y, θ]
    """
    pred = np.asarray(pred)
    dxdy = pred[..., :2].copy()

    if act_norm == 0:
        dxdy[..., 0] *= scale_x
    else:
        dxdy[..., 0] = dxdy[..., 0] * _X_STD + _X_MEAN
        dxdy[..., 1] = dxdy[..., 1] * _Y_STD + _Y_MEAN

    if scale_y is not None:
        dxdy[..., 1] *= scale_y

    theta = wrap_to_pi(np.arctan2(pred[..., 2], pred[..., 3]))
    return np.concatenate([dxdy, theta[..., None]], axis=-1)


def _denorm_action_batch(normed: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """De-normalise from [-1, 1] back to physical units."""
    normed = np.asarray(normed, dtype=np.float64)
    q01 = np.asarray(q01, dtype=np.float64)[None, None, :]
    q99 = np.asarray(q99, dtype=np.float64)[None, None, :]
    return (normed + 1.0) * 0.5 * (q99 - q01) + q01


def deal_action_1226(action_norm: np.ndarray, origin: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Convert normalised delta predictions (ver_1226) to absolute poses.

    action_norm: (B, H, 3) in [-1, 1]
    origin:      None or (B, 3) = [x0, y0, θ0]
    returns:     (B, H, 3) = [x, y, θ] relative to origin
    """
    deltas = _denorm_action_batch(action_norm, _Q01, _Q99)
    deltas[..., 2] = wrap_to_pi(deltas[..., 2])
    deltas = np.asarray(deltas, dtype=np.float64)
    B, H, _ = deltas.shape

    if origin is None:
        x = np.zeros(B, dtype=np.float64)
        y = np.zeros(B, dtype=np.float64)
        th = np.zeros(B, dtype=np.float64)
    else:
        origin = np.asarray(origin, dtype=np.float64)
        if origin.ndim == 1:
            origin = np.broadcast_to(origin[None, :], (B, 3))
        x, y, th = origin[:, 0].copy(), origin[:, 1].copy(), origin[:, 2].copy()

    poses = np.zeros((B, H, 3), dtype=np.float64)
    for t in range(H):
        dx_b, dy_b, dth = deltas[:, t, 0], deltas[:, t, 1], deltas[:, t, 2]
        c, s = np.cos(th), np.sin(th)
        x  += c * dx_b - s * dy_b
        y  += s * dx_b + c * dy_b
        th  = wrap_to_pi(th + dth)
        poses[:, t, 0] = x
        poses[:, t, 1] = y
        poses[:, t, 2] = th

    return poses


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class VLAAgent:
    """Wraps a trained VLA checkpoint for inference."""

    def __init__(
        self,
        ckpt_dir: str,
        model_iter: Optional[int] = None,
        device: str = "cuda",
        qwen_forward_mode: str = "auto",
        ddp_drs_path_overrides: Optional[Dict[str, str]] = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # ── Resolve config path ────────────────────────────────────────────
        if ckpt_dir.endswith(".pt"):
            cfg_path = os.path.join(str(Path(ckpt_dir).parent.parent), "config.yaml")
            if not os.path.exists(cfg_path):
                # checkpoint stored one level deeper (e.g. RL fine-tuned)
                cfg_path = os.path.join(str(Path(ckpt_dir).parent.parent.parent), "config.yaml")
        else:
            cfg_path = os.path.join(ckpt_dir, "config.yaml")

        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"config.yaml not found: {cfg_path}")
        self.model_config = OmegaConf.load(cfg_path)

        # Override base_vlm with the local path set in env.sh (BASE_VLM).
        # The saved config.yaml may contain a hardcoded path from the training machine.
        if os.environ.get("BASE_VLM"):
            OmegaConf.update(self.model_config, "framework.qwenvl.base_vlm",
                             os.environ["BASE_VLM"], force_add=True)
        # The PPU-adapted PyTorch build provides native SDPA.  The released
        # config records the authors' NVIDIA flash-attention setting, which is
        # not installed or needed in this environment.
        OmegaConf.update(
            self.model_config,
            "framework.qwenvl.attn_implementation",
            os.environ.get("VLM_ATTN_IMPLEMENTATION", "sdpa"),
            force_add=True,
        )
        apply_ddp_drs_path_overrides(
            self.model_config, ddp_drs_path_overrides
        )

        # Disable optional data loading heads during inference
        OmegaConf.update(self.model_config, "datasets.reward_data", {"load_reward_data": False}, force_add=True)
        OmegaConf.update(self.model_config, "datasets.vla_data",    {"w_neg_traj": None},         force_add=True)

        # ── Resolve weight path ────────────────────────────────────────────
        if ckpt_dir.endswith(".pt"):
            self.model_path = ckpt_dir
        else:
            final_path = os.path.join(ckpt_dir, "final_model", "pytorch_model.pt")
            flat_path   = os.path.join(ckpt_dir, "pytorch_model.pt")
            ckpt_subdir = os.path.join(ckpt_dir, "checkpoints")
            if model_iter is not None:
                chosen = os.path.join(
                    ckpt_subdir, f"steps_{int(model_iter)}_pytorch_model.pt"
                )
                if not os.path.isfile(chosen):
                    raise FileNotFoundError(
                        f"steps_{model_iter}_pytorch_model.pt not found in {ckpt_subdir}"
                    )
                self.model_path = chosen
            elif os.path.exists(final_path):
                self.model_path = final_path
            elif os.path.exists(flat_path):
                # HuggingFace flat layout: config.yaml + pytorch_model.pt in the same dir
                self.model_path = flat_path
            else:
                if not os.path.isdir(ckpt_subdir):
                    raise FileNotFoundError(f"Neither final_model/pytorch_model.pt, pytorch_model.pt, nor checkpoints/ found under: {ckpt_dir}")
                pat = re.compile(r"steps_(\d+)_pytorch_model\.pt")
                step_files = [f for f in os.listdir(ckpt_subdir) if pat.search(f)]
                if not step_files:
                    raise FileNotFoundError(f"No steps_*_pytorch_model.pt found in {ckpt_subdir}")
                chosen = sorted(step_files, key=lambda s: int(pat.search(s).group(1)))[-1]
                self.model_path = os.path.join(ckpt_subdir, chosen)

        if qwen_forward_mode not in {"auto", "legacy", "optimized"}:
            raise ValueError(
                "qwen_forward_mode must be one of: auto, legacy, optimized"
            )
        if qwen_forward_mode == "auto":
            # Experiments save the exact training source under code/.  The
            # optimized trainer bypasses the unused LM head and conditions the
            # action head on language_model.last_hidden_state.  Older/released
            # checkpoints instead used qw_out.hidden_states[-1].  Those two
            # tensors are not interchangeable, so inference must follow the
            # path used to train the selected checkpoint.
            experiment_root = Path(ckpt_dir)
            if experiment_root.suffix == ".pt":
                experiment_root = experiment_root.parent
                if experiment_root.name in {"checkpoints", "final_model"}:
                    experiment_root = experiment_root.parent
            snapshot = (
                experiment_root
                / "code"
                / "starVLA"
                / "model"
                / "framework"
                / "QwenOFT.py"
            )
            optimized_snapshot = False
            if snapshot.is_file():
                optimized_snapshot = "def _qwen_language_forward" in snapshot.read_text(
                    encoding="utf-8", errors="ignore"
                )
            freeze_modules = str(
                OmegaConf.select(
                    self.model_config, "trainer.freeze_modules", default=""
                )
            )
            saved_hidden_mode = OmegaConf.select(
                self.model_config, "qwen_hidden_state_mode", default=None
            )
            if optimized_snapshot and saved_hidden_mode == "pre_norm":
                # The stable 193514 model helper returns normalized output.
                # Use the legacy full-model path for newer pre-norm snapshots.
                qwen_forward_mode = "legacy"
            elif optimized_snapshot or "qwen_vl_interface.model.lm_head" in freeze_modules:
                qwen_forward_mode = "optimized"
            else:
                qwen_forward_mode = "legacy"
        self.qwen_forward_mode = qwen_forward_mode

        print(f"[Agent] config:  {cfg_path}")
        print(f"[Agent] weights: {self.model_path}")

        # ── Build model ────────────────────────────────────────────────────
        self.uses_ddp_drs = multi_trajectory_enabled(self.model_config)
        if self.uses_ddp_drs:
            from starVLA.model.framework.QwenPI import Qwen_PI

            print("[Agent] Loading opt-in DDP-DRS Qwen+Layerwise-DiT model")
            self.model = Qwen_PI(self.model_config)
        elif "s2" in self.model_path:
            from starVLA.model.framework.QwenOFT_s2 import Qwenvl_OFT_s2

            print("[Agent] Loading s2 model")
            self.model = Qwenvl_OFT_s2(self.model_config)
        else:
            from starVLA.model.framework.QwenOFT import Qwenvl_OFT

            self.model = Qwenvl_OFT(self.model_config, infer_not_load_wan=1)
        if not self.uses_ddp_drs:
            self.model._inference_qwen_forward_mode = self.qwen_forward_mode
        print(f"[Agent] Qwen forward mode: {self.qwen_forward_mode}")

        state = torch.load(self.model_path, map_location="cpu", weights_only=True)
        if self.uses_ddp_drs:
            from starVLA.model.modules.action_model.multi_trajectory.checkpointing import (
                load_base_checkpoint_strict,
            )

            load_base_checkpoint_strict(self.model, state)
        else:
            # Preserve the existing QwenOFT runtime-only Wan compatibility
            # exactly; it is unrelated to the opt-in DDP-DRS checkpoint path.
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            unexpected_runtime_only = [
                key for key in unexpected if not key.startswith("rgb_model.")
            ]
            print(f"[Agent] missing keys: {len(missing)}", missing[:20])
            print(
                f"[Agent] unused rgb_model keys (Wan intentionally disabled): "
                f"{len(unexpected) - len(unexpected_runtime_only)}"
            )
            print(
                f"[Agent] other unexpected keys: {len(unexpected_runtime_only)}",
                unexpected_runtime_only[:20],
            )

        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, batch: Any) -> Dict[str, Any]:
        if self.uses_ddp_drs:
            return self.model.predict_action(examples=batch)
        return self.model.predict_action_infer_1d(batch)


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def infer_and_save(
    ckpt_dir: str,
    datalist_path: str,
    out_dir: str,
    split: str = "test",
    batch_size: int = 8,
    num_workers: int = 7,
    device: str = "cuda",
    model_iter: Optional[int] = None,
    overwrite: bool = True,
    args=None,
):
    os.makedirs(out_dir, exist_ok=True)

    if not overwrite:
        with open(datalist_path, "r", encoding="utf-8") as datalist_file:
            shard_tokens = json.load(datalist_file)[args.rank::args.world_size]
        missing_tokens = [
            token
            for token in shard_tokens
            if not os.path.isfile(os.path.join(out_dir, token + ".npy"))
        ]
        if not missing_tokens:
            print(
                f"[Infer] shard {args.rank}/{args.world_size}: "
                "all predictions already exist"
            )
            return

    agent = VLAAgent(
        ckpt_dir,
        model_iter=model_iter,
        device=device,
        qwen_forward_mode=args.qwen_forward_mode,
        ddp_drs_path_overrides={
            "multi_trajectory.scene_compressor.checkpoint_path": args.scene_compressor_checkpoint_path,
            "multi_trajectory.drivor.checkpoint_path": args.drivor_checkpoint_path,
            "multi_trajectory.suprim.checkpoint_path": args.suprim_checkpoint_path,
            "multi_trajectory.suprim.vocab_path": args.suprim_vocab_path,
        },
    )
    manifest_path = os.path.join(out_dir, f"inference_manifest.rank{args.rank}.json")
    manifest_tmp = manifest_path + f".tmp-{os.getpid()}"
    with open(manifest_tmp, "w", encoding="utf-8") as manifest_file:
        json.dump(
            {
                "checkpoint_dir": str(Path(ckpt_dir).resolve()),
                "checkpoint_file": str(Path(agent.model_path).resolve()),
                "model_iter": model_iter,
                "qwen_forward_mode": agent.qwen_forward_mode,
                "split": split,
                "rank": args.rank,
                "world_size": args.world_size,
            },
            manifest_file,
            indent=2,
            sort_keys=True,
        )
        manifest_file.flush()
        os.fsync(manifest_file.fileno())
    os.replace(manifest_tmp, manifest_path)
    cfg = agent.model_config

    ver_1225 = OmegaConf.select(cfg, "ver_1225", default=False)
    act_norm  = OmegaConf.select(cfg.datasets.vla_data, "act_norm", default=0)

    # Override config to skip unused data heads at inference time
    data_cfg = copy.deepcopy(cfg)
    data_cfg.datasets.video_data.load_2d_data = 0
    data_cfg.datasets.video_data.load_3d_data = 0
    data_cfg.datasets.vla_data.act_norm = act_norm
    data_cfg.w_depth = 0
    data_cfg.enable_image_aug = 0

    dataset = NavSimDataset(
        datalist_path=datalist_path,
        split=split,
        video_data_cfg=data_cfg.datasets.video_data,
        gs_data_cfg=data_cfg.datasets.gs_data,
        reward_data_cfg=data_cfg.datasets.reward_data,
        ver_1225=ver_1225,
        dataset_cfg=data_cfg.datasets.vla_data,
        all_cfg=data_cfg,
        data_root=getattr(args, "data_root", None),
    )
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise ValueError("--rank must be in [0, --world_size)")

    indices = list(range(args.rank, len(dataset), args.world_size))
    if not overwrite:
        indices = [
            index
            for index in indices
            if not os.path.isfile(
                os.path.join(out_dir, dataset.raw_list[index] + ".npy")
            )
        ]
    if args.world_size > 1 or not overwrite:
        dataset = Subset(dataset, indices)
        print(
            f"[Infer] shard {args.rank}/{args.world_size}: "
            f"{len(dataset)} samples remaining"
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    for batch in tqdm(loader, desc="Infer"):
        batch_on_device = to_device(batch, agent.device)
        pred = agent.predict(batch_on_device)
        tokens = [sample["token"] for sample in batch]

        diagnostics = pred.get("multi_trajectory_diagnostics")
        if diagnostics is not None:
            print(
                "[DDP-DRS diagnostics] "
                + json.dumps(
                    {
                        "tokens": tokens,
                        **tensor_to_py(vars(diagnostics)),
                    },
                    sort_keys=True,
                )
            )

        action = pred["normalized_actions"]

        if ver_1225 == 1:
            if args.smooth:
                action = smooth_pose_pred(action, alpha_xy=1.0, alpha_heading=0.4)
            final_xy = deal_action_1225(action, act_norm=act_norm)
        elif ver_1225 == 2:
            final_xy = deal_action_1226(action)
        else:
            final_xy = deal_action(action)

        for i, tok in enumerate(tokens):
            output_path = os.path.join(out_dir, tok + ".npy")
            temporary_path = output_path + f".tmp-{os.getpid()}"
            with open(temporary_path, "wb") as output_file:
                np.save(output_file, final_xy[i])
                output_file.flush()
                os.fsync(output_file.fileno())
            os.replace(temporary_path, output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("NavSim VLA inference")
    p.add_argument("--ckpt_dir",      type=str, required=True)
    p.add_argument("--datalist_path", type=str, required=True)
    p.add_argument("--out_dir",       type=str, required=True)
    p.add_argument("--split",         type=str, default="test")
    p.add_argument("--batch_size",    type=int, default=8)
    p.add_argument("--num_workers",   type=int, default=7)
    p.add_argument("--device",        type=str, default="cuda")
    p.add_argument("--model_iter",    type=int, default=None)
    p.add_argument(
        "--qwen_forward_mode",
        choices=("auto", "legacy", "optimized"),
        default="auto",
        help=(
            "Match the Qwen hidden-state path used during training. auto reads "
            "the saved experiment source/config."
        ),
    )
    p.add_argument("--overwrite",     action="store_true")
    p.add_argument("--smooth",        type=int, default=0)
    p.add_argument("--rank",          type=int, default=0)
    p.add_argument("--world_size",    type=int, default=1)
    p.add_argument("--data_root",     type=str, default=None,
                   help="Root of processed navsim_dataset/. Overrides OPENSCENE_DATA_ROOT.")
    p.add_argument(
        "--scene_compressor_checkpoint_path",
        type=str,
        default=os.environ.get("DDP_DRS_SCENE_COMPRESSOR_CHECKPOINT"),
    )
    p.add_argument(
        "--drivor_checkpoint_path",
        type=str,
        default=os.environ.get("DDP_DRS_DRIVOR_CHECKPOINT"),
    )
    p.add_argument(
        "--suprim_checkpoint_path",
        type=str,
        default=os.environ.get("DDP_DRS_SUPRIM_CHECKPOINT"),
    )
    p.add_argument(
        "--suprim_vocab_path",
        type=str,
        default=os.environ.get("DDP_DRS_SUPRIM_VOCAB"),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    smooth_suffix = "-smooth" if args.smooth != 0 else ""
    checkpoint_suffix = f"-step{args.model_iter}" if args.model_iter is not None else ""
    args.out_dir = os.path.join(
        args.out_dir,
        Path(args.ckpt_dir).name + checkpoint_suffix + smooth_suffix,
        args.split,
    )
    infer_and_save(
        ckpt_dir=args.ckpt_dir,
        datalist_path=args.datalist_path,
        out_dir=args.out_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        model_iter=args.model_iter,
        overwrite=args.overwrite,
        args=args,
    )
