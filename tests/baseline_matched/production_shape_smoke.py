"""Manual Gate-4 smoke with production planner dimensions and stub labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starVLA.model.framework.QwenPI_DrivoRSuprim import QwenPIDrivoRSuprim
from starVLA.model.modules.trajectory_scorer.losses import SUPRIM_METRICS
from starVLA.training.config_loader import load_training_config
from starVLA.training.hierarchical_schedule import HierarchicalTrainingSchedule
from starVLA.training.navsim_metric_supervisor import StubDynamicMetricSupervisor


class ProductionShapeQwenMock(nn.Module):
    def __init__(self, width=2048):
        super().__init__()
        self.model = nn.Module()
        self.model.config = SimpleNamespace(hidden_size=width)
        self.model.visual = nn.Linear(width, width, bias=False)
        self.model.embed_tokens = nn.Linear(width, width, bias=False)
        self.model.language_model = nn.Linear(width, width, bias=False)
        self.model.lm_head = nn.Linear(width, width, bias=False)
        self.model.lm_head.weight = self.model.embed_tokens.weight


def mock_features(framework, examples):
    core = framework.qwen_vl_interface.model
    batch = len(examples)
    width = core.config.hidden_size
    base = torch.randn(
        batch,
        32,
        width,
        device=core.language_model.weight.device,
        dtype=core.language_model.weight.dtype,
    ) * 0.02
    hidden = core.language_model(core.embed_tokens(base) + core.visual(base))
    return hidden[:, :8], hidden, torch.ones(
        batch, 32, dtype=torch.long, device=hidden.device
    )


class StaticTargetStub:
    def get(self, tokens, *, device, dtype):
        target = torch.linspace(0.05, 0.95, 8192, device=device, dtype=dtype)
        return {
            name: target[None].expand(len(tokens), -1)
            for name in SUPRIM_METRICS
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--config",
        default="starVLA/config/training/qwenpi_matched_b4_full.yaml",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Gate-4 production shape smoke requires CUDA")
    device = torch.device(args.device)
    config = load_training_config(args.config)
    # Injected vocabulary/labels make real NAVSIM assets unnecessary.
    config.framework.hierarchical_scorer.joint.vocab_path = None
    model = QwenPIDrivoRSuprim(
        config,
        qwen_vl_interface=ProductionShapeQwenMock(),
        static_vocab=torch.randn(8192, 40, 3) * 0.1,
        static_score_store=StaticTargetStub(),
        qwen_feature_extractor=mock_features,
    ).to(device=device, dtype=torch.bfloat16).eval()
    examples = [
        {
            "image": [object()],
            "lang": "keep straight",
            "state": np.zeros((1, 4), dtype=np.float32),
            "action": np.stack(
                (
                    np.linspace(-0.2, 0.5, 8, dtype=np.float32),
                    np.zeros(8, dtype=np.float32),
                    np.zeros(8, dtype=np.float32),
                    np.ones(8, dtype=np.float32),
                ),
                axis=-1,
            ),
            "token": "stub-token",
        }
    ]
    schedule = HierarchicalTrainingSchedule(
        progress=1.0,
        dynamic_enabled=True,
        num_dynamic_candidates=64,
        dynamic_topm=32,
        lambda_flow=1.0,
        lambda_drivor=1.0,
        lambda_suprim_coarse=1.0,
        lambda_suprim_fine=1.0,
    )
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(
            examples,
            training_schedule=schedule,
            metric_supervisor=StubDynamicMetricSupervisor(),
        )
    torch.cuda.synchronize(device)
    latency = time.perf_counter() - start
    report = {
        "device": str(device),
        "latency_seconds": latency,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        "flow_loss_shape": list(output["losses"]["flow"].shape),
        "dynamic_topm_indices": list(
            output["predictions"]["dynamic_topm_indices"].shape
        ),
        "coarse_topk_indices": list(
            output["predictions"]["coarse_topk_indices"].shape
        ),
        "selected_trajectory_8": list(
            output["predictions"]["selected_trajectory_8"].shape
        ),
        "dit_hidden": model.action_model.hidden_size,
        "dit_layers": len(model.action_model.model.transformer_blocks),
        "qformer_queries": model.scene_encoder.num_queries,
        "qformer_dim": model.scene_encoder.output_dim,
        "dynamic_candidates": model.num_dynamic_candidates,
        "candidate_chunk_size": model.candidate_chunk_size,
        "static_vocab": model._static_vocab_size,
        "joint_candidates": model._static_vocab_size + model.final_dynamic_topm,
        "coarse_topk": model.joint_coarse_topk,
        "refinement_layers": model.refinement_layers,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
