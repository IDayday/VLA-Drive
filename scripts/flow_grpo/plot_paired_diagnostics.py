"""Plot only actual CPU diagnostics, never label them formal learning curves."""
from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path("reports/ddp_flow_grpo_paired")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
colors = ["#176cb0", "#bb3a53"]
modules = ["language", "history", "projector", "action"]
for j, variant in enumerate(("frozen", "unfrozen")):
    label = "F" if j == 0 else "U"
    comparison = json.loads(
        (root / f"numerical_committed_{variant}/summary.json").read_text()
    )["groups"]["8"]
    x = np.arange(4) + (j - 0.5) * 0.3
    axes[0].bar(
        x,
        [comparison["gradients"]["modules"][m]["relative_l2"] for m in modules],
        width=0.3,
        label=label,
        color=colors[j],
    )
    values = [
        comparison["moments"]["modules"]["optimizer_state/" + key]["relative_l2"]
        for key in ("exp_avg", "exp_avg_sq")
    ]
    values.append(comparison["updates"]["modules"]["whole_model"]["relative_l2"])
    axes[1].bar(
        np.arange(3) + (j - 0.5) * 0.3, values, width=0.3, label=label, color=colors[j]
    )
    steps = json.loads(
        (root / f"real_inner_cpu_{variant}/inner_resume.json").read_text()
    )["steps"]
    probes = [steps[0]["pre"], steps[0]["post"], steps[1]["post"]]
    lo = [p["ratio_min"] for p in probes]
    hi = [p["ratio_max"] for p in probes]
    axes[2].plot([0, 1, 2], lo, "o-", label=label + " min", color=colors[j])
    axes[2].plot([0, 1, 2], hi, "o--", label=label + " max", color=colors[j])
    axes[2].fill_between([0, 1, 2], lo, hi, alpha=0.08, color=colors[j])
axes[0].set_xticks(range(4), modules, rotation=15)
axes[0].set_title("G8/K10: chunk1 vs chunk2 gradients")
axes[1].set_xticks(
    range(3), ["Adam first moment", "second moment", "weight update"], rotation=15
)
axes[1].set_title("Actual CPU Adam / forward update")
for ax in axes[:2]:
    ax.set_yscale("log")
    ax.set_ylabel("Relative L2 difference")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
axes[2].set_yscale("log")
axes[2].set_xticks([0, 1, 2])
axes[2].set_xlabel("Optimizer updates on the SAME chain")
axes[2].set_ylabel("Current / ORIGINAL behavior surrogate ratio")
axes[2].set_title("Official advantages = 0; replay + reference")
axes[2].legend(ncol=2, fontsize=8)
axes[2].grid(axis="y", alpha=0.2)
fig.suptitle(
    "Real full-model CPU FP32 diagnostics — NOT a CUDA release or paired RL result",
    fontsize=12,
)
fig.text(
    0.5,
    0.015,
    "Both resumed second updates match continuous model, Adam, scheduler, RNG and fixed-noise ODE exactly. BF16 chunk FAIL remains unchanged.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.06, 1, 0.93))
fig.savefig(root / "cpu_diagnostics.svg")
fig.savefig(root / "cpu_diagnostics.png", dpi=180)
