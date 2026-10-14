#!/usr/bin/env python3
"""Read-only post-lock analysis; never select or train a model from dev results."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from iqe.contracts import require
from iqe.io import atomic_json, atomic_bytes, file_hash, read_json
from iqe.robust_scorer.config import CELLS, load
from iqe.specialist_study.bank import load_bank
from iqe.specialist_study.metrics import select


def analyze(config):
    final = read_json(config.root / "FINAL_RESULTS.json")
    choice = read_json(config.root / "MODEL_CHOICE.json")
    require(final["bindings"]["choice_hash"] == file_hash(config.root / "MODEL_CHOICE.json"),
            "model choice changed after evaluation")
    require(set(final["cells"]) == set(CELLS), "incomplete four-cell evaluation")
    evidence = {
        "kind": "post_lock_readonly_generalization_and_risk",
        "final_result_hash": file_hash(config.root / "FINAL_RESULTS.json"),
        "choice_hash": file_hash(config.root / "MODEL_CHOICE.json"),
        "selected_before_dev": choice["selected"]["cell"],
        "new_optimizer_steps": 0,
        "used_to_change_choice_or_recipe": False,
        "new_navtest": False,
        "score_units": "[0,1] internally; percentage points in prose and figure",
        "cells": {},
    }
    for cell in CELLS:
        rows = {}
        identity = next(row for row in choice["cells"] if row["cell"] == cell)
        for role in ("scorer_fit", "dev_report"):
            meta, bank = load_bank(config.source(), "deploy", role)
            folder = config.root / "evaluation" / cell / role
            receipt = read_json(folder / "COMPLETE.json")
            require(receipt["checksum"] == file_hash(folder / "predictions.npz"), "prediction checksum changed")
            require(receipt["identity"]["bank_hash"] == meta["hash"]
                    and receipt["identity"]["checkpoint_hash"] == identity["checkpoint_hash"]
                    and receipt["identity"]["selection_hash"] == evidence["choice_hash"],
                    "prediction/model/bank identity mismatch")
            with np.load(folder / "predictions.npz") as archive:
                prediction = {key: archive[key].copy() for key in archive.files}
            r = np.asarray(bank["scores"], dtype=np.float64)
            p = np.asarray(prediction["values"], dtype=np.float64)
            require(r.shape == p.shape and np.isfinite(r).all() and np.isfinite(p).all()
                    and np.asarray(bank["valid"]).all(), "full matched population required; do not filter failures")
            selected = select(p, bank["valid"], expert_ids=meta["provenance"]["sources"])
            index = np.arange(len(r))
            value, base = r[index, selected], r[:, 0]
            delta = value - base
            summary = final["cells"][cell] if role == "dev_report" else final["fit_diagnostics"][cell]
            require(abs(delta.mean() - summary["selected_gain_01"]) < 1e-7, "reported selection changed")
            high_zero = (base >= .8) & (value <= 1e-8)
            zero = r <= 1e-8
            negative = delta < -1e-8
            severe_cost = -delta[high_zero].sum() / len(r)
            negative_cost = np.maximum(-delta, 0).mean()
            rows[role] = {
                "scene_denominator": len(r), "source_logs": len(set(meta["source_logs"])),
                "candidate_denominator": r.size, "bank_hash": meta["hash"],
                "prediction_checksum": receipt["checksum"], "selected_gain_01": float(delta.mean()),
                "oracle_capture": summary["oracle_capture"],
                "positive_switch_count": int((delta > 1e-8).sum()),
                "negative_switch_count": int(negative.sum()),
                "non_base_tie_count": int(((selected != 0) & (np.abs(delta) <= 1e-8)).sum()),
                "high_base_denominator": int((base >= .8).sum()),
                "high_to_zero_count": int(high_zero.sum()),
                "high_to_zero_cost_01": float(severe_cost),
                "negative_switch_cost_01": float(negative_cost),
                "high_to_zero_share_of_negative_cost": float(severe_cost / negative_cost) if negative_cost else None,
                "zero_candidate_count": int(zero.sum()),
                "zero_candidate_mean_prediction": float(p[zero].mean()) if zero.any() else None,
                "zero_candidates_predicted_at_least_0_8": int((zero & (p >= .8)).sum()),
                "selected_zero_count": int((value <= 1e-8).sum()),
                "selected_zero_mean_prediction": float(p[index, selected][value <= 1e-8].mean()) if (value <= 1e-8).any() else None,
                "gain_ci": summary["gain_ci"], "safety": summary["safety"],
                "base_01": float(base.mean()), "selected_01": float(value.mean()),
                "oracle_01": float(r.max(1).mean()),
            }
            if "nonzero_logits" in prediction:
                prob = 1 / (1 + np.exp(-np.clip(prediction["nonzero_logits"].astype(float), -80, 80)))
                rows[role]["joint_nonzero_probability_on_true_zero"] = float(prob[zero].mean()) if zero.any() else None
        curves = []
        for step in config.validation_steps:
            path = config.root / 'scorers' / cell / f'validation_{step:06d}.json'
            stats = read_json(path)
            curves.append(dict(step=step, validation_hash=file_hash(path),
                               selected_01=stats['selected_01'], high_to_zero=stats['high_to_zero'],
                               completed_ranking_updates=max(0, step-config.phase_switch)))
        evidence["cells"][cell] = {
            "stage_val_selected_step": identity["step"],
            "ranking_updates_in_selected_checkpoint": max(0, identity["step"]-config.phase_switch),
            "completed_training_budget": config.updates, "stage_val_learning_curve": curves, **rows}
    destination = config.root / "qualification/generalization_audit"
    atomic_json(destination / "REPORT.json", evidence, immutable=True)
    lines = ["# 自训 Scorer 留出与风险诊断", "",
             "全量固定 D_S 与 dev 的只读事后分析。模型已在读取 dev 前锁定；本分析没有重新选模型、调阈值或训练。",
             "所有候选和场景保留，真实零分没有过滤。分数差以百分制百分点展示。", "",
             "| 组别 | D_S 收益 | dev 收益 | dev 高分转零 / Base高分 | 该类误选占全部负收益 | dev 真零分候选平均预测 |",
             "|---|---:|---:|---:|---:|---:|"]
    for cell, row in evidence["cells"].items():
        f, d = row["scorer_fit"], row["dev_report"]
        fraction = d["high_to_zero_share_of_negative_cost"]
        text = "NA" if fraction is None else f"{100*fraction:.1f}%"
        lines.append(f"| {cell} | {100*f['selected_gain_01']:+.4f} | {100*d['selected_gain_01']:+.4f} | "
                     f"{d['high_to_zero_count']} / {d['high_base_denominator']} | {text} | "
                     f"{100*d['zero_candidate_mean_prediction']:.2f} |")
    lines += ["", "高分转零分的损失计入共同场景总分母；其占比按真实损失总量计算，而非只比较错误次数。",
              "四组均完成6000步预算，但验证集选出的端点是 R00/1000、R01/3000、R10/1000、R11/1000。"
              "这些端点都在排序阶段开始前；不能将它们的开发收益归功于后3000步排序训练。"
              "继续训练后的验证变化同时混合了曝光和排序阶段变化，不能据此单独断言排序损失有害。",
              "训练集与开发集的基础质量和独立 log 数不同；这些结果定位现象，不单独证明表征、数据或目标中的某一个具有因果责任。",
              "新四组的全部训练已经结束。图和报告不增加新的模型选择机会，也不将总是回退 Base 记为 Scorer 成功。"]
    atomic_bytes(destination / "REPORT.md", ("\n".join(lines) + "\n").encode(), immutable=True)
    return evidence, destination


def figure(evidence, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = list(CELLS)
    x = np.arange(len(cells))
    fit = [100*evidence["cells"][c]["scorer_fit"]["selected_gain_01"] for c in cells]
    dev = [100*evidence["cells"][c]["dev_report"]["selected_gain_01"] for c in cells]
    ci = [evidence["cells"][c]["dev_report"]["gain_ci"] for c in cells]
    error = [[v-100*d["lower_95"] for v, d in zip(dev, ci)],
             [100*d["upper_95"]-v for v, d in zip(dev, ci)]]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax[0].bar(x-.18, fit, .34, label="D_S fit scenes", color="#4c78a8")
    ax[0].bar(x+.18, dev, .34, label="Fixed dev", color="#f58518", yerr=error, capsize=3)
    ax[0].axhline(0, color="black", linewidth=.7)
    ax[0].set(xticks=x, xticklabels=cells, ylabel="Selected PDMS minus Base (pp)")
    ax[0].legend()
    ax[1].bar(x, [evidence["cells"][c]["dev_report"]["high_to_zero_count"] for c in cells], color="#e45756")
    ax[1].set(xticks=x, xticklabels=cells, ylabel="Base >= 0.8, selected <= 1e-8 (scenes)")
    fig.suptitle("From-scratch Scorers: fixed four-candidate pool; one training seed")
    fig.supxlabel("Fit: 16,581 scenes / 211 logs; dev: 1,696 / 16. Error bars: paired-log bootstrap 95% CI.", fontsize=9)
    fig.savefig(destination / "generalization.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    evidence, destination = analyze(load(args.config))
    figure(evidence, destination)
    print(destination)
