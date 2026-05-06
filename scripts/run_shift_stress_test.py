from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.loaders import generate_world, load_experiment_config, sample_observations
from src.models import (
    certified_topk_set,
    contextual_contrast,
    contextual_contrast_se,
    exact_stable_topk_set,
    fit_contextual_bt,
    fit_projection_from_prob_table,
    robust_pairwise_bounds_matrix,
)


RADIUS_GRID = [0.00, 0.08, 0.18, 0.28, 0.38]
CI_MATCH_GRID = np.linspace(0.25, 3.0, 56)
METHODS = ["fixed_target_ci_certified_bt", "fixed_target_ci_matched_bt", "ss_cbt"]
METHOD_LABELS = {
    "fixed_target_ci_certified_bt": "Fixed-Target CI Cert.",
    "fixed_target_ci_matched_bt": "CI Cert. (Matched Size)",
    "ss_cbt": "DR-CBT",
}
METHOD_COLORS = {
    "fixed_target_ci_certified_bt": "#b56576",
    "fixed_target_ci_matched_bt": "#7b6fd0",
    "ss_cbt": "#2a9d5b",
}
SCENARIO_LABELS = {
    "exact_bt_mild_shift": "Exact + Mild Shift",
    "misspecified_shift": "Misspecified Shift",
    "low_overlap_sparse": "Low Overlap + Sparse",
}


def _true_group_thetas(world: dict[str, object], n_items: int, n_groups: int) -> np.ndarray:
    thetas = np.zeros((n_groups, n_items))
    pair_probs = world["pair_probs"]
    assert isinstance(pair_probs, list)
    for g in range(n_groups):
        prob_table = pair_probs[g]
        assert isinstance(prob_table, dict)
        thetas[g] = fit_projection_from_prob_table(prob_table, n_items)
    return thetas


def _topk_stats(predicted: set[int], truth: set[int]) -> tuple[float, float, float]:
    overlap = len(predicted & truth)
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(truth) if truth else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def _fixed_target_ci_bounds(
    theta: np.ndarray,
    contextual_fit,
    q_nominal: np.ndarray,
    z_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_items = theta.shape[1]
    lower_mat = np.zeros((n_items, n_items))
    upper_mat = np.zeros((n_items, n_items))
    for i in range(n_items):
        for j in range(i + 1, n_items):
            point = contextual_contrast(theta, i, j, q_nominal)
            se = contextual_contrast_se(contextual_fit, i, j, q_nominal)
            lower = point - z_value * se
            upper = point + z_value * se
            lower_mat[i, j] = lower
            upper_mat[i, j] = upper
            lower_mat[j, i] = -upper
            upper_mat[j, i] = -lower
    return lower_mat, upper_mat


def _matched_ci_topk(
    theta: np.ndarray,
    contextual_fit,
    q_nominal: np.ndarray,
    top_k: int,
    target_size: int,
) -> tuple[np.ndarray, np.ndarray, set[int], float]:
    best = None
    for z_value in CI_MATCH_GRID:
        lower_mat, upper_mat = _fixed_target_ci_bounds(theta, contextual_fit, q_nominal, float(z_value))
        predicted = set(certified_topk_set(lower_mat, top_k))
        candidate = (
            abs(len(predicted) - target_size),
            -len(predicted),
            float(z_value),
            lower_mat,
            upper_mat,
            predicted,
        )
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    assert best is not None
    return best[3], best[4], best[5], best[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run uncertainty-radius stress tests for DR-CBT vs fixed-target CI certification")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_cfg, scenarios = load_experiment_config(args.config)
    rows: list[dict[str, float | int | str]] = []

    for scenario_idx, scenario in enumerate(scenarios):
        for replicate in range(global_cfg.n_replicates):
            seed = global_cfg.seed + 5000 * scenario_idx + replicate
            rng = np.random.default_rng(seed)
            world = generate_world(global_cfg, scenario, rng)
            data = sample_observations(global_cfg, scenario, world, rng)

            n_items = global_cfg.n_items
            n_groups = global_cfg.n_groups
            q_nominal = scenario.nominal_target_probs
            true_thetas = _true_group_thetas(world, n_items, n_groups)
            contextual_fit = fit_contextual_bt(data, n_items, n_groups, global_cfg.penalty)

            fixed_lower_mat, fixed_upper_mat = _fixed_target_ci_bounds(
                contextual_fit.theta,
                contextual_fit,
                q_nominal,
                global_cfg.alpha_z,
            )
            fixed_topk = set(certified_topk_set(fixed_lower_mat, global_cfg.top_k))

            for radius in RADIUS_GRID:
                true_lower_mat, true_upper_mat = robust_pairwise_bounds_matrix(
                    true_thetas,
                    q_nominal,
                    radius,
                    global_cfg.uncertainty["lower_ratio"],
                    global_cfg.uncertainty["upper_ratio"],
                )
                ss_lower_raw, ss_upper_raw = robust_pairwise_bounds_matrix(
                    contextual_fit.theta,
                    q_nominal,
                    radius,
                    global_cfg.uncertainty["lower_ratio"],
                    global_cfg.uncertainty["upper_ratio"],
                )
                ss_lower_mat = ss_lower_raw.copy()
                ss_upper_mat = ss_upper_raw.copy()
                for i in range(n_items):
                    for j in range(i + 1, n_items):
                        se = contextual_contrast_se(contextual_fit, i, j, q_nominal)
                        ss_lower_mat[i, j] = ss_lower_raw[i, j] - global_cfg.alpha_z * se
                        ss_upper_mat[i, j] = ss_upper_raw[i, j] + global_cfg.alpha_z * se
                        ss_lower_mat[j, i] = -ss_upper_mat[i, j]
                        ss_upper_mat[j, i] = -ss_lower_mat[i, j]

                true_topk = set(
                    exact_stable_topk_set(
                        true_thetas,
                        q_nominal,
                        radius,
                        global_cfg.uncertainty["lower_ratio"],
                        global_cfg.uncertainty["upper_ratio"],
                        global_cfg.top_k,
                    )
                )
                ss_topk = set(
                    exact_stable_topk_set(
                        contextual_fit.theta,
                        q_nominal,
                        radius,
                        global_cfg.uncertainty["lower_ratio"],
                        global_cfg.uncertainty["upper_ratio"],
                        global_cfg.top_k,
                    )
                )
                matched_lower_mat, matched_upper_mat, matched_topk, matched_z = _matched_ci_topk(
                    contextual_fit.theta,
                    contextual_fit,
                    q_nominal,
                    global_cfg.top_k,
                    target_size=len(ss_topk),
                )

                method_bounds = {
                    "fixed_target_ci_certified_bt": (fixed_lower_mat, fixed_upper_mat, fixed_topk),
                    "fixed_target_ci_matched_bt": (matched_lower_mat, matched_upper_mat, matched_topk),
                    "ss_cbt": (ss_lower_mat, ss_upper_mat, ss_topk),
                }

                n_true_stable = 0
                for i in range(n_items):
                    for j in range(i + 1, n_items):
                        if true_lower_mat[i, j] > 0 or true_upper_mat[i, j] < 0:
                            n_true_stable += 1

                for method, (lower_mat, upper_mat, predicted_topk) in method_bounds.items():
                    declared = 0
                    correct = 0
                    false_robust = 0
                    for i in range(n_items):
                        for j in range(i + 1, n_items):
                            true_sign = 1 if true_lower_mat[i, j] > 0 else (-1 if true_upper_mat[i, j] < 0 else 0)
                            pred_sign = 1 if lower_mat[i, j] > 0 else (-1 if upper_mat[i, j] < 0 else 0)
                            if pred_sign != 0:
                                declared += 1
                                if pred_sign == true_sign:
                                    correct += 1
                                else:
                                    false_robust += 1

                    pair_precision = correct / declared if declared else 0.0
                    pair_recall = correct / n_true_stable if n_true_stable else 0.0
                    false_robust_rate = false_robust / declared if declared else 0.0
                    topk_precision, topk_recall, topk_f1 = _topk_stats(predicted_topk, true_topk)
                    rows.append(
                        {
                            "scenario": scenario.name,
                            "replicate": replicate,
                            "radius": radius,
                            "method": method,
                            "robust_pair_precision": pair_precision,
                            "robust_pair_recall": pair_recall,
                            "false_robust_rate": false_robust_rate,
                            "topk_precision": topk_precision,
                            "topk_recall": topk_recall,
                            "topk_f1": topk_f1,
                            "avg_set_size": float(len(predicted_topk)),
                            "matched_ci_z": matched_z if method == "fixed_target_ci_matched_bt" else np.nan,
                        }
                    )

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(output_dir / "raw_shift_stress_metrics.csv", index=False)

    summary = raw_df.groupby(["scenario", "radius", "method"], as_index=False).agg(
        robust_pair_precision=("robust_pair_precision", "mean"),
        robust_pair_recall=("robust_pair_recall", "mean"),
        false_robust_rate=("false_robust_rate", "mean"),
        topk_precision=("topk_precision", "mean"),
        topk_recall=("topk_recall", "mean"),
        topk_f1=("topk_f1", "mean"),
        avg_set_size=("avg_set_size", "mean"),
    )
    summary.to_csv(output_dir / "summary_shift_stress_metrics.csv", index=False)

    fig, axes = plt.subplots(2, len(scenarios), figsize=(12.4, 6.0), sharex=True)
    if len(scenarios) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    metric_specs = [
        ("false_robust_rate", "False Robust-Declaration Rate", (0.0, 0.45)),
        ("topk_f1", "Robust Top-3 F1", (0.0, 1.02)),
    ]

    for col_idx, scenario in enumerate(scenarios):
        scenario_df = summary[summary["scenario"] == scenario.name].copy()
        for row_idx, (metric, ylabel, ylim) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            for method in METHODS:
                method_df = scenario_df[scenario_df["method"] == method].sort_values("radius")
                ax.plot(
                    method_df["radius"],
                    method_df[metric],
                    marker="o",
                    linewidth=2.0,
                    markersize=5.5,
                    color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method],
                )
            ax.set_ylim(*ylim)
            ax.set_xticks(RADIUS_GRID)
            ax.grid(axis="y", linestyle="--", alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if row_idx == 0:
                ax.set_title(SCENARIO_LABELS.get(scenario.name, scenario.name))
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            if row_idx == len(metric_specs) - 1:
                ax.set_xlabel(r"Uncertainty radius $\ell_1$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(output_dir / "shift_stress_suite.png", dpi=220, bbox_inches="tight")
    plt.savefig(output_dir / "shift_stress_suite.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
