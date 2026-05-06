from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from src.data.loaders import GlobalConfig, ScenarioConfig
from src.models import (
    aggregate_scores,
    certified_topk_set,
    contextual_contrast,
    contextual_contrast_se,
    exact_stable_topk_set,
    fit_contextual_bt,
    fit_marginal_bt,
    fit_projection_from_prob_table,
    marginal_contrast_se,
    robust_linear_bounds,
    robust_pairwise_bounds_matrix,
)


def _topk_from_scores(scores: np.ndarray, k: int) -> list[int]:
    ranking = np.argsort(scores)[::-1]
    return [int(x) for x in ranking[:k]]


def _true_group_thetas(world: dict[str, Any], n_items: int, n_groups: int) -> np.ndarray:
    thetas = np.zeros((n_groups, n_items))
    for g in range(n_groups):
        thetas[g] = fit_projection_from_prob_table(world["pair_probs"][g], n_items)
    return thetas


def evaluate_replicate(
    global_cfg: GlobalConfig,
    scenario: ScenarioConfig,
    world: dict[str, Any],
    data: pd.DataFrame,
    replicate_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_items = global_cfg.n_items
    n_groups = global_cfg.n_groups
    top_k = global_cfg.top_k
    alpha_z = global_cfg.alpha_z
    q_true = scenario.true_target_probs
    q_nominal = scenario.nominal_target_probs

    source_mix = data["group"].value_counts(normalize=True).sort_index()
    q_source = np.array([float(source_mix.get(g, 0.0)) for g in range(n_groups)])

    true_thetas = _true_group_thetas(world, n_items, n_groups)
    true_lower_mat, _ = robust_pairwise_bounds_matrix(
        true_thetas,
        q_nominal,
        global_cfg.uncertainty["l1_radius"],
        global_cfg.uncertainty["lower_ratio"],
        global_cfg.uncertainty["upper_ratio"],
    )
    true_topk = set(
        exact_stable_topk_set(
            true_thetas,
            q_nominal,
            global_cfg.uncertainty["l1_radius"],
            global_cfg.uncertainty["lower_ratio"],
            global_cfg.uncertainty["upper_ratio"],
            top_k,
        )
    )

    marginal_fit = fit_marginal_bt(data, n_items)
    contextual_fit = fit_contextual_bt(data, n_items, n_groups, global_cfg.penalty)

    marginal_scores = marginal_fit.theta[0]
    source_scores = aggregate_scores(contextual_fit.theta, q_source)
    fixed_scores = aggregate_scores(contextual_fit.theta, q_nominal)
    robust_lower_mat, robust_upper_mat = robust_pairwise_bounds_matrix(
        contextual_fit.theta,
        q_nominal,
        global_cfg.uncertainty["l1_radius"],
        global_cfg.uncertainty["lower_ratio"],
        global_cfg.uncertainty["upper_ratio"],
    )
    robust_topk = set(
        exact_stable_topk_set(
            contextual_fit.theta,
            q_nominal,
            global_cfg.uncertainty["l1_radius"],
            global_cfg.uncertainty["lower_ratio"],
            global_cfg.uncertainty["upper_ratio"],
            top_k,
        )
    )
    fixed_ci_lower_mat = np.zeros((n_items, n_items))

    pair_rows: list[dict[str, Any]] = []
    for i, j in combinations(range(n_items), 2):
        true_contrast = contextual_contrast(true_thetas, i, j, q_true)
        true_sign = 1 if true_contrast > 0 else (-1 if true_contrast < 0 else 0)

        marginal_point = float(marginal_scores[i] - marginal_scores[j])
        marginal_se = marginal_contrast_se(marginal_fit.covariance, n_items, i, j)
        source_lower = contextual_contrast(contextual_fit.theta, i, j, q_source) - alpha_z * contextual_contrast_se(contextual_fit, i, j, q_source)
        source_upper = contextual_contrast(contextual_fit.theta, i, j, q_source) + alpha_z * contextual_contrast_se(contextual_fit, i, j, q_source)
        fixed_lower = contextual_contrast(contextual_fit.theta, i, j, q_nominal) - alpha_z * contextual_contrast_se(contextual_fit, i, j, q_nominal)
        fixed_upper = contextual_contrast(contextual_fit.theta, i, j, q_nominal) + alpha_z * contextual_contrast_se(contextual_fit, i, j, q_nominal)
        fixed_ci_lower_mat[i, j] = fixed_lower
        fixed_ci_lower_mat[j, i] = -fixed_upper

        methods = [
            {
                "method": "marginal_bt",
                "lower": marginal_point - alpha_z * marginal_se,
                "upper": marginal_point + alpha_z * marginal_se,
            },
            {
                "method": "source_contextual_bt",
                "lower": source_lower,
                "upper": source_upper,
            },
            {
                "method": "fixed_target_contextual_bt",
                "lower": fixed_lower,
                "upper": fixed_upper,
            },
            {
                "method": "fixed_target_ci_certified_bt",
                "lower": fixed_lower,
                "upper": fixed_upper,
            },
        ]
        robust_lower = float(robust_lower_mat[i, j])
        robust_upper = float(robust_upper_mat[i, j])
        robust_se = contextual_contrast_se(contextual_fit, i, j, q_nominal)
        methods.append(
            {
                "method": "ss_cbt",
                "lower": robust_lower - alpha_z * robust_se,
                "upper": robust_upper + alpha_z * robust_se,
            }
        )

        for method in methods:
            lower = float(method["lower"])
            upper = float(method["upper"])
            declared = (lower > 0.0) or (upper < 0.0)
            pred_sign = 1 if lower > 0.0 else (-1 if upper < 0.0 else 0)
            pair_rows.append(
                {
                    "scenario": scenario.name,
                    "replicate": replicate_id,
                    "method": method["method"],
                    "pair": f"{i}-{j}",
                    "true_contrast": true_contrast,
                    "lower": lower,
                    "upper": upper,
                    "coverage": float(lower <= true_contrast <= upper),
                    "declared": float(declared),
                    "correct_declaration": float(declared and pred_sign == true_sign),
                    "false_stable": float(declared and pred_sign != true_sign),
                }
            )

    topk_rows = []
    point_sets = {
        "marginal_bt": set(_topk_from_scores(marginal_scores, top_k)),
        "source_contextual_bt": set(_topk_from_scores(source_scores, top_k)),
        "fixed_target_contextual_bt": set(_topk_from_scores(fixed_scores, top_k)),
        "fixed_target_ci_certified_bt": set(certified_topk_set(fixed_ci_lower_mat, top_k)),
        "ss_cbt": robust_topk,
    }
    for method, predicted in point_sets.items():
        overlap = len(predicted & true_topk)
        precision = overlap / len(predicted) if predicted else 0.0
        recall = overlap / len(true_topk) if true_topk else 0.0
        topk_rows.append(
            {
                "scenario": scenario.name,
                "replicate": replicate_id,
                "method": method,
                "topk_precision": precision,
                "topk_recall": recall,
                "set_size": float(len(predicted)),
                "shortfall": float(max(0, top_k - len(predicted)) / top_k),
            }
        )
    return pd.DataFrame(pair_rows), pd.DataFrame(topk_rows)
