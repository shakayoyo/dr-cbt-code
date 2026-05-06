from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from scipy.special import expit


@dataclass
class FitResult:
    theta: np.ndarray
    covariance: np.ndarray
    success: bool
    metadata: dict[str, Any] | None = None


def _marginal_design_matrix(item_i: np.ndarray, item_j: np.ndarray, n_items: int) -> np.ndarray:
    dim = n_items - 1
    design = np.zeros((len(item_i), dim))
    for row, (ii, jj) in enumerate(zip(item_i, item_j, strict=True)):
        if ii > 0:
            design[row, ii - 1] += 1.0
        if jj > 0:
            design[row, jj - 1] -= 1.0
    return design


def _pair_design_weights(groups: np.ndarray, item_i: np.ndarray, item_j: np.ndarray, n_groups: int, n_items: int) -> tuple[np.ndarray, np.ndarray]:
    group_counts = np.bincount(groups, minlength=n_groups).astype(float)
    group_weights = group_counts / max(group_counts.sum(), 1.0)
    pair_codes = item_i * n_items + item_j
    obs_weights = np.ones(len(groups), dtype=float)
    for g in range(n_groups):
        mask = groups == g
        if not np.any(mask):
            continue
        codes = pair_codes[mask]
        unique_codes, counts = np.unique(codes, return_counts=True)
        support_size = max(len(unique_codes), 1)
        weight_map = {int(code): float(group_counts[g] / (support_size * count)) for code, count in zip(unique_codes, counts, strict=True)}
        obs_weights[mask] = np.asarray([weight_map[int(code)] for code in codes], dtype=float)
    return obs_weights, group_weights


def _contextual_parameterization(group_weights: np.ndarray, n_items: int) -> tuple[int, list[int], dict[int, int], np.ndarray]:
    positive_groups = [int(g) for g, weight in enumerate(group_weights) if weight > 0.0]
    if not positive_groups:
        raise ValueError("At least one group must have positive source support.")
    anchor_group = positive_groups[-1]
    free_groups = [g for g in range(len(group_weights)) if g != anchor_group]
    free_group_index = {g: idx for idx, g in enumerate(free_groups)}
    anchor_coeff = np.zeros(len(group_weights))
    for g in free_groups:
        anchor_coeff[g] = group_weights[g] / group_weights[anchor_group]
    return anchor_group, free_groups, free_group_index, anchor_coeff


def _contextual_design_matrix(
    groups: np.ndarray,
    item_i: np.ndarray,
    item_j: np.ndarray,
    n_items: int,
    group_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, list[int], dict[int, int], np.ndarray]:
    dim_base = n_items - 1
    anchor_group, free_groups, free_group_index, anchor_coeff = _contextual_parameterization(group_weights, n_items)
    dim_delta = len(free_groups) * (n_items - 1)
    dim = dim_base + dim_delta
    design = np.zeros((len(groups), dim))

    def delta_offset(group: int, item: int) -> int:
        return dim_base + free_group_index[group] * (n_items - 1) + (item - 1)

    for row, (g, ii, jj) in enumerate(zip(groups, item_i, item_j, strict=True)):
        if ii > 0:
            design[row, ii - 1] += 1.0
        if jj > 0:
            design[row, jj - 1] -= 1.0

        if g != anchor_group:
            if ii > 0:
                design[row, delta_offset(g, ii)] += 1.0
            if jj > 0:
                design[row, delta_offset(g, jj)] -= 1.0
            continue

        for h in free_groups:
            coeff = -anchor_coeff[h]
            if ii > 0:
                design[row, delta_offset(h, ii)] += coeff
            if jj > 0:
                design[row, delta_offset(h, jj)] -= coeff

    return design, group_weights, anchor_group, free_groups, free_group_index, anchor_coeff


def _contextual_penalty_matrix(
    n_items: int,
    free_groups: list[int],
    free_group_index: dict[int, int],
    anchor_coeff: np.ndarray,
) -> np.ndarray:
    dim_base = n_items - 1
    dim = dim_base + len(free_groups) * (n_items - 1)
    penalty_matrix = np.zeros((dim, dim))
    if not free_groups:
        return penalty_matrix
    for item in range(1, n_items):
        for g in free_groups:
            idx_g = dim_base + free_group_index[g] * (n_items - 1) + (item - 1)
            penalty_matrix[idx_g, idx_g] += 1.0 + anchor_coeff[g] ** 2
            for h in free_groups:
                if h == g:
                    continue
                idx_h = dim_base + free_group_index[h] * (n_items - 1) + (item - 1)
                penalty_matrix[idx_g, idx_h] += anchor_coeff[g] * anchor_coeff[h]
    return penalty_matrix


def _unpack_contextual_theta(
    x: np.ndarray,
    n_items: int,
    n_groups: int,
    anchor_group: int,
    free_groups: list[int],
    free_group_index: dict[int, int],
    anchor_coeff: np.ndarray,
) -> np.ndarray:
    dim_base = n_items - 1
    base = np.zeros(n_items)
    base[1:] = x[:dim_base]
    delta = np.zeros((n_groups, n_items))
    for g in free_groups:
        start = dim_base + free_group_index[g] * (n_items - 1)
        delta[g, 1:] = x[start : start + (n_items - 1)]
    for g in free_groups:
        delta[anchor_group, 1:] -= anchor_coeff[g] * delta[g, 1:]
    theta = base[None, :] + delta
    theta[:, 0] = 0.0
    return theta


def _sandwich_covariance(design: np.ndarray, obs_weights: np.ndarray, probabilities: np.ndarray, outcomes: np.ndarray, penalty_matrix: np.ndarray | None = None) -> np.ndarray:
    if penalty_matrix is None:
        penalty_matrix = np.zeros((design.shape[1], design.shape[1]))
    hessian = design.T @ ((obs_weights * probabilities * (1.0 - probabilities))[:, None] * design) + penalty_matrix
    score_terms = (obs_weights * (outcomes - probabilities))[:, None] * design
    meat = score_terms.T @ score_terms
    hessian_inv = np.linalg.pinv(hessian + 1e-8 * np.eye(hessian.shape[0]))
    return hessian_inv @ meat @ hessian_inv


def fit_marginal_bt(data: pd.DataFrame, n_items: int) -> FitResult:
    obs = data[["item_i", "item_j", "y"]].to_numpy(dtype=float)
    item_i = obs[:, 0].astype(int)
    item_j = obs[:, 1].astype(int)
    y = obs[:, 2]
    design = _marginal_design_matrix(item_i, item_j, n_items)
    dim = design.shape[1]

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        z = design @ x
        p = np.clip(expit(z), 1e-8, 1.0 - 1e-8)
        loss = -np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        grad = design.T @ (p - y)
        return float(loss), grad

    result = minimize(
        fun=lambda x: objective(x)[0],
        x0=np.zeros(dim),
        jac=lambda x: objective(x)[1],
        method="L-BFGS-B",
    )
    theta = np.zeros(n_items)
    theta[1:] = result.x
    probabilities = np.clip(expit(design @ result.x), 1e-8, 1.0 - 1e-8)
    covariance = _sandwich_covariance(design, np.ones(len(y)), probabilities, y)
    return FitResult(theta=theta[None, :], covariance=covariance, success=bool(result.success))


def fit_contextual_bt(data: pd.DataFrame, n_items: int, n_groups: int, penalty: float) -> FitResult:
    obs = data[["group", "item_i", "item_j", "y"]].to_numpy(dtype=float)
    groups = obs[:, 0].astype(int)
    item_i = obs[:, 1].astype(int)
    item_j = obs[:, 2].astype(int)
    y = obs[:, 3]
    obs_weights, group_weights = _pair_design_weights(groups, item_i, item_j, n_groups, n_items)
    design, _, anchor_group, free_groups, free_group_index, anchor_coeff = _contextual_design_matrix(
        groups,
        item_i,
        item_j,
        n_items,
        group_weights,
    )
    penalty_matrix = _contextual_penalty_matrix(n_items, free_groups, free_group_index, anchor_coeff)
    dim = design.shape[1]

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        z = design @ x
        p = np.clip(expit(z), 1e-8, 1.0 - 1e-8)
        loss = -np.sum(obs_weights * (y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
        loss += penalty * float(x @ penalty_matrix @ x)
        grad = design.T @ (obs_weights * (p - y))
        grad += 2.0 * penalty * (penalty_matrix @ x)
        return float(loss), grad

    result = minimize(
        fun=lambda x: objective(x)[0],
        x0=np.zeros(dim),
        jac=lambda x: objective(x)[1],
        method="L-BFGS-B",
    )
    theta = _unpack_contextual_theta(
        result.x,
        n_items,
        n_groups,
        anchor_group,
        free_groups,
        free_group_index,
        anchor_coeff,
    )
    probabilities = np.clip(expit(design @ result.x), 1e-8, 1.0 - 1e-8)
    covariance = _sandwich_covariance(design, obs_weights, probabilities, y, penalty_matrix=2.0 * penalty * penalty_matrix)
    metadata = {
        "group_weights": group_weights,
        "anchor_group": anchor_group,
        "free_groups": free_groups,
        "free_group_index": free_group_index,
    }
    return FitResult(theta=theta, covariance=covariance, success=bool(result.success), metadata=metadata)


def fit_projection_from_prob_table(prob_table: dict[tuple[int, int], float], n_items: int) -> np.ndarray:
    pairs = sorted(prob_table)
    item_i = np.asarray([i for i, _ in pairs], dtype=int)
    item_j = np.asarray([j for _, j in pairs], dtype=int)
    y = np.asarray([prob_table[pair] for pair in pairs], dtype=float)
    dim = n_items - 1

    def unpack(x: np.ndarray) -> np.ndarray:
        theta = np.zeros(n_items)
        theta[1:] = x
        return theta

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        theta = unpack(x)
        z = theta[item_i] - theta[item_j]
        p = np.clip(expit(z), 1e-8, 1.0 - 1e-8)
        loss = -np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        resid = p - y
        grad = np.zeros(dim)
        np.add.at(grad, item_i - 1, resid)
        np.add.at(grad, item_j - 1, -resid)
        return float(loss), grad

    result = minimize(
        fun=lambda x: objective(x)[0],
        x0=np.zeros(dim),
        jac=lambda x: objective(x)[1],
        method="L-BFGS-B",
    )
    return unpack(result.x)


def aggregate_scores(theta: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.asarray(q) @ np.asarray(theta)


def contextual_contrast(theta: np.ndarray, i: int, j: int, q: np.ndarray) -> float:
    return float(np.dot(q, theta[:, i] - theta[:, j]))


def contextual_contrast_se(fit: FitResult, i: int, j: int, q: np.ndarray) -> float:
    if fit.metadata is None:
        raise ValueError("Contextual contrast standard errors require contextual fit metadata.")
    n_items = fit.theta.shape[1]
    group_weights = np.asarray(fit.metadata["group_weights"])
    anchor_group = int(fit.metadata["anchor_group"])
    free_groups = [int(g) for g in fit.metadata["free_groups"]]
    free_group_index = {int(k): int(v) for k, v in fit.metadata["free_group_index"].items()}

    dim_base = n_items - 1
    grad = np.zeros(fit.covariance.shape[0])
    if i > 0:
        grad[i - 1] += 1.0
    if j > 0:
        grad[j - 1] -= 1.0
    for g in free_groups:
        coeff = float(q[g] - q[anchor_group] * group_weights[g] / group_weights[anchor_group])
        if i > 0:
            grad[dim_base + free_group_index[g] * (n_items - 1) + (i - 1)] += coeff
        if j > 0:
            grad[dim_base + free_group_index[g] * (n_items - 1) + (j - 1)] -= coeff
    variance = float(grad @ fit.covariance @ grad)
    return float(np.sqrt(max(variance, 0.0)))


def marginal_contrast_se(covariance: np.ndarray, n_items: int, i: int, j: int) -> float:
    grad = np.zeros(n_items - 1)
    if i > 0:
        grad[i - 1] += 1.0
    if j > 0:
        grad[j - 1] -= 1.0
    variance = float(grad @ covariance @ grad)
    return float(np.sqrt(max(variance, 0.0)))


def _linprog_bounds_template(q0: np.ndarray, l1_radius: float, lower_ratio: float, upper_ratio: float) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]:
    n_groups = len(q0)
    if np.any(q0 < -1e-10):
        raise ValueError("Target probabilities must be nonnegative.")
    if not np.isclose(float(np.sum(q0)), 1.0, atol=1e-8):
        raise ValueError("Target probabilities must sum to one.")
    lower = np.clip(lower_ratio * q0, 0.0, 1.0)
    upper = np.clip(upper_ratio * q0, 0.0, 1.0)
    zero_support = q0 <= 1e-12
    lower[zero_support] = 0.0
    upper[zero_support] = 0.0
    q_bounds = [(float(lower[g]), float(min(1.0, upper[g]))) for g in range(n_groups)]
    t_bounds = [(0.0, None) for _ in range(n_groups)]

    a_eq = np.zeros((1, 2 * n_groups))
    a_eq[0, :n_groups] = 1.0
    b_eq = np.array([1.0])

    a_ub = []
    b_ub = []
    for g in range(n_groups):
        row_upper = np.zeros(2 * n_groups)
        row_upper[g] = 1.0
        row_upper[n_groups + g] = -1.0
        a_ub.append(row_upper)
        b_ub.append(q0[g])

        row_lower = np.zeros(2 * n_groups)
        row_lower[g] = -1.0
        row_lower[n_groups + g] = -1.0
        a_ub.append(row_lower)
        b_ub.append(-q0[g])

    row_budget = np.zeros(2 * n_groups)
    row_budget[n_groups:] = 1.0
    a_ub.append(row_budget)
    b_ub.append(l1_radius)
    return a_eq, b_eq, q_bounds + t_bounds, (np.asarray(a_ub), np.asarray(b_ub))


def robust_linear_bounds(c: np.ndarray, q0: np.ndarray, l1_radius: float, lower_ratio: float, upper_ratio: float) -> tuple[float, float]:
    n_groups = len(q0)
    a_eq, b_eq, bounds, (a_ub, b_ub) = _linprog_bounds_template(q0, l1_radius, lower_ratio, upper_ratio)
    objective = np.concatenate([c, np.zeros(n_groups)])

    lower_result = linprog(c=objective, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    upper_result = linprog(c=-objective, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not lower_result.success or not upper_result.success:
        raise RuntimeError("Robust linear program failed")
    return float(lower_result.fun), float(-upper_result.fun)


def robust_score_bounds(theta: np.ndarray, q0: np.ndarray, l1_radius: float, lower_ratio: float, upper_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    lowers = []
    uppers = []
    for item in range(theta.shape[1]):
        lo, hi = robust_linear_bounds(theta[:, item], q0, l1_radius, lower_ratio, upper_ratio)
        lowers.append(lo)
        uppers.append(hi)
    return np.asarray(lowers), np.asarray(uppers)


def robust_pairwise_bounds_matrix(
    theta: np.ndarray,
    q0: np.ndarray,
    l1_radius: float,
    lower_ratio: float,
    upper_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_items = theta.shape[1]
    lower = np.zeros((n_items, n_items))
    upper = np.zeros((n_items, n_items))
    for i in range(n_items):
        for j in range(i + 1, n_items):
            lo, hi = robust_linear_bounds(theta[:, i] - theta[:, j], q0, l1_radius, lower_ratio, upper_ratio)
            lower[i, j] = lo
            upper[i, j] = hi
            lower[j, i] = -hi
            upper[j, i] = -lo
    return lower, upper


def certified_topk_set(lower_pairwise: np.ndarray, k: int) -> list[int]:
    n_items = lower_pairwise.shape[0]
    stable_items: list[int] = []
    for i in range(n_items):
        certified_wins = 0
        for j in range(n_items):
            if i == j:
                continue
            if lower_pairwise[i, j] > 0:
                certified_wins += 1
        if certified_wins >= n_items - k:
            stable_items.append(i)
    return stable_items


def exact_stable_topk_set(
    theta: np.ndarray,
    q0: np.ndarray,
    l1_radius: float,
    lower_ratio: float,
    upper_ratio: float,
    k: int,
    feasibility_tol: float = 1e-9,
) -> list[int]:
    n_items = theta.shape[1]
    if k >= n_items:
        return list(range(n_items))

    a_eq, b_eq, bounds, (base_a_ub, base_b_ub) = _linprog_bounds_template(
        q0,
        l1_radius,
        lower_ratio,
        upper_ratio,
    )
    zero_objective = np.zeros(base_a_ub.shape[1], dtype=float)
    stable_items: list[int] = []

    for item in range(n_items):
        opponents = [idx for idx in range(n_items) if idx != item]
        item_is_stable = True
        for subset in combinations(opponents, k):
            extra_rows = []
            extra_rhs = []
            for rival in subset:
                row = np.zeros_like(zero_objective)
                row[: len(q0)] = -(theta[:, rival] - theta[:, item])
                extra_rows.append(row)
                extra_rhs.append(-feasibility_tol)
            a_ub = np.vstack([base_a_ub, np.asarray(extra_rows, dtype=float)])
            b_ub = np.concatenate([base_b_ub, np.asarray(extra_rhs, dtype=float)])
            result = linprog(
                c=zero_objective,
                A_ub=a_ub,
                b_ub=b_ub,
                A_eq=a_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
            )
            if result.success:
                item_is_stable = False
                break
        if item_is_stable:
            stable_items.append(item)

    return stable_items
