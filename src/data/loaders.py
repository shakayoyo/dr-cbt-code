from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.special import expit


@dataclass(frozen=True)
class GlobalConfig:
    seed: int
    n_items: int
    n_groups: int
    top_k: int
    n_replicates: int
    alpha_z: float
    penalty: float
    uncertainty: dict[str, float]


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    n_samples: int
    misspec_scale: float
    source_probs: np.ndarray
    true_target_probs: np.ndarray
    nominal_target_probs: np.ndarray
    graph_mode: str


def _normalize_probs(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr / arr.sum()
    return arr


def load_experiment_config(path: str | Path) -> tuple[GlobalConfig, list[ScenarioConfig]]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    global_block = payload["global"]
    global_cfg = GlobalConfig(
        seed=int(global_block["seed"]),
        n_items=int(global_block["n_items"]),
        n_groups=int(global_block["n_groups"]),
        top_k=int(global_block["top_k"]),
        n_replicates=int(global_block["n_replicates"]),
        alpha_z=float(global_block["alpha_z"]),
        penalty=float(global_block["penalty"]),
        uncertainty={
            "l1_radius": float(global_block["uncertainty"]["l1_radius"]),
            "lower_ratio": float(global_block["uncertainty"]["lower_ratio"]),
            "upper_ratio": float(global_block["uncertainty"]["upper_ratio"]),
        },
    )
    scenarios: list[ScenarioConfig] = []
    for row in payload["scenarios"]:
        scenarios.append(
            ScenarioConfig(
                name=str(row["name"]),
                n_samples=int(row["n_samples"]),
                misspec_scale=float(row["misspec_scale"]),
                source_probs=_normalize_probs(row["source_probs"]),
                true_target_probs=_normalize_probs(row["true_target_probs"]),
                nominal_target_probs=_normalize_probs(row["nominal_target_probs"]),
                graph_mode=str(row["graph_mode"]),
            )
        )
    return global_cfg, scenarios


def _all_pairs(n_items: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_items) for j in range(i + 1, n_items)]


def _connected_pairs(n_items: int, mode: str, rng: np.random.Generator) -> list[tuple[int, int]]:
    all_pairs = _all_pairs(n_items)
    if mode == "dense":
        return all_pairs

    chain = {(i, i + 1) for i in range(n_items - 1)}
    extras_pool = [pair for pair in all_pairs if pair not in chain]
    rng.shuffle(extras_pool)
    if mode == "medium":
        extra_count = max(0, n_items)
    elif mode == "sparse":
        extra_count = max(0, n_items // 2)
    else:
        raise ValueError(f"Unknown graph mode: {mode}")
    selected = list(chain) + extras_pool[:extra_count]
    return sorted(selected)


def generate_world(global_cfg: GlobalConfig, scenario: ScenarioConfig, rng: np.random.Generator) -> dict[str, Any]:
    n_items = global_cfg.n_items
    n_groups = global_cfg.n_groups

    global_skill = rng.normal(0.0, 1.0, size=n_items)
    global_skill -= global_skill.mean()

    group_offsets = rng.normal(0.0, 0.75, size=(n_groups, n_items))
    interaction_left = rng.normal(0.0, 1.0, size=(n_groups, n_items))
    interaction_right = rng.normal(0.0, 1.0, size=(n_groups, n_items))

    pair_lists = []
    prob_tables = []
    for group in range(n_groups):
        pair_lists.append(_connected_pairs(n_items, scenario.graph_mode, rng))
        group_prob: dict[tuple[int, int], float] = {}
        for i, j in _all_pairs(n_items):
            logit = (global_skill[i] + group_offsets[group, i]) - (global_skill[j] + group_offsets[group, j])
            if scenario.misspec_scale > 0.0:
                antisym = interaction_left[group, i] * interaction_right[group, j]
                antisym -= interaction_left[group, j] * interaction_right[group, i]
                logit += 0.35 * scenario.misspec_scale * antisym
            group_prob[(i, j)] = float(expit(logit))
        prob_tables.append(group_prob)

    return {
        "global_skill": global_skill,
        "group_offsets": group_offsets,
        "pair_lists": pair_lists,
        "pair_probs": prob_tables,
    }


def sample_observations(
    global_cfg: GlobalConfig,
    scenario: ScenarioConfig,
    world: dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    groups = rng.choice(global_cfg.n_groups, size=scenario.n_samples, p=scenario.source_probs)
    rows: list[dict[str, Any]] = []
    for group in groups:
        pairs = world["pair_lists"][int(group)]
        idx = int(rng.integers(0, len(pairs)))
        i, j = pairs[idx]
        prob = world["pair_probs"][int(group)][(i, j)]
        y = int(rng.random() < prob)
        rows.append({"group": int(group), "item_i": i, "item_j": j, "y": y})
    return pd.DataFrame(rows)
