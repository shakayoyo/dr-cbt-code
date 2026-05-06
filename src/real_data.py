from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset

from src.models import (
    aggregate_scores,
    certified_topk_set,
    exact_stable_topk_set,
    fit_marginal_bt,
    fit_contextual_bt,
    contextual_contrast_se,
    robust_linear_bounds,
    robust_pairwise_bounds_matrix,
)


QUESTION_URL = "https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts/raw/main/raw/question.jsonl"
ARENA_DATASET = "lmarena-ai/arena-human-preference-140k"
ATP_MATCH_URLS = {
    2021: "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2021.csv",
    2022: "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2022.csv",
    2023: "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2023.csv",
    2024: "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2024.csv",
}
WTA_MATCH_URLS = {
    2021: "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_2021.csv",
    2022: "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_2022.csv",
    2023: "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_2023.csv",
    2024: "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_2024.csv",
}


def load_mt_bench_real_data(cache_dir: str | Path | None = None) -> tuple[pd.DataFrame, list[str], list[str]]:
    ds_judgments = load_dataset("lmsys/mt_bench_human_judgments", split="human", cache_dir=str(cache_dir) if cache_dir else None)
    ds_questions = load_dataset("json", data_files=QUESTION_URL, split="train", cache_dir=str(cache_dir) if cache_dir else None)

    judgments = ds_judgments.to_pandas()
    questions = ds_questions.to_pandas()[["question_id", "category"]]
    df = judgments.merge(questions, on="question_id", how="left")
    df = df[df["winner"].isin(["model_a", "model_b"])].copy()

    models = sorted(set(df["model_a"]).union(set(df["model_b"])))
    categories = sorted(df["category"].dropna().unique().tolist())
    model_to_idx = {name: idx for idx, name in enumerate(models)}
    cat_to_idx = {name: idx for idx, name in enumerate(categories)}

    df["group"] = df["category"].map(cat_to_idx)
    df["item_i"] = df["model_a"].map(model_to_idx)
    df["item_j"] = df["model_b"].map(model_to_idx)
    df["y"] = (df["winner"] == "model_a").astype(int)
    return df[["question_id", "category", "group", "item_i", "item_j", "y"]].reset_index(drop=True), models, categories


def load_arena_language_real_data(
    cache_dir: str | Path | None = None,
    languages: list[str] | None = None,
    top_n_models: int = 8,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    ds = load_dataset(ARENA_DATASET, split="train", cache_dir=str(cache_dir) if cache_dir else None)
    df = ds.to_pandas()
    df = df[df["winner"].isin(["model_a", "model_b"])].copy()

    if languages is None:
        languages = ["en", "pl", "ru", "zh", "de", "ja"]
    df = df[df["language"].isin(languages)].copy()

    model_counts = pd.concat([df["model_a"], df["model_b"]]).value_counts()
    models = model_counts.head(top_n_models).index.tolist()
    df = df[df["model_a"].isin(models) & df["model_b"].isin(models)].copy()

    languages = [lang for lang in languages if lang in set(df["language"])]
    model_to_idx = {name: idx for idx, name in enumerate(models)}
    lang_to_idx = {name: idx for idx, name in enumerate(languages)}

    df["group"] = df["language"].map(lang_to_idx)
    df["item_i"] = df["model_a"].map(model_to_idx)
    df["item_j"] = df["model_b"].map(model_to_idx)
    df["y"] = (df["winner"] == "model_a").astype(int)
    return df[["language", "group", "item_i", "item_j", "y"]].reset_index(drop=True), models, languages


def load_atp_surface_real_data(
    years: list[int] | None = None,
    surfaces: list[str] | None = None,
    top_n_players: int = 20,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    if years is None:
        years = [2021, 2022, 2023, 2024]
    if surfaces is None:
        surfaces = ["Hard", "Clay", "Grass"]

    frames = [pd.read_csv(ATP_MATCH_URLS[year]) for year in years]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["surface"].isin(surfaces)].copy()
    decisive_mask = ~df["score"].fillna("").str.contains(
        "W/O|RET|DEF|Played and unfinished|Walkover",
        regex=True,
    )
    df = df[decisive_mask].copy()

    player_counts = pd.concat([df["winner_name"], df["loser_name"]]).value_counts()
    players = player_counts.head(top_n_players).index.tolist()
    df = df[df["winner_name"].isin(players) & df["loser_name"].isin(players)].copy()

    surfaces = [surface for surface in surfaces if surface in set(df["surface"])]
    player_to_idx = {name: idx for idx, name in enumerate(players)}
    surface_to_idx = {name: idx for idx, name in enumerate(surfaces)}

    df["group"] = df["surface"].map(surface_to_idx)
    df["item_i"] = df["winner_name"].map(player_to_idx)
    df["item_j"] = df["loser_name"].map(player_to_idx)
    df["y"] = 1
    return (
        df[["surface", "group", "item_i", "item_j", "y"]].reset_index(drop=True),
        players,
        surfaces,
    )


def load_wta_surface_real_data(
    years: list[int] | None = None,
    surfaces: list[str] | None = None,
    top_n_players: int = 16,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    if years is None:
        years = [2023, 2024]
    if surfaces is None:
        surfaces = ["Hard", "Clay", "Grass"]

    frames = [pd.read_csv(WTA_MATCH_URLS[year]) for year in years]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["surface"].isin(surfaces)].copy()
    decisive_mask = ~df["score"].fillna("").str.contains(
        "W/O|RET|DEF|Played and unfinished|Walkover",
        regex=True,
    )
    df = df[decisive_mask].copy()

    player_counts = pd.concat([df["winner_name"], df["loser_name"]]).value_counts()
    players = player_counts.head(top_n_players).index.tolist()
    df = df[df["winner_name"].isin(players) & df["loser_name"].isin(players)].copy()

    surfaces = [surface for surface in surfaces if surface in set(df["surface"])]
    player_to_idx = {name: idx for idx, name in enumerate(players)}
    surface_to_idx = {name: idx for idx, name in enumerate(surfaces)}

    df["group"] = df["surface"].map(surface_to_idx)
    df["item_i"] = df["winner_name"].map(player_to_idx)
    df["item_j"] = df["loser_name"].map(player_to_idx)
    df["y"] = 1
    return (
        df[["surface", "group", "item_i", "item_j", "y"]].reset_index(drop=True),
        players,
        surfaces,
    )


def _target_mixtures(categories: list[str], source_probs: np.ndarray) -> dict[str, np.ndarray]:
    n = len(categories)
    uniform = np.ones(n) / n
    coding_heavy = np.full(n, 0.05)
    writing_heavy = np.full(n, 0.05)

    idx = {name: i for i, name in enumerate(categories)}
    for key, value in {
        "coding": 0.36,
        "math": 0.18,
        "stem": 0.16,
        "reasoning": 0.12,
        "extraction": 0.08,
    }.items():
        if key in idx:
            coding_heavy[idx[key]] = value
    coding_heavy /= coding_heavy.sum()

    for key, value in {
        "writing": 0.30,
        "roleplay": 0.24,
        "humanities": 0.20,
        "reasoning": 0.10,
        "extraction": 0.08,
    }.items():
        if key in idx:
            writing_heavy[idx[key]] = value
    writing_heavy /= writing_heavy.sum()
    return {
        "source_mix": source_probs,
        "balanced": uniform,
        "coding_heavy": coding_heavy,
        "writing_heavy": writing_heavy,
    }


def _arena_language_mixtures(languages: list[str], source_probs: np.ndarray) -> dict[str, np.ndarray]:
    n = len(languages)
    idx = {name: i for i, name in enumerate(languages)}
    uniform = np.ones(n) / n
    english_heavy = np.full(n, 0.0)
    non_english_heavy = np.full(n, 0.0)

    if "en" in idx:
        english_heavy[idx["en"]] = 0.60
        residual = 0.40 / max(n - 1, 1)
        for lang in languages:
            if lang != "en":
                english_heavy[idx[lang]] = residual

        non_english_heavy[idx["en"]] = 0.10
        residual = 0.90 / max(n - 1, 1)
        for lang in languages:
            if lang != "en":
                non_english_heavy[idx[lang]] = residual
    else:
        english_heavy = uniform.copy()
        non_english_heavy = uniform.copy()

    return {
        "source_mix": source_probs,
        "balanced": uniform,
        "english_heavy": english_heavy,
        "non_english_heavy": non_english_heavy,
    }


def _atp_surface_mixtures(surfaces: list[str], source_probs: np.ndarray) -> dict[str, np.ndarray]:
    n = len(surfaces)
    idx = {name: i for i, name in enumerate(surfaces)}
    uniform = np.ones(n) / n

    def _heavy_mix(primary: str) -> np.ndarray:
        mix = np.full(n, 0.15)
        if primary in idx:
            mix[idx[primary]] = 0.70
        return mix / mix.sum()

    return {
        "source_mix": source_probs,
        "balanced": uniform,
        "hard_heavy": _heavy_mix("Hard"),
        "clay_heavy": _heavy_mix("Clay"),
        "grass_heavy": _heavy_mix("Grass"),
    }


def _summarize_real_rankings(rankings: pd.DataFrame, top_k: int) -> pd.DataFrame:
    summary_rows = []
    for mix_name, sub in rankings.groupby("mixture"):
        certified_members = sub[sub["certified_stable_topk_member"] == 1]["model"].tolist()
        exact_members = sub[sub["exact_stable_topk_member"] == 1]["model"].tolist()
        summary_rows.append(
            {
                "mixture": mix_name,
                "top1_model": sub.sort_values("rank").iloc[0]["model"],
                "top3_fixed": ", ".join(sub.sort_values("rank").head(top_k)["model"]),
                "certified_stable_topk_size": int(sub["certified_stable_topk_member"].sum()),
                "certified_stable_topk_models": ", ".join(certified_members),
                "exact_stable_topk_size": int(sub["exact_stable_topk_member"].sum()),
                "exact_stable_topk_models": ", ".join(exact_members),
            }
        )
    return pd.DataFrame(summary_rows)


def _topk_from_scores(scores: np.ndarray, top_k: int) -> set[int]:
    return set(np.argsort(scores)[::-1][:top_k].tolist())


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _real_method_comparison_against_reference(
    contextual_fit: Any,
    theta: np.ndarray,
    marginal_scores: np.ndarray,
    q_source: np.ndarray,
    reference_theta: np.ndarray,
    mixtures: dict[str, np.ndarray],
    top_k: int,
    l1_radius: float,
    lower_ratio: float,
    upper_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_items = theta.shape[1]
    source_scores = aggregate_scores(theta, q_source)
    stable_core = None
    stable_pairs: dict[tuple[int, int], int] = {}

    for mix_name, q in mixtures.items():
        scores = aggregate_scores(reference_theta, q)
        topk = _topk_from_scores(scores, top_k)
        stable_core = topk if stable_core is None else stable_core & topk
        for i in range(n_items):
            for j in range(i + 1, n_items):
                stable_pairs.setdefault((i, j), []).append(_sign(float(np.dot(q, reference_theta[:, i] - reference_theta[:, j]))))
    assert stable_core is not None
    stable_core = set(stable_core)
    stable_pair_sign = {
        pair: signs[0] if len(set(signs)) == 1 else 0 for pair, signs in stable_pairs.items()
    }

    metric_rows: list[dict[str, Any]] = []
    stable_core_rows = [{"item": int(i)} for i in sorted(stable_core)]
    n_stable_pairs = sum(1 for sign in stable_pair_sign.values() if sign != 0)

    for mix_name, q in mixtures.items():
        pair_lower, _ = robust_pairwise_bounds_matrix(theta, q, l1_radius, lower_ratio, upper_ratio)
        fixed_ci_lower = np.zeros((n_items, n_items))
        method_topk = {
            "marginal_bt": _topk_from_scores(marginal_scores, top_k),
            "source_contextual_bt": _topk_from_scores(source_scores, top_k),
            "fixed_target_contextual_bt": _topk_from_scores(aggregate_scores(theta, q), top_k),
        }
        for i in range(n_items):
            for j in range(i + 1, n_items):
                fixed_point = float(np.dot(q, theta[:, i] - theta[:, j]))
                fixed_se = contextual_contrast_se(contextual_fit, i, j, q)
                lo = fixed_point - 1.64 * fixed_se
                hi = fixed_point + 1.64 * fixed_se
                fixed_ci_lower[i, j] = lo
                fixed_ci_lower[j, i] = -hi
        method_topk["fixed_target_ci_certified_bt"] = set(certified_topk_set(fixed_ci_lower, top_k))
        method_topk["ss_cbt"] = set(
            exact_stable_topk_set(
                theta,
                q,
                l1_radius,
                lower_ratio,
                upper_ratio,
                top_k,
            )
        )

        pair_stats = {
            "marginal_bt": {"declared": 0, "correct": 0},
            "source_contextual_bt": {"declared": 0, "correct": 0},
            "fixed_target_contextual_bt": {"declared": 0, "correct": 0},
            "fixed_target_ci_certified_bt": {"declared": 0, "correct": 0},
            "ss_cbt": {"declared": 0, "correct": 0},
        }
        for i in range(n_items):
            for j in range(i + 1, n_items):
                reference_sign = stable_pair_sign[(i, j)]
                fixed_point = float(np.dot(q, theta[:, i] - theta[:, j]))
                fixed_se = contextual_contrast_se(contextual_fit, i, j, q)
                fixed_lo = fixed_point - 1.64 * fixed_se
                fixed_hi = fixed_point + 1.64 * fixed_se
                method_signs = {
                    "marginal_bt": (_sign(float(marginal_scores[i] - marginal_scores[j])), True),
                    "source_contextual_bt": (_sign(float(np.dot(q_source, theta[:, i] - theta[:, j]))), True),
                    "fixed_target_contextual_bt": (_sign(fixed_point), True),
                    "fixed_target_ci_certified_bt": (
                        1 if fixed_lo > 0 else (-1 if fixed_hi < 0 else 0),
                        bool((fixed_lo > 0) or (fixed_hi < 0)),
                    ),
                }
                lo, hi = robust_linear_bounds(theta[:, i] - theta[:, j], q, l1_radius, lower_ratio, upper_ratio)
                method_signs["ss_cbt"] = (
                    1 if lo > 0 else (-1 if hi < 0 else 0),
                    bool((lo > 0) or (hi < 0)),
                )
                for method, (pred_sign, declared) in method_signs.items():
                    if not declared:
                        continue
                    pair_stats[method]["declared"] += 1
                    if reference_sign != 0 and pred_sign == reference_sign:
                        pair_stats[method]["correct"] += 1

        for method, predicted in method_topk.items():
            overlap = len(predicted & stable_core)
            precision = overlap / len(predicted) if predicted else 0.0
            recall = overlap / len(stable_core) if stable_core else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            declared = pair_stats[method]["declared"]
            correct = pair_stats[method]["correct"]
            metric_rows.append(
                {
                    "mixture": mix_name,
                    "method": method,
                    "stable_pair_precision": correct / declared if declared else 0.0,
                    "stable_pair_recall": correct / n_stable_pairs if n_stable_pairs else 0.0,
                    "topk_precision": precision,
                    "topk_recall": recall,
                    "topk_f1": f1,
                    "set_size": float(len(predicted)),
                }
            )

    metric_df = pd.DataFrame(metric_rows)
    summary_df = metric_df.groupby("method", as_index=False).agg(
        stable_pair_precision=("stable_pair_precision", "mean"),
        stable_pair_recall=("stable_pair_recall", "mean"),
        topk_precision=("topk_precision", "mean"),
        topk_recall=("topk_recall", "mean"),
        topk_f1=("topk_f1", "mean"),
        avg_set_size=("set_size", "mean"),
    )
    return summary_df, metric_df, pd.DataFrame(stable_core_rows)


def _real_method_comparison(
    contextual_fit: Any,
    theta: np.ndarray,
    marginal_scores: np.ndarray,
    q_source: np.ndarray,
    mixtures: dict[str, np.ndarray],
    top_k: int,
    l1_radius: float,
    lower_ratio: float,
    upper_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return _real_method_comparison_against_reference(
        contextual_fit=contextual_fit,
        theta=theta,
        marginal_scores=marginal_scores,
        q_source=q_source,
        reference_theta=theta,
        mixtures=mixtures,
        top_k=top_k,
        l1_radius=l1_radius,
        lower_ratio=lower_ratio,
        upper_ratio=upper_ratio,
    )


def _stratified_half_split(data: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, sub in data.groupby("group", sort=True):
        idx = sub.index.to_numpy()
        perm = rng.permutation(idx)
        cut = len(perm) // 2
        cut = min(max(cut, 1), len(perm) - 1)
        train_parts.append(data.loc[perm[:cut]])
        test_parts.append(data.loc[perm[cut:]])
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, test_df


def run_real_holdout_validation(
    data: pd.DataFrame,
    groups: list[str],
    mixtures: dict[str, np.ndarray],
    penalty: float,
    l1_radius: float,
    lower_ratio: float,
    upper_ratio: float,
    top_k: int,
    *,
    n_splits: int = 8,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_items = int(max(data["item_i"].max(), data["item_j"].max()) + 1)
    split_rows: list[pd.DataFrame] = []
    for split in range(n_splits):
        train_df, test_df = _stratified_half_split(data, seed + split)
        train_source = train_df["group"].value_counts(normalize=True).sort_index()
        q_source_train = np.array([float(train_source.get(g, 0.0)) for g in range(len(groups))])

        train_marginal_fit = fit_marginal_bt(train_df[["item_i", "item_j", "y"]], n_items)
        train_contextual_fit = fit_contextual_bt(train_df, n_items, len(groups), penalty)
        test_contextual_fit = fit_contextual_bt(test_df, n_items, len(groups), penalty)

        summary_df, detail_df, _ = _real_method_comparison_against_reference(
            contextual_fit=train_contextual_fit,
            theta=train_contextual_fit.theta,
            marginal_scores=train_marginal_fit.theta[0],
            q_source=q_source_train,
            reference_theta=test_contextual_fit.theta,
            mixtures=mixtures,
            top_k=top_k,
            l1_radius=l1_radius,
            lower_ratio=lower_ratio,
            upper_ratio=upper_ratio,
        )
        summary_df["split"] = split
        detail_df["split"] = split
        split_rows.append(summary_df)

    split_summary = pd.concat(split_rows, ignore_index=True)
    overall_summary = split_summary.groupby("method", as_index=False).agg(
        stable_pair_precision=("stable_pair_precision", "mean"),
        stable_pair_recall=("stable_pair_recall", "mean"),
        topk_precision=("topk_precision", "mean"),
        topk_recall=("topk_recall", "mean"),
        topk_f1=("topk_f1", "mean"),
        avg_set_size=("avg_set_size", "mean"),
    )
    return overall_summary, split_summary


def run_matched_ci_comparison(
    data: pd.DataFrame,
    groups: list[str],
    mixtures: dict[str, np.ndarray],
    penalty: float,
    l1_radius: float,
    lower_ratio: float,
    upper_ratio: float,
    top_k: int,
    *,
    z_grid: np.ndarray | None = None,
) -> pd.DataFrame:
    if z_grid is None:
        z_grid = np.linspace(0.1, 2.5, 49)

    n_items = int(max(data["item_i"].max(), data["item_j"].max()) + 1)
    source_mix = data["group"].value_counts(normalize=True).sort_index()
    q_source = np.array([float(source_mix.get(g, 0.0)) for g in range(len(groups))])
    marginal_fit = fit_marginal_bt(data[["item_i", "item_j", "y"]], n_items)
    contextual_fit = fit_contextual_bt(data, n_items, len(groups), penalty)

    ss_summary, _, _ = _real_method_comparison_against_reference(
        contextual_fit=contextual_fit,
        theta=contextual_fit.theta,
        marginal_scores=marginal_fit.theta[0],
        q_source=q_source,
        reference_theta=contextual_fit.theta,
        mixtures=mixtures,
        top_k=top_k,
        l1_radius=l1_radius,
        lower_ratio=lower_ratio,
        upper_ratio=upper_ratio,
    )
    ss_row = ss_summary[ss_summary["method"] == "ss_cbt"].iloc[0]
    target_size = float(ss_row["avg_set_size"])

    stable_core = None
    stable_pairs: dict[tuple[int, int], list[int]] = {}
    for q in mixtures.values():
        topk = _topk_from_scores(aggregate_scores(contextual_fit.theta, q), top_k)
        stable_core = set(topk) if stable_core is None else stable_core & set(topk)
        for i in range(n_items):
            for j in range(i + 1, n_items):
                stable_pairs.setdefault((i, j), []).append(_sign(float(np.dot(q, contextual_fit.theta[:, i] - contextual_fit.theta[:, j]))))
    assert stable_core is not None
    stable_pair_sign = {pair: vals[0] if len(set(vals)) == 1 else 0 for pair, vals in stable_pairs.items()}

    best_summary: dict[str, float] | None = None
    best_objective: tuple[float, float] | None = None
    for z in z_grid:
        rows: list[dict[str, float]] = []
        for q in mixtures.values():
            fixed_ci_lower = np.zeros((n_items, n_items))
            declared = 0
            correct = 0
            for i in range(n_items):
                for j in range(i + 1, n_items):
                    fixed_point = float(np.dot(q, contextual_fit.theta[:, i] - contextual_fit.theta[:, j]))
                    fixed_se = contextual_contrast_se(contextual_fit, i, j, q)
                    lo = fixed_point - float(z) * fixed_se
                    hi = fixed_point + float(z) * fixed_se
                    fixed_ci_lower[i, j] = lo
                    fixed_ci_lower[j, i] = -hi
                    ref_sign = stable_pair_sign[(i, j)]
                    pred_sign = 1 if lo > 0.0 else (-1 if hi < 0.0 else 0)
                    if (lo > 0.0) or (hi < 0.0):
                        declared += 1
                        if ref_sign != 0 and pred_sign == ref_sign:
                            correct += 1

            predicted = set(certified_topk_set(fixed_ci_lower, top_k))
            overlap = len(predicted & stable_core)
            precision = overlap / len(predicted) if predicted else 0.0
            recall = overlap / len(stable_core) if stable_core else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            rows.append(
                {
                    "stable_pair_precision": correct / declared if declared else 0.0,
                    "topk_f1": f1,
                    "set_size": float(len(predicted)),
                }
            )
        matched_df = pd.DataFrame(rows)
        avg_size = float(matched_df["set_size"].mean())
        objective = (abs(avg_size - target_size), -float(matched_df["topk_f1"].mean()))
        if best_objective is None or objective < best_objective:
            best_objective = objective
            best_summary = {
                "matched_ci_z": float(z),
                "matched_ci_stable_pair_precision": float(matched_df["stable_pair_precision"].mean()),
                "matched_ci_topk_f1": float(matched_df["topk_f1"].mean()),
                "matched_ci_avg_set_size": avg_size,
            }

    assert best_summary is not None
    return pd.DataFrame(
        [
            {
                "ss_cbt_stable_pair_precision": float(ss_row["stable_pair_precision"]),
                "ss_cbt_topk_f1": float(ss_row["topk_f1"]),
                "ss_cbt_avg_set_size": float(ss_row["avg_set_size"]),
                **best_summary,
            }
        ]
    )


def _run_real_analysis(
    data: pd.DataFrame,
    models: list[str],
    groups: list[str],
    mixtures: dict[str, np.ndarray],
    output_path: Path,
    penalty: float,
    l1_radius: float,
    lower_ratio: float,
    upper_ratio: float,
    top_k: int,
    group_name: str,
) -> dict[str, Any]:
    output_path.mkdir(parents=True, exist_ok=True)
    n_items = len(models)
    n_groups = len(groups)

    marginal_fit = fit_marginal_bt(data[["item_i", "item_j", "y"]], n_items)
    fit = fit_contextual_bt(data, n_items, n_groups, penalty)

    ranking_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for mix_name, q in mixtures.items():
        scores = aggregate_scores(fit.theta, q)
        pair_lower, pair_upper = robust_pairwise_bounds_matrix(fit.theta, q, l1_radius, lower_ratio, upper_ratio)
        fixed_rank = np.argsort(scores)[::-1]
        certified_set = certified_topk_set(pair_lower, top_k)
        exact_set = exact_stable_topk_set(
            fit.theta,
            q,
            l1_radius,
            lower_ratio,
            upper_ratio,
            top_k,
        )
        for rank, item in enumerate(fixed_rank, start=1):
            item = int(item)
            ranking_rows.append(
                {
                    "mixture": mix_name,
                    "model": models[item],
                    "rank": rank,
                    "score": float(scores[item]),
                    "certified_wins": int(np.sum(pair_lower[item, :] > 0.0)),
                    "possible_outrankers": int(np.sum(pair_lower[:, item] >= 0.0) - 1),
                    "certified_stable_topk_member": int(item in certified_set),
                    "exact_stable_topk_member": int(item in exact_set),
                }
            )
        for i in range(n_items):
            for j in range(i + 1, n_items):
                lo = float(pair_lower[i, j])
                hi = float(pair_upper[i, j])
                pair_rows.append(
                    {
                        "mixture": mix_name,
                        "model_i": models[i],
                        "model_j": models[j],
                        "fixed_contrast": float(np.dot(q, fit.theta[:, i] - fit.theta[:, j])),
                        "robust_lower": lo,
                        "robust_upper": hi,
                        "stable_order": int((lo > 0.0) or (hi < 0.0)),
                    }
                )

    rankings = pd.DataFrame(ranking_rows)
    pairs = pd.DataFrame(pair_rows)
    summary = _summarize_real_rankings(rankings, top_k=top_k)
    source_mix = np.asarray(mixtures["source_mix"])
    comparison_summary, comparison_detail, stable_core = _real_method_comparison(
        contextual_fit=fit,
        theta=fit.theta,
        marginal_scores=marginal_fit.theta[0],
        q_source=source_mix,
        mixtures=mixtures,
        top_k=top_k,
        l1_radius=l1_radius,
        lower_ratio=lower_ratio,
        upper_ratio=upper_ratio,
    )

    rankings.to_csv(output_path / "real_rankings.csv", index=False)
    pairs.to_csv(output_path / "real_pairwise_bounds.csv", index=False)
    summary.to_csv(output_path / "real_summary.csv", index=False)
    comparison_summary.to_csv(output_path / "real_method_comparison_summary.csv", index=False)
    comparison_detail.to_csv(output_path / "real_method_comparison_detail.csv", index=False)
    stable_core.to_csv(output_path / "real_stable_core.csv", index=False)
    pd.DataFrame({group_name: groups, "source_prob": list(mixtures["source_mix"])}).to_csv(output_path / f"real_{group_name}s.csv", index=False)
    return {
        "n_rows": int(len(data)),
        "n_models": n_items,
        "n_groups": n_groups,
        "models": models,
        "groups": groups,
        "summary_path": str(output_path / "real_summary.csv"),
        "comparison_summary_path": str(output_path / "real_method_comparison_summary.csv"),
    }


def run_mt_bench_analysis(
    output_dir: str | Path,
    penalty: float,
    l1_radius: float,
    lower_ratio: float,
    upper_ratio: float,
    top_k: int,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    data, models, categories = load_mt_bench_real_data(cache_dir=cache_dir)
    output_path = Path(output_dir)
    source_mix = data["group"].value_counts(normalize=True).sort_index()
    source_probs = np.array([float(source_mix.get(g, 0.0)) for g in range(len(categories))])
    mixtures = _target_mixtures(categories, source_probs)
    outputs = _run_real_analysis(
        data=data,
        models=models,
        groups=categories,
        mixtures=mixtures,
        output_path=output_path,
        penalty=penalty,
        l1_radius=l1_radius,
        lower_ratio=lower_ratio,
        upper_ratio=upper_ratio,
        top_k=top_k,
        group_name="category",
    )
    outputs["n_categories"] = outputs.pop("n_groups")
    outputs["categories"] = outputs.pop("groups")
    return outputs


def run_arena_language_analysis(
    output_dir: str | Path,
    penalty: float,
    l1_radius: float,
    lower_ratio: float,
    upper_ratio: float,
    top_k: int,
    cache_dir: str | Path | None = None,
    languages: list[str] | None = None,
    top_n_models: int = 8,
) -> dict[str, Any]:
    data, models, languages = load_arena_language_real_data(
        cache_dir=cache_dir,
        languages=languages,
        top_n_models=top_n_models,
    )
    output_path = Path(output_dir)
    source_mix = data["group"].value_counts(normalize=True).sort_index()
    source_probs = np.array([float(source_mix.get(g, 0.0)) for g in range(len(languages))])
    mixtures = _arena_language_mixtures(languages, source_probs)
    outputs = _run_real_analysis(
        data=data,
        models=models,
        groups=languages,
        mixtures=mixtures,
        output_path=output_path,
        penalty=penalty,
        l1_radius=l1_radius,
        lower_ratio=lower_ratio,
        upper_ratio=upper_ratio,
        top_k=top_k,
        group_name="language",
    )
    outputs["n_languages"] = outputs.pop("n_groups")
    outputs["languages"] = outputs.pop("groups")
    return outputs


def run_atp_surface_analysis(
    output_dir: str | Path,
    penalty: float,
    l1_radius: float,
    lower_ratio: float,
    upper_ratio: float,
    top_k: int,
    years: list[int] | None = None,
    surfaces: list[str] | None = None,
    top_n_players: int = 20,
) -> dict[str, Any]:
    data, players, surfaces = load_atp_surface_real_data(
        years=years,
        surfaces=surfaces,
        top_n_players=top_n_players,
    )
    output_path = Path(output_dir)
    source_mix = data["group"].value_counts(normalize=True).sort_index()
    source_probs = np.array([float(source_mix.get(g, 0.0)) for g in range(len(surfaces))])
    mixtures = _atp_surface_mixtures(surfaces, source_probs)
    outputs = _run_real_analysis(
        data=data,
        models=players,
        groups=surfaces,
        mixtures=mixtures,
        output_path=output_path,
        penalty=penalty,
        l1_radius=l1_radius,
        lower_ratio=lower_ratio,
        upper_ratio=upper_ratio,
        top_k=top_k,
        group_name="surface",
    )
    outputs["n_surfaces"] = outputs.pop("n_groups")
    outputs["surfaces"] = outputs.pop("groups")
    return outputs


def run_wta_surface_analysis(
    output_dir: str | Path,
    penalty: float,
    l1_radius: float,
    lower_ratio: float,
    upper_ratio: float,
    top_k: int,
    years: list[int] | None = None,
    surfaces: list[str] | None = None,
    top_n_players: int = 16,
) -> dict[str, Any]:
    data, players, surfaces = load_wta_surface_real_data(
        years=years,
        surfaces=surfaces,
        top_n_players=top_n_players,
    )
    output_path = Path(output_dir)
    source_mix = data["group"].value_counts(normalize=True).sort_index()
    source_probs = np.array([float(source_mix.get(g, 0.0)) for g in range(len(surfaces))])
    mixtures = _atp_surface_mixtures(surfaces, source_probs)
    outputs = _run_real_analysis(
        data=data,
        models=players,
        groups=surfaces,
        mixtures=mixtures,
        output_path=output_path,
        penalty=penalty,
        l1_radius=l1_radius,
        lower_ratio=lower_ratio,
        upper_ratio=upper_ratio,
        top_k=top_k,
        group_name="surface",
    )
    outputs["n_surfaces"] = outputs.pop("n_groups")
    outputs["surfaces"] = outputs.pop("groups")
    return outputs
