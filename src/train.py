from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.loaders import generate_world, load_experiment_config, sample_observations
from src.evaluate import evaluate_replicate
from src.utils.logging import make_logger, write_json


def _summarize_pairs(pair_df: pd.DataFrame) -> pd.DataFrame:
    grouped = pair_df.groupby(["scenario", "method"], as_index=False).agg(
        coverage=("coverage", "mean"),
        declaration_rate=("declared", "mean"),
        correct_rate=("correct_declaration", "mean"),
        false_stable_count=("false_stable", "sum"),
        declaration_count=("declared", "sum"),
    )
    grouped["false_stable_rate"] = grouped["false_stable_count"] / grouped["declaration_count"].clip(lower=1.0)
    grouped["stable_recall"] = grouped["correct_rate"]
    return grouped.drop(columns=["false_stable_count", "declaration_count"])


def _summarize_topk(topk_df: pd.DataFrame) -> pd.DataFrame:
    return topk_df.groupby(["scenario", "method"], as_index=False).agg(
        topk_precision=("topk_precision", "mean"),
        topk_recall=("topk_recall", "mean"),
        avg_set_size=("set_size", "mean"),
        avg_shortfall=("shortfall", "mean"),
    )


def run_experiment_suite(config_path: str | Path, output_dir: str | Path, resume: bool = False) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger = make_logger(output_path / "run.log")
    global_cfg, scenarios = load_experiment_config(config_path)
    write_json(output_path / "run_manifest.json", {"config_path": str(config_path), "resume": bool(resume)})

    pair_path = output_path / "raw_pair_metrics.csv"
    topk_path = output_path / "raw_topk_metrics.csv"
    if resume and pair_path.exists() and topk_path.exists():
        logger.info("Resume requested and raw outputs already exist. Reusing prior results.")
        pair_df = pd.read_csv(pair_path)
        topk_df = pd.read_csv(topk_path)
    else:
        pair_frames = []
        topk_frames = []
        for scenario_idx, scenario in enumerate(scenarios):
            for replicate in range(global_cfg.n_replicates):
                seed = global_cfg.seed + 1000 * scenario_idx + replicate
                rng = __import__("numpy").random.default_rng(seed)
                world = generate_world(global_cfg, scenario, rng)
                data = sample_observations(global_cfg, scenario, world, rng)
                pair_df_rep, topk_df_rep = evaluate_replicate(global_cfg, scenario, world, data, replicate)
                pair_frames.append(pair_df_rep)
                topk_frames.append(topk_df_rep)
                logger.info("Completed scenario=%s replicate=%d", scenario.name, replicate)
        pair_df = pd.concat(pair_frames, ignore_index=True)
        topk_df = pd.concat(topk_frames, ignore_index=True)
        pair_df.to_csv(pair_path, index=False)
        topk_df.to_csv(topk_path, index=False)

    pair_summary = _summarize_pairs(pair_df)
    topk_summary = _summarize_topk(topk_df)
    pair_summary.to_csv(output_path / "summary_pair_metrics.csv", index=False)
    topk_summary.to_csv(output_path / "summary_topk_metrics.csv", index=False)
    return {
        "raw_pair_metrics": str(pair_path),
        "raw_topk_metrics": str(topk_path),
        "summary_pair_metrics": str(output_path / "summary_pair_metrics.csv"),
        "summary_topk_metrics": str(output_path / "summary_topk_metrics.csv"),
    }
