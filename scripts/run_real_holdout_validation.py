from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.real_data import (
    _arena_language_mixtures,
    _atp_surface_mixtures,
    _target_mixtures,
    load_arena_language_real_data,
    load_atp_surface_real_data,
    load_mt_bench_real_data,
    load_wta_surface_real_data,
    run_real_holdout_validation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    global_cfg = cfg["global"]
    penalty = float(global_cfg["penalty"])
    top_k = int(global_cfg["top_k"])
    l1_radius = float(global_cfg["uncertainty"]["l1_radius"])
    lower_ratio = float(global_cfg["uncertainty"]["lower_ratio"])
    upper_ratio = float(global_cfg["uncertainty"]["upper_ratio"])
    cache_dir = cfg.get("cache_dir")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_specs: list[tuple[str, pd.DataFrame, list[str], dict[str, float]]] = []

    mt_data, _, mt_categories = load_mt_bench_real_data(cache_dir=cache_dir)
    mt_source = mt_data["group"].value_counts(normalize=True).sort_index()
    mt_probs = np.array([float(mt_source.get(g, 0.0)) for g in range(len(mt_categories))])
    benchmark_specs.append(
        ("mt_bench", mt_data, mt_categories, _target_mixtures(mt_categories, mt_probs))
    )

    arena_data, _, arena_languages = load_arena_language_real_data(cache_dir=cache_dir)
    arena_source = arena_data["group"].value_counts(normalize=True).sort_index()
    arena_probs = np.array([float(arena_source.get(g, 0.0)) for g in range(len(arena_languages))])
    benchmark_specs.append(
        ("arena_language", arena_data, arena_languages, _arena_language_mixtures(arena_languages, arena_probs))
    )

    atp_data, _, atp_surfaces = load_atp_surface_real_data()
    atp_source = atp_data["group"].value_counts(normalize=True).sort_index()
    atp_probs = np.array([float(atp_source.get(g, 0.0)) for g in range(len(atp_surfaces))])
    benchmark_specs.append(
        ("atp_surface", atp_data, atp_surfaces, _atp_surface_mixtures(atp_surfaces, atp_probs))
    )

    wta_data, _, wta_surfaces = load_wta_surface_real_data()
    wta_source = wta_data["group"].value_counts(normalize=True).sort_index()
    wta_probs = np.array([float(wta_source.get(g, 0.0)) for g in range(len(wta_surfaces))])
    benchmark_specs.append(
        ("wta_surface", wta_data, wta_surfaces, _atp_surface_mixtures(wta_surfaces, wta_probs))
    )

    combined_rows: list[pd.DataFrame] = []
    for benchmark, data, groups, mixtures in benchmark_specs:
        benchmark_dir = output_dir / benchmark
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        summary_df, split_df = run_real_holdout_validation(
            data=data,
            groups=groups,
            mixtures=mixtures,
            penalty=penalty,
            l1_radius=l1_radius,
            lower_ratio=lower_ratio,
            upper_ratio=upper_ratio,
            top_k=top_k,
            n_splits=args.splits,
            seed=args.seed,
        )
        summary_df["benchmark"] = benchmark
        split_df["benchmark"] = benchmark
        summary_df.to_csv(benchmark_dir / "holdout_summary.csv", index=False)
        split_df.to_csv(benchmark_dir / "holdout_split_metrics.csv", index=False)
        combined_rows.append(summary_df)

    combined = pd.concat(combined_rows, ignore_index=True)
    combined.to_csv(output_dir / "holdout_real_method_comparison_summary.csv", index=False)


if __name__ == "__main__":
    main()
