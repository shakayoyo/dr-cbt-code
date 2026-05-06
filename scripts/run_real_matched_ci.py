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
    run_matched_ci_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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

    benchmark_rows: list[pd.DataFrame] = []

    mt_data, _, mt_categories = load_mt_bench_real_data(cache_dir=cache_dir)
    mt_source = mt_data["group"].value_counts(normalize=True).sort_index()
    mt_probs = np.array([float(mt_source.get(g, 0.0)) for g in range(len(mt_categories))])
    mt_df = run_matched_ci_comparison(
        data=mt_data,
        groups=mt_categories,
        mixtures=_target_mixtures(mt_categories, mt_probs),
        penalty=penalty,
        l1_radius=l1_radius,
        lower_ratio=lower_ratio,
        upper_ratio=upper_ratio,
        top_k=top_k,
    )
    mt_df["benchmark"] = "mt_bench"
    benchmark_rows.append(mt_df)

    arena_data, _, arena_languages = load_arena_language_real_data(cache_dir=cache_dir)
    arena_source = arena_data["group"].value_counts(normalize=True).sort_index()
    arena_probs = np.array([float(arena_source.get(g, 0.0)) for g in range(len(arena_languages))])
    arena_df = run_matched_ci_comparison(
        data=arena_data,
        groups=arena_languages,
        mixtures=_arena_language_mixtures(arena_languages, arena_probs),
        penalty=penalty,
        l1_radius=l1_radius,
        lower_ratio=lower_ratio,
        upper_ratio=upper_ratio,
        top_k=top_k,
    )
    arena_df["benchmark"] = "arena_language"
    benchmark_rows.append(arena_df)

    atp_data, _, atp_surfaces = load_atp_surface_real_data()
    atp_source = atp_data["group"].value_counts(normalize=True).sort_index()
    atp_probs = np.array([float(atp_source.get(g, 0.0)) for g in range(len(atp_surfaces))])
    atp_df = run_matched_ci_comparison(
        data=atp_data,
        groups=atp_surfaces,
        mixtures=_atp_surface_mixtures(atp_surfaces, atp_probs),
        penalty=penalty,
        l1_radius=l1_radius,
        lower_ratio=lower_ratio,
        upper_ratio=upper_ratio,
        top_k=top_k,
    )
    atp_df["benchmark"] = "atp_surface"
    benchmark_rows.append(atp_df)

    wta_data, _, wta_surfaces = load_wta_surface_real_data()
    wta_source = wta_data["group"].value_counts(normalize=True).sort_index()
    wta_probs = np.array([float(wta_source.get(g, 0.0)) for g in range(len(wta_surfaces))])
    wta_df = run_matched_ci_comparison(
        data=wta_data,
        groups=wta_surfaces,
        mixtures=_atp_surface_mixtures(wta_surfaces, wta_probs),
        penalty=penalty,
        l1_radius=l1_radius,
        lower_ratio=lower_ratio,
        upper_ratio=upper_ratio,
        top_k=top_k,
    )
    wta_df["benchmark"] = "wta_surface"
    benchmark_rows.append(wta_df)

    combined = pd.concat(benchmark_rows, ignore_index=True)
    combined.to_csv(output_dir / "real_matched_ci_summary.csv", index=False)


if __name__ == "__main__":
    main()
