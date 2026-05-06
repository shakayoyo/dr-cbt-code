from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src.real_data import run_arena_language_analysis, run_atp_surface_analysis, run_mt_bench_analysis, run_wta_surface_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real-data contextual ranking analyses")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    global_cfg = config["global"]
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    mt_outputs = run_mt_bench_analysis(
        output_dir=output_root / "mt_bench",
        penalty=float(global_cfg["penalty"]),
        l1_radius=float(global_cfg["uncertainty"]["l1_radius"]),
        lower_ratio=float(global_cfg["uncertainty"]["lower_ratio"]),
        upper_ratio=float(global_cfg["uncertainty"]["upper_ratio"]),
        top_k=int(global_cfg["top_k"]),
        cache_dir=args.cache_dir,
    )
    arena_outputs = run_arena_language_analysis(
        output_dir=output_root / "arena_language",
        penalty=float(global_cfg["penalty"]),
        l1_radius=float(global_cfg["uncertainty"]["l1_radius"]),
        lower_ratio=float(global_cfg["uncertainty"]["lower_ratio"]),
        upper_ratio=float(global_cfg["uncertainty"]["upper_ratio"]),
        top_k=int(global_cfg["top_k"]),
        cache_dir=args.cache_dir,
    )
    atp_outputs = run_atp_surface_analysis(
        output_dir=output_root / "atp_surface",
        penalty=float(global_cfg["penalty"]),
        l1_radius=float(global_cfg["uncertainty"]["l1_radius"]),
        lower_ratio=float(global_cfg["uncertainty"]["lower_ratio"]),
        upper_ratio=float(global_cfg["uncertainty"]["upper_ratio"]),
        top_k=int(global_cfg["top_k"]),
    )
    wta_outputs = run_wta_surface_analysis(
        output_dir=output_root / "wta_surface",
        penalty=float(global_cfg["penalty"]),
        l1_radius=float(global_cfg["uncertainty"]["l1_radius"]),
        lower_ratio=float(global_cfg["uncertainty"]["lower_ratio"]),
        upper_ratio=float(global_cfg["uncertainty"]["upper_ratio"]),
        top_k=int(global_cfg["top_k"]),
    )
    combined = pd.concat(
        [
            pd.read_csv(mt_outputs["comparison_summary_path"]).assign(benchmark="mt_bench"),
            pd.read_csv(arena_outputs["comparison_summary_path"]).assign(benchmark="arena_language"),
            pd.read_csv(atp_outputs["comparison_summary_path"]).assign(benchmark="atp_surface"),
            pd.read_csv(wta_outputs["comparison_summary_path"]).assign(benchmark="wta_surface"),
        ],
        ignore_index=True,
    )
    combined.to_csv(output_root / "real_method_comparison_summary.csv", index=False)
    print(json.dumps({"mt_bench": mt_outputs, "arena_language": arena_outputs, "atp_surface": atp_outputs, "wta_surface": wta_outputs}, indent=2))


if __name__ == "__main__":
    main()
