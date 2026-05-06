from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.real_data import run_arena_language_analysis, run_atp_surface_analysis, run_mt_bench_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Run uncertainty-radius sensitivity on real benchmarks")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--radii", nargs="+", type=float, default=[0.10, 0.18, 0.26, 0.34])
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    global_cfg = config["global"]
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    benchmark_fns = {
        "mt_bench": run_mt_bench_analysis,
        "arena_language": run_arena_language_analysis,
        "atp_surface": run_atp_surface_analysis,
    }

    rows: list[dict[str, float | str]] = []
    for radius in args.radii:
        radius_dir = output_root / f"l1_{radius:.2f}"
        radius_dir.mkdir(parents=True, exist_ok=True)
        for benchmark, fn in benchmark_fns.items():
            kwargs = {
                "output_dir": radius_dir / benchmark,
                "penalty": float(global_cfg["penalty"]),
                "l1_radius": float(radius),
                "lower_ratio": float(global_cfg["uncertainty"]["lower_ratio"]),
                "upper_ratio": float(global_cfg["uncertainty"]["upper_ratio"]),
                "top_k": int(global_cfg["top_k"]),
            }
            if benchmark != "atp_surface":
                kwargs["cache_dir"] = args.cache_dir
            fn(**kwargs)
            summary = pd.read_csv(radius_dir / benchmark / "real_method_comparison_summary.csv")
            summary["benchmark"] = benchmark
            summary["l1_radius"] = radius
            rows.extend(summary.to_dict(orient="records"))

    out = pd.DataFrame(rows)
    out.to_csv(output_root / "real_uncertainty_sensitivity_summary.csv", index=False)


if __name__ == "__main__":
    main()
