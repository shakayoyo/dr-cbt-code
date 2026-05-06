from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import yaml

from src.real_data import run_arena_language_analysis, run_atp_surface_analysis, run_mt_bench_analysis


def l1_radius_from_target_sample(n_groups: int, target_sample_size: int, delta: float) -> float:
    if n_groups < 2:
        return 0.0
    return math.sqrt(2.0 * math.log((2**n_groups - 2) / delta) / target_sample_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run target-sample calibrated uncertainty analyses")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--target-sizes", nargs="+", type=int, default=[50, 100, 200, 500])
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    global_cfg = config["global"]
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    benchmark_specs = {
        "mt_bench": (8, run_mt_bench_analysis, True),
        "arena_language": (6, run_arena_language_analysis, True),
        "atp_surface": (3, run_atp_surface_analysis, False),
    }

    rows: list[dict[str, float | str | int]] = []
    for benchmark, (n_groups, fn, uses_cache) in benchmark_specs.items():
        for target_size in args.target_sizes:
            radius = l1_radius_from_target_sample(n_groups=n_groups, target_sample_size=target_size, delta=args.delta)
            bench_dir = output_root / benchmark / f"m_{target_size}"
            kwargs = {
                "output_dir": bench_dir,
                "penalty": float(global_cfg["penalty"]),
                "l1_radius": float(radius),
                "lower_ratio": float(global_cfg["uncertainty"]["lower_ratio"]),
                "upper_ratio": float(global_cfg["uncertainty"]["upper_ratio"]),
                "top_k": int(global_cfg["top_k"]),
            }
            if uses_cache:
                kwargs["cache_dir"] = args.cache_dir
            fn(**kwargs)
            summary = pd.read_csv(bench_dir / "real_method_comparison_summary.csv")
            summary["benchmark"] = benchmark
            summary["target_sample_size"] = target_size
            summary["delta"] = args.delta
            summary["implied_l1_radius"] = radius
            rows.extend(summary.to_dict(orient="records"))

    out = pd.DataFrame(rows)
    out.to_csv(output_root / "target_sample_calibration_summary.csv", index=False)


if __name__ == "__main__":
    main()
