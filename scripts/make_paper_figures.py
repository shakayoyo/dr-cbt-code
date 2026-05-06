from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect paper-facing figures")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for stem in (
        "coverage_by_scenario",
        "false_order_rate",
        "topk_recall",
        "main_synthetic_figure",
        "real_method_comparison",
        "real_uncertainty_focus",
        "method_overview",
    ):
        for ext in (".png", ".pdf"):
            src = input_dir / f"{stem}{ext}"
            if src.exists():
                shutil.copy2(src, output_dir / src.name)


if __name__ == "__main__":
    main()
