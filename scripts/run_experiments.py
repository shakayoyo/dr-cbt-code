from __future__ import annotations

import argparse
import json

from src.train import run_experiment_suite


def main() -> None:
parser = argparse.ArgumentParser(description="Run DR-CBT synthetic experiments")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    outputs = run_experiment_suite(args.config, args.output_dir, resume=args.resume)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
