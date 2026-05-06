#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python -m scripts.run_experiments --config config/default.yaml --output-dir results/final_run
python -m scripts.run_real_data --config config/default.yaml --output-dir results/real_run
python -m scripts.run_shift_stress_test --config config/default.yaml --output-dir results/shift_stress_run
python -m scripts.generate_tables_figures --input-dir results/final_run --output-dir results/final_run --real-dir results/real_run/mt_bench
python -m scripts.make_paper_figures --input-dir results/final_run --output-dir results/final_run/paper_figures
