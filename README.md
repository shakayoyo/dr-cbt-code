# DR-CBT Experiment Code

This repository contains the experimental code and bundled outputs for the paper
_Robust Ranking under Target Distribution Uncertainty in Covariate-Assisted Bradley--Terry Models_.

The method in the paper is referred to as **Deployment-Robust Contextual Bradley--Terry (DR-CBT)**.
Some internal filenames and package metadata still use the earlier `sscbt` shorthand; those names refer to the same codebase.

## Repository layout

- `src/`: contextual Bradley--Terry models, training code, evaluation code, and real-data pipelines
- `scripts/`: entry-point scripts for synthetic experiments, real-data studies, stress tests, and figure generation
- `config/`: experiment configurations
- `results/`: machine-readable outputs and paper-ready figures from the main runs
- `docs/`: lightweight traceability notes
- `run.sh`: convenience script for the main synthetic, real-data, and stress-test runs

## Environment

The code was developed for Python `>=3.10`.

Install the dependencies with:

```bash
pip install -r requirements.txt
```

or

```bash
pip install -e .
```

## Main runs

The simplest entry point is:

```bash
bash run.sh
```

This launches:

1. the main synthetic benchmark,
2. the real-data benchmark suite,
3. the deployment-shift stress test, and
4. paper figure generation.

Equivalent Python entry points include:

```bash
python -m scripts.run_experiments --config config/default.yaml --output-dir results/final_run
python -m scripts.run_real_data --config config/default.yaml --output-dir results/real_run
python -m scripts.run_shift_stress_test --config config/default.yaml --output-dir results/shift_stress_run
python -m scripts.generate_tables_figures --input-dir results/final_run --output-dir results/final_run --real-dir results/real_run/mt_bench
```

## Included outputs

This repository includes the final result folders used in the paper:

- `results/final_run/`: synthetic benchmark summaries and paper figures
- `results/real_run/`: MT-Bench, Arena, ATP, and WTA real-data benchmark outputs
- `results/shift_stress_run/`: deployment-shift stress-test outputs
- `results/matched_ci_run/`: matched-size comparison against the fixed-target CI baseline
- `results/sensitivity_run/`: uncertainty-radius sensitivity outputs
- `results/calibration_run/`: target-sample calibration outputs

Intermediate smoke tests, refresh runs, and earlier radius-specific variants are intentionally omitted from this repository snapshot.

## Notes on the real-data benchmarks

The real-data pipeline includes the benchmark suites used in the paper, including:

- MT-Bench category mixtures,
- Arena language mixtures,
- ATP surface mixtures, and
- WTA surface mixtures.

The evaluation code reports both certified stable top-`k` summaries and, on the small benchmark instances considered in the paper, exact stable top-`k` summaries.
