# DR-CBT Experiment Traceability

## Goal

Produce the synthetic and real-data evidence used in the paper
_Robust Ranking under Target Distribution Uncertainty in Covariate-Assisted Bradley--Terry Models_
by comparing deployment-robust reporting against fixed-target and point-ranking baselines.

## Implemented synthetic scenarios

1. `exact_bt_mild_shift`
   Tests whether DR-CBT remains competitive when the contextual Bradley--Terry model is correctly specified and target mismatch is mild.
2. `misspecified_shift`
   Tests whether deployment-robust reporting helps when the target distribution is misspecified and pairwise probabilities include non-BT interaction effects.
3. `low_overlap_sparse`
   Tests the weak-overlap regime emphasized by the theory.

## Implemented methods

1. `marginal_bt`
   Ignores context and fits one Bradley--Terry model to all observations.
2. `source_contextual_bt`
   Fits the pooled contextual model and aggregates scores with the empirical source context mix.
3. `fixed_target_contextual_bt`
   Fits the pooled contextual model and aggregates with the nominal target distribution.
4. `fixed_target_ci_certified_bt`
   Starts from the fixed-target contextual fit and abstains using fixed-target confidence intervals.
5. `dr_cbt`
   Uses the same contextual fit, then computes deployment-robust pairwise bounds and stable summaries over the target uncertainty set.

## Bundled output artifacts

- `results/final_run/summary_pair_metrics.csv`: aggregated synthetic pairwise summaries
- `results/final_run/summary_topk_metrics.csv`: aggregated synthetic top-k summaries
- `results/final_run/comparison_table.csv`: compact synthetic comparison table
- `results/final_run/ablation_table.csv`: scenario-wise synthetic breakdown
- `results/final_run/paper_figures/`: synthetic and paper-facing figures
- `results/real_run/`: MT-Bench, Arena, ATP, and WTA real-data outputs
- `results/shift_stress_run/summary_shift_stress_metrics.csv`: deployment-shift stress-test summary
- `results/matched_ci_run/real_matched_ci_summary.csv`: matched-size comparison against the CI baseline
- `results/sensitivity_run/real_uncertainty_sensitivity_summary.csv`: uncertainty-radius sensitivity summary
- `results/calibration_run/target_sample_calibration_summary.csv`: target-sample calibration summary

## Result-to-paper mapping

- Synthetic interval-inclusion and false-stable findings come from `results/final_run/summary_pair_metrics.csv` and the figures in `results/final_run/paper_figures/`.
- Synthetic top-3 findings come from `results/final_run/summary_topk_metrics.csv`.
- Real-data benchmark summaries come from `results/real_run/`.
- The deployment-family stress-test figure comes from `results/shift_stress_run/shift_stress_suite.pdf`.
- Matched-size CI comparisons, uncertainty-radius sensitivity, and target-sample calibration results come from `results/matched_ci_run/`, `results/sensitivity_run/`, and `results/calibration_run/`, respectively.
