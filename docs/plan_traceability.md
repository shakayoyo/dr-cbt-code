# SS-CBT Experiment Traceability

## Goal

Produce real synthetic evidence for the SS-CBT paper draft by testing whether robust ranking summaries are more reliable than point rankings under contextual shift.

## Implemented experiment matrix

1. `exact_bt_mild_shift`
   Tests that SS-CBT is competitive when the contextual BT model is correctly specified and the target mismatch is mild.
2. `misspecified_shift`
   Tests whether robustness helps when the target distribution is misspecified and pairwise probabilities include non-BT interaction effects.
3. `low_overlap_sparse`
   Tests the failure regime highlighted in the theory: weak overlap plus sparse within-group comparison graphs.

## Implemented methods

1. `marginal_bt`
   Ignores context and fits one Bradley-Terry model to all observations.
2. `source_contextual_bt`
   Fits the pooled contextual model and aggregates scores with the empirical source context mix.
3. `fixed_target_contextual_bt`
   Fits the pooled contextual model and aggregates with the nominal target distribution.
4. `ss_cbt`
   Uses the same reweighted pooled contextual estimator, then computes robust pairwise contrast bounds and pairwise-certified top-k sets over the target uncertainty set.

## Output artifacts

- `raw_pair_metrics.csv`: per-replicate, per-method pairwise metrics
- `raw_topk_metrics.csv`: per-replicate, per-method top-k metrics
- `summary_pair_metrics.csv`: aggregated pairwise summary table
- `summary_topk_metrics.csv`: aggregated top-k summary table
- `comparison_table.csv`: compact paper-facing comparison table
- `ablation_table.csv`: scenario-wise ablation table
- `coverage_by_scenario.png/.pdf`
- `false_order_rate.png/.pdf`
- `topk_recall.png/.pdf`
- `real_run/real_summary.csv`: MT-Bench real-data ranking summary across deployment mixtures
- `real_run/real_pairwise_bounds.csv`: robust pairwise bounds for the real-data illustration

## Result-to-paper mapping

- H1 maps to pairwise coverage in `summary_pair_metrics.csv`.
- H2 maps to `topk_precision`, `topk_recall`, and `avg_set_size` in `summary_topk_metrics.csv`.
- H3 maps to `declaration_rate`, `false_stable_rate`, and `avg_set_size` across scenarios.
- H4 maps to the contextual methods outperforming `marginal_bt` under rare-group settings.
