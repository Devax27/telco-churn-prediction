# Final evaluation protocol (written before the test set is read)

- **Model:** logistic regression, C = 1.0 (chosen in Step 7 by the rule in `src/train_compare.py`), trained on all 5,634 training customers.
- **Threshold:** the theoretical threshold for the base-case cost scenario, offer_cost / (success_rate x value_saved) = 20 / (0.3 x 400) = 0.1667. Chosen in Step 8 using training data only. All cost figures are illustrative assumptions for a fictional dataset.
- **Test set:** 1,409 customers (about 374 churners), read once by `src/final_evaluation.py`. The script refuses to overwrite `docs/final_test_results.csv` unless run with `--rerun`; any rerun will be disclosed in the README.
- **Reported:** PR-AUC, ROC-AUC, Brier score, and at the threshold: share contacted, precision, recall and net benefit per 1,000 customers, against contacting everyone. Uncertainty: 95% percentile intervals from 2,000 bootstrap resamples of the test set.
- **No retuning:** whatever the results are, the model, features and threshold stay as they are. Disappointing results get reported, not fixed.
- **Flags decided in advance:**
  1. If the 95% interval for test PR-AUC excludes the cross-validation estimate (0.6611), investigate for split or leakage errors. The model is not retuned.
  2. If the lower end of the interval for "net benefit minus contact-everyone" is not above zero, the README says the advantage over blanket contact is not statistically clear on this test set.