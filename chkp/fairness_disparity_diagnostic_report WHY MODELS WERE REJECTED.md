# Diagnostic Report: Missing Fair Reweighing Bars in `chkp/exp_18.10_23h45m`

**Date:** 2026-10-19  
**Target Directory:** `chkp/exp_18.10_23h45m/`  
**Author:** Eleftherios Trivizakis & Google DeepMind Antigravity  

---

## 1. Executive Summary

In all fairness disparity plots generated inside `chkp/exp_18.10_23h45m/`:
- `fairness_error_rate_by_subgroup_[bias].png`
- `fairness_selection_rate_disparity_[bias].png`
- `fairness_gap_reduction_[bias].png`

The **Green bar** represents **"Fair"** mitigation (primarily AIF360 sample **Reweighing**, as defined in `COLOR_PALETTE["Fair"] = "#2ca02c"`).

Across the 15 evaluated classifier families, the green bar does **not appear** in:
1. **KNN** (`k_neighbors`)
2. **Voting Ensemble** (`voting_ensemble`)
3. **Gaussian Process** (`gaussian_process`)
4. **Decision Tree** (`decision_tree`)
5. **Naive Bayes** (`naive_bayes`)
6. **Ridge Classifier** (`ridge`)
7. **SGD Classifier** (`sgd`)

This report identifies what those sub-plots are and details the two distinct root causes for their absence.

---

## 2. Identification of the "4 Subgroup Sub-Plots"

In the multi-panel subgroup disparity figures (`fairness_error_rate_by_subgroup_*.png` and `fairness_selection_rate_disparity_*.png`), each sub-plot corresponds to one classifier family evaluated across sensitive demographic subgroups.

Besides the 3 explicitly named classifiers (**KNN**, **Voting Ensemble**, and **Gaussian Process**), the **other 4 sub-plots** lacking the green bar are:
1. **Decision Tree** (`decision_tree`)
2. **Naive Bayes** (`naive_bayes`)
3. **Ridge Classifier** (`ridge`)
4. **SGD Classifier** (`sgd`)

Together, these form a group of 7 classifiers (out of 15 total) that had no green bar plotted at the time `exp_18.10_23h45m` was generated.

---

## 3. Root Cause Analysis

### Cause A: Algorithmic Incompatibility in Scikit-Learn (KNN, Gaussian Process, Voting Ensemble)
AIF360 Pre-Processing Reweighing works by computing non-negative sample weights $W$ for training instances to equalize selection probabilities between privileged and unprivileged groups:
$$\text{clf.fit}(X_{tr}, y_{train}, \text{sample\_weight}=W)$$

However, within `scikit-learn`:
1. **`KNeighborsClassifier`**: Instance-based nearest-neighbor search does not accept sample weights during `.fit()`. Calling `.fit(X, y, sample_weight=...)` raises:
   `TypeError: fit() got an unexpected keyword argument 'sample_weight'`
2. **`GaussianProcessClassifier`**: The standard Gaussian Process classification implementation (Laplace approximation for binary classification) does not support sample weights in its `.fit()` method.
3. **`VotingClassifier`**: Does not natively accept sample weights uniformly across heterogeneous base estimators during `.fit()`.

Consequently, in `analysisv3.py` (`get_all_classifiers()`), no `fair_reweighted_knn`, `fair_reweighted_gp`, or `fair_reweighted_voting` models were ever instantiated.

---

### Cause B: Execution Timeline Discrepancy (Decision Tree, Naive Bayes, Ridge, SGD)
While Decision Tree, Naive Bayes, Ridge, and SGD **do** support sample weights in Scikit-Learn:
1. When the experiment `exp_18.10_23h45m` was initiated and the plots were generated at **23:55–23:56 on Oct 18**, only the original **8 core classifiers** had completed reweighing:
   - Random Forest, Logistic Regression, SVM, Gradient Boosting, AdaBoost, MLP 16, MLP 32, and MLP 64.
2. The remaining models (`fair_reweighted_dt`, `fair_reweighted_nb`, `fair_reweighted_ridge`, `fair_reweighted_sgd`) were added and evaluated **after midnight** (between 00:16 and 00:20 on Oct 19) for Affiliation and Gender, but were never executed for Ethnicity.
3. Because `plotClassFairness.py` was executed at 23:56, before those models or predictions existed on disk, the script found no `Fair` variant records and thus rendered only the **Red** (`Biased`) and **Blue** (`BGMMOCE`) bars.

---

## 4. Comprehensive Classifier Status Table (`exp_18.10_23h45m`)

| Classifier Family | Scikit-Learn `sample_weight` Support | Reweighed Model Defined? | Present in `exp_18.10_23h45m` Plots? | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Random Forest** | Yes | `fair_reweighted_rf` | **Yes (Green Bar)** | Standard core model |
| **Logistic Regression** | Yes | `fair_reweighted_lr` | **Yes (Green Bar)** | Standard core model |
| **SVM (RBF)** | Yes | `fair_reweighted_svm` | **Yes (Green Bar)** | Standard core model |
| **Gradient Boosting** | Yes | `fair_reweighted_boost` | **Yes (Green Bar)** | Standard core model |
| **AdaBoost** | Yes | `fair_reweighted_ada` | **Yes (Green Bar)** | Standard core model |
| **MLP (16)** | Yes | `fair_reweighted_mlp16` | **Yes (Green Bar)** | Standard core model |
| **MLP (32)** | Yes | `fair_reweighted_mlp32` | **Yes (Green Bar)** | Standard core model |
| **MLP (64)** | Yes | `fair_reweighted_mlp64` | **Yes (Green Bar)** | Standard core model |
| **KNN** | **No** | None | **Missing** | Unsupported in scikit-learn |
| **Gaussian Process** | **No** | None | **Missing** | Unsupported in scikit-learn |
| **Voting Ensemble** | **No** | None | **Missing** | Unsupported in scikit-learn |
| **Decision Tree** | Yes | `fair_reweighted_dt` | **Missing** | Trained post-plot generation |
| **Naive Bayes** | Yes | `fair_reweighted_nb` | **Missing** | Trained post-plot generation |
| **Ridge** | Yes | `fair_reweighted_ridge` | **Missing** | Trained post-plot generation |
| **SGD** | Yes | `fair_reweighted_sgd` | **Missing** | Trained post-plot generation |

---

## 5. Architectural Improvements

To resolve these discrepancies across future runs:
1. **Remove Unsupported Classifiers**: Exclude KNN, Gaussian Process, and Voting Ensemble so all retained models cleanly support the complete fairness mitigation suite (Pre-processing Reweighing, In-processing, and Post-processing).
2. **Split SVM Kernels**: Treat `SVM-RBF`, `SVM-Linear`, `SVM-Poly`, and `SVM-Sigmoid` as distinct, dedicated classifier families with their own baseline, reweighted, and mitigated pipelines.
3. **Add MLP (128)**: Expand the single-layer neural network capacity suite (`mlp16`, `mlp32`, `mlp64`, `mlp128`).
4. **Synchronized Pipeline Execution**: Automate complete multi-bias execution (`biased` and `BGMMOCE`), ensuring all models finish before invoking evaluation and visualization scripts.
