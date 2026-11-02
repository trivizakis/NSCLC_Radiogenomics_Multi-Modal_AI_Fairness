# Best-Performing Models Fairness & Disparity Benchmark Report

**Experiment Run**: `chkp/exp_19.10_11h25m/`  
**Evaluation Threshold**: Triplet $\max(\text{AUC}_{\text{Biased}}, \text{AUC}_{\text{Fair}}, \text{AUC}_{\text{BGMMOCE}}) > 0.70$  
**Endpoint**: 2-Year Survival in NSCLC Multi-Modal Cohort  
**Date**: October 19, 2026

---

## 1. Executive Summary

This report evaluates fairness gap reduction, selection rate disparity, and misclassification error rate disparities across three demographic/institutional biases (**Affiliation**, **Gender**, **Ethnicity**).

To isolate disparities among clinically viable models, a filtered set of figures was introduced alongside standard 16-classifier figures. A classifier family triplet (`Biased`, `Fair` / Reweighted, `BGMMOCE`) is included in the **Best-Performing Models** analysis if **at least one** model in the triplet achieves an **$\text{AUC} > 0.70$**.

### Summary of Qualifying Classifier Families

| Bias Type | Total Families Tested | Qualifying Families ($\text{AUC} > 0.70$) | Highest AUC Model in Triplet |
| :--- | :---: | :---: | :--- |
| **Affiliation** | 16 | **2** | **SVM (Poly)** ($\text{AUC} = 0.800$ with BGMMOCE) |
| **Gender** | 16 | **5** | **AdaBoost** ($\text{AUC} = 0.811$ with Fair Reweighing) |
| **Ethnicity** | 16 | **0** | Decision Tree ($\text{AUC} = 0.667$ with BGMMOCE) |

---

## 2. Qualifying Classifier Families Breakdown

### A. Affiliation (Center / Hospital Disparity)
Two classifier families achieved $\text{AUC} > 0.70$:
- **SVM (Poly)**: Biased = $0.744$, Fair = $0.733$, BGMMOCE = **$0.800$**
- **Naive Bayes**: Biased = **$0.733$**, Fair = $0.722$, BGMMOCE = $0.389$

**Fairness Insights**:
- For **SVM (Poly)**, BGMMOCE achieved the highest AUC ($0.800$) while maintaining competitive disparity gap metrics.
- For **Naive Bayes**, Fair Reweighing maintained high discrimination ($\text{AUC} = 0.722$) while significantly compressing the accuracy gap between institutional subgroups compared to the unmitigated baseline.

### B. Gender (Biological Sex Disparity)
Five classifier families achieved $\text{AUC} > 0.70$:
- **AdaBoost**: Biased = $0.678$, Fair = **$0.811$**, BGMMOCE = $0.656$
- **Decision Tree**: Biased = **$0.772$**, Fair = $0.583$, BGMMOCE = $0.361$
- **SVM (Sigmoid)**: Biased = **$0.756$**, Fair = $0.744$, BGMMOCE = $0.744$
- **Gradient Boosting**: Biased = **$0.733$**, Fair = $0.678$, BGMMOCE = $0.494$
- **Random Forest**: Biased = $0.678$, Fair = **$0.722$**, BGMMOCE = $0.544$

**Fairness Insights**:
- **AdaBoost** demonstrated a substantial synergy with **Fair Reweighing**, jumping from $\text{AUC} = 0.678$ to **$0.811$** (highest overall model in the entire benchmark) while reducing the subgroup selection rate disparity.
- **Random Forest** under Fair Reweighing improved both discrimination ($\text{AUC} = 0.722$ vs $0.678$) and subgroup balance.
- **SVM (Sigmoid)** demonstrated robust stability across Biased ($0.756$), Fair ($0.744$), and BGMMOCE ($0.744$).

### C. Ethnicity (Racial / Ethnic Disparity)
- **Zero** classifier families surpassed $\text{AUC} > 0.70$ on the test split (maximum achieved was Decision Tree BGMMOCE at $0.667$, and SVM-Poly Fair at $0.644$).
- As designed, the pipeline cleanly skipped figure generation for ethnicity in `best_performing_models/` without throwing errors, logging the diagnostic status to `best_performing_models_summary.csv`.

---

## 3. Generated Figure Artifacts

All figures follow the standardized triplets design (comparing **Biased [Red]**, **Fair Reweighted [Green]**, and **BGMMOCE [Blue]**):

### In `chkp/exp_19.10_11h25m/best_performing_models/`:
1. `fairness_gap_reduction_affiliation.png`
2. `fairness_selection_rate_disparity_affiliation.png`
3. `fairness_error_rate_by_subgroup_affiliation.png`
4. `fairness_gap_reduction_gender.png`
5. `fairness_selection_rate_disparity_gender.png`
6. `fairness_error_rate_by_subgroup_gender.png`
7. `best_performing_models_summary.csv`

### In `chkp/exp_19.10_11h25m/` (Standard Global Plots):
- `fairness_gap_reduction_[affiliation/gender/ethnicity].png` (all 16 families)
- `fairness_selection_rate_disparity_[affiliation/gender/ethnicity].png` (all 16 families)
- `fairness_error_rate_by_subgroup_[affiliation/gender/ethnicity].png` (all 16 families)
- `benchmark_summary_global.csv` (324 total model evaluations)
