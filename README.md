# NSCLC Radiogenomics Multi-Modal AI Fairness Benchmark

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Fairness: AIF360](https://img.shields.io/badge/Fairness-AIF360-brightgreen.svg)](https://github.com/Trusted-AI/AIF360)

Comprehensive machine learning benchmark evaluating demographic and institutional fairness in **2-Year Survival Prediction for Non-Small Cell Lung Cancer (NSCLC)** using multi-modal early fusion (Clinical data, Radiomics textures, and RNA-seq gene expression).

---

## 📌 Overview

Multi-modal predictive models in precision oncology risk entrenching or amplifying disparities across protected patient subgroups. This repository establishes a rigorous auditing and debiasing benchmark across:
- **16 Classifier Families**: Tree-based, ensembles, linear, SVM kernels, naive Bayes, and neural networks (MLP).
- **3 Sensitive Demographic & Institutional Attributes**:
  1. `affiliation`: Hospital / institutional data acquisition site.
  2. `gender`: Biological sex disparity.
  3. `ethnicity`: Racial / ethnic group representation.
- **Fairness Mitigation Strategies**:
  - **Biased Baseline (Red)**: Observational, unmitigated training distribution.
  - **Fair Reweighing (Green)**: AIF360 sample reweighting based on subgroup marginals.
  - **BGMMOCE (Blue)**: Balanced Gaussian Mixture Model Over-sampling with Clustering & Ensembling.
  - Pre-processing (Disparate Impact Remover), Post-processing (Reject Option Classification, Equalized Odds).

---

## 📂 Repository Structure

```text
├── analysisv3.py                               # Core multi-modal benchmark training & SHAP explainability
├── getResultsV2.py                             # Result aggregator, ROC curves, AUROC & disparity metrics
├── plotClassFairness.py                        # Fairness gap reduction & subgroup error rate visualizations
├── HOWTO_RUN_GUIDE.md                          # Comprehensive step-by-step execution & troubleshooting guide
├── chkp/
│   ├── fairness_disparity_diagnostic_report WHY MODELS WERE REJECTED.md  # Detailed rejection diagnostics
│   └── exp_19.10_11h25m/                       # Benchmark experiment outputs (metrics & summaries)
│       ├── benchmark_summary_global.csv        # Global performance across all evaluated models
│       ├── benchmark_summary_affiliation.csv   # Affiliation-specific summary metrics
│       ├── benchmark_summary_gender.csv        # Gender-specific summary metrics
│       ├── benchmark_summary_ethnicity.csv     # Ethnicity-specific summary metrics
│       ├── best_performing_models/             # Filtered clinically viable models (AUC > 0.70)
│       │   ├── best_performing_models_summary.csv
│       │   └── report.md                       # High-level evaluation report
│       ├── affiliation/                        # Detailed classifier JSONs and feature importances
│       ├── gender/
│       └── ethnicity/
├── LICENSE                                     # MIT License
└── README.md
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
conda create -n nsclc_fairness python=3.10 -y
conda activate nsclc_fairness
pip install scikit-learn numpy scipy pandas aif360 shap lifelines matplotlib seaborn
```

### 2. End-to-End Pipeline Execution

```bash
# 1. Run Biased Baseline & AIF360 Fair Reweighing
python analysisv3.py --bias all --training-type biased --exp-name exp_run

# 2. Run BGMMOCE Balanced Training
python analysisv3.py --bias all --training-type BGMMOCE --exp-name exp_run

# 3. Aggregate Performance Metrics & Radar Charts
python getResultsV2.py --base-dir chkp/exp_run

# 4. Generate Fairness Disparity Plots & Best-Performing Models (AUC > 0.70)
python plotClassFairness.py --base-dir chkp/exp_run --auc-threshold 0.70
```

For complete command-line arguments and deep architectural details, consult the [HOWTO_RUN_GUIDE.md](HOWTO_RUN_GUIDE.md).

---

## 📊 Key Findings

- **Best Overall Discriminative Model**: **AdaBoost** under Fair Reweighing achieved the highest discrimination ($\text{AUC} = 0.811$) while narrowing selection rate disparity across gender.
- **Institutional Disparity (Affiliation)**: **SVM (Polynomial kernel)** achieved $\text{AUC} = 0.800$ with BGMMOCE mitigation.
- **Algorithmic Compatibility**: AIF360 Reweighing requires sample weight support during `.fit()`; classifiers without native weight support (e.g. KNN, Gaussian Process) were systematically cataloged in the [Rejection Diagnostic Report](chkp/fairness_disparity_diagnostic_report%20WHY%20MODELS%20WERE%20REJECTED.md).

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).
