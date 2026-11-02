# How-To Run Guide: NSCLC Multi-Modal Fairness Benchmark

This repository implements a multi-modal machine learning benchmark for **2-Year Survival Prediction in Non-Small Cell Lung Cancer (NSCLC)**, with a focus on demographic and institutional fairness auditing and disparity mitigation.

---

## 1. Quick Start (End-to-End Execution)

To run a complete benchmark across all 3 biases (`affiliation`, `gender`, `ethnicity`), both training splits (`biased` + Fair Reweighing vs `BGMMOCE`), and automatically generate all performance metrics and fairness disparity plots:

```bash
# 1. Activate environment
conda activate nsclc_fairness

# 2. Run the complete pipeline (auto-names experiment: chkp/exp_[DD.MM]_[HHhMMm]/)
python analysisv3.py --bias all --training-type biased --exp-name exp_run && \
python analysisv3.py --bias all --training-type BGMMOCE --exp-name exp_run && \
python getResultsV2.py --base-dir chkp/exp_run && \
python plotClassFairness.py --base-dir chkp/exp_run
```

*(Note: When both `biased` and `BGMMOCE` runs finish, `analysisv3.py` will also automatically invoke `getResultsV2.py` and `plotClassFairness.py` if not called explicitly).*

---

## 2. Environment & Prerequisites

### Python Environment
Ensure you are using the dedicated `nsclc_fairness` conda environment:
```bash
conda activate nsclc_fairness
# Or use direct interpreter:
/home/trivizakis/anaconda3/envs/nsclc_fairness/bin/python
```

### Key Dependencies
- **Machine Learning**: `scikit-learn`, `numpy`, `scipy`, `pandas`
- **Fairness & Debiasing**: `aif360`
- **Explainability**: `shap`
- **Survival Analysis**: `lifelines`
- **Visualization**: `matplotlib`, `seaborn`

---

## 3. Benchmark Architecture

### A. Multi-Modal Early Fusion
The pipeline ingests and standardizes three modalities from `dataset/`:
1. **Clinical Data** (`clinical.csv`): Demographics, stage, histology, pack years.
2. **Radiomics Features** (`radiomics.csv`): High-dimensional CT image textures, shapes, and wavelets.
3. **RNA-seq Genomics** (`rnaseq.csv`): Gene expression profiles.

Features are scaled via `RobustScaler` (5th–95th percentile) and filtered via hierarchical selection (`SelectKBest` with ANOVA F-test and L1-penalized feature attribution).

### B. Sensitive Attributes (Biases Evaluated)
1. **Affiliation (`affiliation`)**: Institutional / hospital data source.
2. **Gender (`gender`)**: Biological sex disparity.
3. **Ethnicity (`ethnicity`)**: Racial / ethnic background.

### C. The 16 Classifier Families
The benchmark evaluates 16 classifier families supporting sample reweighting:
1. `random_forest` — Random Forest
2. `decision_tree` — Decision Tree
3. `adaboost` — AdaBoost
4. `gradient_boosting` — Gradient Boosting
5. `logistic_regression` — Logistic Regression (L2)
6. `ridge` — Ridge Classifier
7. `sgd` — Stochastic Gradient Descent (Log Loss)
8. `naive_bayes` — Gaussian Naive Bayes
9. `svm_rbf` — Support Vector Machine (RBF kernel)
10. `svm_linear` — Support Vector Machine (Linear kernel)
11. `svm_poly` — Support Vector Machine (Polynomial kernel)
12. `svm_sigmoid` — Support Vector Machine (Sigmoid kernel)
13. `mlp16` — Multi-Layer Perceptron (Hidden layer: 16)
14. `mlp32` — Multi-Layer Perceptron (Hidden layer: 32)
15. `mlp64` — Multi-Layer Perceptron (Hidden layer: 64)
16. `mlp128` — Multi-Layer Perceptron (Hidden layer: 128)

*(Note: KNN, Gaussian Process, and Voting Ensembles were excluded because they do not natively support sample weights for AIF360 Fair Reweighing).*

### D. Model Variants Evaluated per Family (The Triplet)
For each classifier family, three primary configurations are evaluated:
- **Biased (Red)**: Trained on the observational, unmitigated training distribution.
- **Fair / Reweighted (Green)**: Trained using AIF360 Sample Reweighing (`sample_weight` computed based on sensitive group frequencies).
- **BGMMOCE (Blue)**: Trained on the Balanced Gaussian Mixture Model Over-sampling with Clustering & Ensembling split.

---

## 4. Step-by-Step Execution Guide

### Step 1: Train Biased & Fair-Reweighted Models
```bash
python analysisv3.py --bias all --training-type biased --exp-name exp_my_run
```
- Fits models on `biased` training set.
- Generates AIF360 sample weights and fits `reweighing` models.
- Generates SHAP explainability plots, confusion matrices, and metrics JSON files.

### Step 2: Train BGMMOCE Mitigated Models
```bash
python analysisv3.py --bias all --training-type BGMMOCE --exp-name exp_my_run
```
- Fits models on the `BGMMOCE` balanced training split.
- Saves model artifacts, predictions, and test metrics in the matching experiment directory.

### Step 3: Performance Aggregation & Radar Plots
```bash
python getResultsV2.py --base-dir chkp/exp_my_run
```
- Computes ROC curves, AUROC, C-Index, Brier score, ECE, and Fair Statistical Parity differences.
- Outputs `benchmark_summary_global.csv` and radar charts comparing Biased vs Fair vs BGMMOCE.

### Step 4: Fairness & Disparity Visualizations
```bash
python plotClassFairness.py --base-dir chkp/exp_my_run --auc-threshold 0.70
```
This generates two distinct sets of figures:

#### 1. Standard Figures (All 16 Classifiers)
Saved in `chkp/exp_my_run/`:
- `fairness_gap_reduction_[bias].png`: Accuracy gap between privileged and unprivileged groups.
- `fairness_selection_rate_disparity_[bias].png`: Positive prediction rate across subgroups.
- `fairness_error_rate_by_subgroup_[bias].png`: Subgroup misclassification rates.

#### 2. Best-Performing Models (Filtered Triplets: $\text{AUC} > 0.70$)
Saved in `chkp/exp_my_run/best_performing_models/`:
- Filters triplets to only include families where $\max(\text{AUC}_{\text{Biased}}, \text{AUC}_{\text{Fair}}, \text{AUC}_{\text{BGMMOCE}}) > 0.70$.
- Uses clean titles (`Fairness Gap Reduction`, `Fairness Selection Rate`, `Fairness Error Rate`) without mathematical equations or cluttered subtitles.
- Saves summary table: `best_performing_models_summary.csv`.
- If no models in a bias reach $\text{AUC} > 0.70$ (e.g. ethnicity), skips plotting for that bias with an informative diagnostic message.

---

## 5. Output Directory Structure

Each experiment run generates the following directory layout:

```text
chkp/exp_[date]_[time]/
├── benchmark_summary_global.csv             # Complete benchmark summary across all 324 evaluations
├── benchmark_summary_affiliation.csv
├── benchmark_summary_gender.csv
├── benchmark_summary_ethnicity.csv
│
├── fairness_gap_reduction_affiliation.png   # Standard 16-classifier gap plot
├── fairness_gap_reduction_gender.png
├── fairness_gap_reduction_ethnicity.png
├── fairness_selection_rate_disparity_*.png  # Standard 16-classifier selection rate plots
├── fairness_error_rate_by_subgroup_*.png    # Standard 16-classifier error rate plots
│
├── best_performing_models/                  # Filtered models with AUC > 0.70
│   ├── best_performing_models_summary.csv   # Triplet AUCs and qualification flags
│   ├── fairness_gap_reduction_affiliation.png
│   ├── fairness_gap_reduction_gender.png
│   ├── fairness_selection_rate_disparity_affiliation.png
│   ├── fairness_selection_rate_disparity_gender.png
│   ├── fairness_error_rate_by_subgroup_affiliation.png
│   ├── fairness_error_rate_by_subgroup_gender.png
│   ├── affiliation/                         # Bias-specific subfolder copies
│   └── gender/
│
├── affiliation/                             # Per-bias detailed model folders
│   ├── random_forest/
│   │   ├── biased/
│   │   │   ├── model.pkl
│   │   │   ├── preds.npy
│   │   │   ├── probs.npy
│   │   │   ├── random_forest_metrics.json
│   │   │   ├── random_forest_cm.png
│   │   │   └── random_forest_shap_summary.png
│   │   ├── reweighing/                      # Fair reweighed variant
│   │   └── BGMMOCE/                         # BGMMOCE variant
│   ├── svm_poly/
│   └── ... (all 16 families)
├── gender/
└── ethnicity/
```

---

## 6. Command-Line Reference

### `analysisv3.py`
| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--bias` | `str` | `affiliation` | Sensitive attribute to evaluate (`affiliation`, `gender`, `ethnicity`, or `all`) |
| `--training-type` | `str` | `biased` | Training set split: `biased` or `BGMMOCE` |
| `--exp-name` | `str` | `None` | Experiment folder name (defaults to timestamp `exp_DD.MM_HHhMMm`) |
| `--output-dir` | `str` | `None` | Override the full output path |
| `--use-smote` | `flag` | `False` | Apply SMOTE resampling on the training set |

### `getResultsV2.py`
| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--base-dir` | `str` | `None` | Path to experiment folder (e.g. `chkp/exp_19.10_11h25m/`) |
| `--exp-name` | `str` | `None` | Name of experiment folder under `chkp/` |

### `plotClassFairness.py`
| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--base-dir` | `str` | `None` | Path to experiment folder (e.g. `chkp/exp_19.10_11h25m/`) |
| `--exp-name` | `str` | `None` | Name of experiment folder under `chkp/` |
| `--auc-threshold`| `float`| `0.70` | Minimum triplet AUC threshold for `best_performing_models` |

---

## 7. Troubleshooting & FAQs

### Q: Why do KNN, Gaussian Process, and Voting Ensembles not appear?
**A**: In scikit-learn, `KNeighborsClassifier`, `GaussianProcessClassifier`, and `VotingClassifier` do not accept `sample_weight` in their `.fit()` methods. They were removed to prevent missing Fair Reweighted bars and ensure every plotted classifier family has a complete, fair triplet (`Biased`, `Fair`, `BGMMOCE`).

### Q: Why are there no plots for Ethnicity in `best_performing_models/`?
**A**: If none of the 16 classifier families in a bias category achieve $\text{AUC} > 0.70$ on the test split (for example, ethnicity peak AUC was $0.667$), the script intentionally skips generating empty subplots for that bias and documents the status in `best_performing_models_summary.csv`.

### Q: What does the warning `No module named 'tensorflow'` mean?
**A**: This is an internal warning from AIF360 indicating that `AdversarialDebiasing` (which requires TensorFlow) is not installed. It does not affect our benchmark, as we utilize scikit-learn native mitigations (AIF360 Reweighing, Disparate Impact Remover, and BGMMOCE).
