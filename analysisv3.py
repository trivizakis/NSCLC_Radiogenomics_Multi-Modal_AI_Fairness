#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSCLC Fairness Benchmark: Lung Cancer 2-Year Survival Multi-Modal Classification

Multi-modal early-fusion analysis combining Clinical data, Radiomics, and RNA-seq.
Integrates:
- Comprehensive scikit-learn models (linear, SVM, tree, ensemble, probabilistic, neural network)
- Expanded AIF360 bias mitigation methods (pre-processing, in-processing, post-processing)
- Modality-aware hierarchical feature selection with early-fusion
- SHAP explainability for test inference with modality contribution attribution
- Fairness metrics logging (Statistical Parity, Disparate Impact, Equal Opportunity, Average Odds)

@author: trivizakis & Google DeepMind Antigravity
"""

import os
import sys
import time
import glob
import json
import warnings
import argparse
from datetime import datetime
import pickle as pkl
import numpy as np
import pandas as pd
import scipy.stats as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Selection & Scalers
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel, VarianceThreshold

# Models (Comprehensive Scikit-Learn Ensemble)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Metrics & Resampling
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    brier_score_loss,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE
from lifelines.utils import concordance_index


# AIF360 Fairness & Mitigation
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
from aif360.algorithms.inprocessing import (
    PrejudiceRemover,
    ExponentiatedGradientReduction,
    GridSearchReduction
)
from aif360.algorithms.postprocessing import (
    RejectOptionClassification,
    EqOddsPostprocessing,
    CalibratedEqOddsPostprocessing
)

# SHAP Explainability
import shap

warnings.filterwarnings('ignore')


def shorten_feature_name(name):
    """
    Shortens feature names for SHAP and figures:
    Removes 'rad_' prefix and '1_original_' or 'original_1_original_'.
    """
    s = str(name)
    if s.startswith("rad_"):
        s = s[4:]
    s = s.replace("original_1_original_", "")
    s = s.replace("1_original_", "")
    return s


def get_classifier_family_and_variant(name, training_type="biased"):
    """
    Maps a model name to its classifier family and comparable variant subfolder:
    Returns (classifier_family, variant_name)
    Hierarchy: <bias> / <classifier_family> / <variant>
    """
    is_bgmmoce = (str(training_type).upper() == "BGMMOCE")

    base_family_map = {
        "random_forest": "random_forest",
        "logistic_l2": "logistic_regression",
        "logistic_l1": "logistic_regression",
        "logistic_elastic": "logistic_regression",
        "svm_rbf": "svm_rbf",
        "svm_linear": "svm_linear",
        "svm_poly": "svm_poly",
        "svm_sigmoid": "svm_sigmoid",
        "gradient_boosting": "gradient_boosting",
        "adaboost": "adaboost",
        "mlp16": "mlp16",
        "mlp32": "mlp32",
        "mlp64": "mlp64",
        "mlp128": "mlp128",
        "decision_tree": "decision_tree",
        "naive_bayes": "naive_bayes",
        "ridge": "ridge",
        "sgd": "sgd"
    }

    if not name.startswith("fair_"):
        family = base_family_map.get(name, name)
        variant = "BGMMOCE" if is_bgmmoce else "biased"
        return family, variant

    prefix = "BGMMOCE_" if is_bgmmoce else ""

    if "reweighted" in name:
        fam_key = name.replace("fair_reweighted_", "")
        fam_map = {
            "rf": "random_forest",
            "lr": "logistic_regression",
            "svm_rbf": "svm_rbf",
            "svm_linear": "svm_linear",
            "svm_poly": "svm_poly",
            "svm_sigmoid": "svm_sigmoid",
            "svm": "svm_rbf",
            "boost": "gradient_boosting",
            "ada": "adaboost",
            "dt": "decision_tree",
            "nb": "naive_bayes",
            "ridge": "ridge",
            "sgd": "sgd",
            "mlp16": "mlp16",
            "mlp32": "mlp32",
            "mlp64": "mlp64",
            "mlp128": "mlp128"
        }
        return fam_map.get(fam_key, fam_key), f"{prefix}reweighing"

    if "dir" in name:
        fam_key = name.replace("fair_dir_", "")
        fam_map = {
            "rf": "random_forest",
            "lr": "logistic_regression",
            "svm_rbf": "svm_rbf",
            "svm_linear": "svm_linear",
            "svm_poly": "svm_poly",
            "svm_sigmoid": "svm_sigmoid",
            "svm": "svm_rbf",
            "boost": "gradient_boosting",
            "ada": "adaboost",
            "dt": "decision_tree",
            "nb": "naive_bayes",
            "ridge": "ridge",
            "sgd": "sgd",
            "mlp16": "mlp16",
            "mlp32": "mlp32",
            "mlp64": "mlp64",
            "mlp128": "mlp128"
        }
        return fam_map.get(fam_key, fam_key), f"{prefix}disparate_impact_remover"

    if "prejudice" in name:
        return "logistic_regression", f"{prefix}inprocessing"

    if "exp_grad" in name or "grid_search" in name:
        if "rf" in name:
            return "random_forest", f"{prefix}reductions"
        elif "dt" in name:
            return "decision_tree", f"{prefix}reductions"
        else:
            return "logistic_regression", f"{prefix}reductions"

    if "roc" in name or "eq_odds" in name or "calib" in name:
        if "rf" in name:
            return "random_forest", f"{prefix}postprocessing"
        elif "mlp16" in name:
            return "mlp16", f"{prefix}postprocessing"
        elif "mlp32" in name:
            return "mlp32", f"{prefix}postprocessing"
        elif "mlp64" in name:
            return "mlp64", f"{prefix}postprocessing"
        elif "mlp128" in name:
            return "mlp128", f"{prefix}postprocessing"
        elif "svm_linear" in name:
            return "svm_linear", f"{prefix}postprocessing"
        elif "svm_poly" in name:
            return "svm_poly", f"{prefix}postprocessing"
        elif "svm_sigmoid" in name:
            return "svm_sigmoid", f"{prefix}postprocessing"
        elif "svm_rbf" in name or "svm" in name:
            return "svm_rbf", f"{prefix}postprocessing"
        elif "boost" in name:
            return "gradient_boosting", f"{prefix}postprocessing"
        elif "ada" in name:
            return "adaboost", f"{prefix}postprocessing"
        elif "dt" in name:
            return "decision_tree", f"{prefix}postprocessing"
        elif "nb" in name:
            return "naive_bayes", f"{prefix}postprocessing"
        elif "sgd" in name:
            return "sgd", f"{prefix}postprocessing"
        else:
            return "logistic_regression", f"{prefix}postprocessing"

    return "other", f"{prefix}mitigated"


# ==============================================================================
# 1. Multi-Modal Data Loader
# ==============================================================================
class MultimodalDataLoader:
    """
    Loads and aligns Clinical, Radiomics, and RNA-seq data for a given bias attribute.
    
    Training paths: dataset/[biased or BGMMOCE]/[bias type]/[clinical, radiomics, rnaseq]
    Testing paths:  dataset/testing_set/[bias type]/[clinical, radiomics, rnaseq]
    """
    def __init__(self, base_dir="dataset"):
        self.base_dir = base_dir

    def _find_csv(self, folder, modality_keyword):
        """Locates the CSV file corresponding to a modality keyword."""
        pattern = os.path.join(folder, f"*{modality_keyword}*.csv")
        files = glob.glob(pattern)
        if not files:
            # Fallback: inspect all csvs in folder
            for f in os.listdir(folder):
                if f.endswith('.csv') and modality_keyword.lower() in f.lower():
                    return os.path.join(folder, f)
            raise FileNotFoundError(f"Could not find CSV containing '{modality_keyword}' in {folder}")
        return sorted(files)[0]

    def load_split(self, split_folder, bias_type):
        """Loads and aligns the 3 CSVs in a single split folder."""
        clin_file = self._find_csv(split_folder, "clinical")
        rad_file  = self._find_csv(split_folder, "radiomics")
        rna_file  = self._find_csv(split_folder, "rnaseq")

        print(f"  [Loader] Clinical:  {os.path.basename(clin_file)}")
        print(f"  [Loader] Radiomics: {os.path.basename(rad_file)}")
        print(f"  [Loader] RNA-seq:   {os.path.basename(rna_file)}")

        df_clin = pd.read_csv(clin_file)
        df_rad  = pd.read_csv(rad_file)
        df_rna  = pd.read_csv(rna_file)

        # Detect target (2-Year Survival: column before the last one)
        target_col = df_clin.columns[-2]
        # Detect sensitive attribute (last column in CSVs)
        bias_col = df_clin.columns[-1]

        # Alignment: Check Patient_ID presence
        if 'Patient_ID' in df_clin.columns and 'Patient_ID' in df_rad.columns and 'Patient_ID' in df_rna.columns:
            # Sort or inner merge by Patient_ID to align rows
            df_clin = df_clin.sort_values('Patient_ID').reset_index(drop=True)
            df_rad  = df_rad.sort_values('Patient_ID').reset_index(drop=True)
            df_rna  = df_rna.sort_values('Patient_ID').reset_index(drop=True)

            common_ids = sorted(list(set(df_clin['Patient_ID']) & set(df_rad['Patient_ID']) & set(df_rna['Patient_ID'])))
            df_clin = df_clin[df_clin['Patient_ID'].isin(common_ids)].sort_values('Patient_ID').reset_index(drop=True)
            df_rad  = df_rad[df_rad['Patient_ID'].isin(common_ids)].sort_values('Patient_ID').reset_index(drop=True)
            df_rna  = df_rna[df_rna['Patient_ID'].isin(common_ids)].sort_values('Patient_ID').reset_index(drop=True)
            patient_ids = df_clin['Patient_ID'].tolist()
        else:
            # If no Patient_ID (e.g., BGMMOCE), align by common minimum length
            min_len = min(len(df_clin), len(df_rad), len(df_rna))
            df_clin = df_clin.iloc[:min_len].reset_index(drop=True)
            df_rad  = df_rad.iloc[:min_len].reset_index(drop=True)
            df_rna  = df_rna.iloc[:min_len].reset_index(drop=True)
            patient_ids = [f"sample_{i}" for i in range(min_len)]

        y = df_clin[target_col].astype(int).values
        s = df_clin[bias_col].astype(int).values

        # Strip non-feature columns
        drop_cols_clin = [c for c in ['Patient_ID', target_col, bias_col] if c in df_clin.columns]
        drop_cols_rad  = [c for c in ['Patient_ID', target_col, bias_col] if c in df_rad.columns]
        drop_cols_rna  = [c for c in ['Patient_ID', target_col, bias_col] if c in df_rna.columns]

        X_clin = df_clin.drop(columns=drop_cols_clin).apply(pd.to_numeric, errors='coerce').fillna(0.0)
        X_rad  = df_rad.drop(columns=drop_cols_rad).apply(pd.to_numeric, errors='coerce').fillna(0.0)
        X_rna  = df_rna.drop(columns=drop_cols_rna).apply(pd.to_numeric, errors='coerce').fillna(0.0)

        # Prefix columns to avoid collisions
        X_clin.columns = [f"clin_{c}" if not c.startswith("clin_") else c for c in X_clin.columns]
        X_rad.columns  = [f"rad_{c}" if not c.startswith("rad_") else c for c in X_rad.columns]
        X_rna.columns  = [f"rna_{c}" if not c.startswith("rna_") else c for c in X_rna.columns]

        return {
            "clinical": X_clin,
            "radiomics": X_rad,
            "rnaseq": X_rna,
            "y": y,
            "s": s,
            "patient_ids": patient_ids,
            "target_col": target_col,
            "bias_col": bias_col
        }

    def load_dataset(self, bias_type="affiliation", training_set_type="biased"):
        """Loads both training and testing splits."""
        train_dir = os.path.join(self.base_dir, training_set_type, bias_type)
        test_dir  = os.path.join(self.base_dir, "testing_set", bias_type)

        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Testing directory not found: {test_dir}")

        print(f"\n[Loader] Loading Training Set from: {train_dir}")
        train_data = self.load_split(train_dir, bias_type)
        print(f"[Loader] Loading Testing Set from:  {test_dir}")
        test_data  = self.load_split(test_dir, bias_type)

        return train_data, test_data


# ==============================================================================
# 2. Multi-Modal Early-Fusion & Feature Selection
# ==============================================================================
class MultimodalEarlyFusionSelector:
    """
    Performs modality-specific preprocessing:
    - Clinical data: Scaled directly; NO feature selection; passed directly to early fusion.
    - Radiomics data: Scaled; VarianceThreshold -> ANOVA F-test -> L1 Logistic Regression.
    - RNA-seq data: Scaled; VarianceThreshold -> ANOVA F-test -> L1 Logistic Regression.
    - Early Fusion: Concatenates clinical, selected radiomics, and selected rnaseq into X_fused.
    """
    def __init__(self, hypes):
        self.hypes = hypes
        self.selected_radiomics = []
        self.selected_rnaseq = []
        self.clinical_features = []
        self.all_fused_features = []
        self.feature_modality_map = {}

    def _select_features(self, X_tr, y_tr, modality_name):
        """Hierarchical selection: Variance -> ANOVA -> L1 Sparsity with safe fallback."""
        # 1. Variance Filter
        remover = VarianceThreshold(threshold=self.hypes.get("variance_threshold", 0.0))
        X_var = remover.fit_transform(X_tr)
        cols_after_var = X_tr.columns[remover.get_support(indices=True)]

        if len(cols_after_var) == 0:
            print(f"    [Warning] VarianceThreshold removed all features for {modality_name}, keeping original.")
            X_var = X_tr.values
            cols_after_var = X_tr.columns

        # 2. ANOVA Filter
        if self.hypes.get("use_custom_anova", True):
            f_scores, p_values = f_classif(X_var, y_tr)
            p_thresh = self.hypes.get("p_value", 0.05)
            crit_val = st.f.ppf(
                q=1 - p_thresh,
                dfn=len(np.unique(y_tr)) - 1,
                dfd=len(y_tr) - len(np.unique(y_tr))
            )
            selected_idx = [i for i, p in enumerate(p_values) if p < p_thresh and f_scores[i] >= crit_val]
            
            # Safe fallback if custom ANOVA selects too few features
            min_needed = self.hypes.get("min_features_per_modality", 5)
            if len(selected_idx) < min_needed:
                k_fallback = min(self.hypes.get("k_best", 20), X_var.shape[1])
                selector = SelectKBest(f_classif, k=k_fallback)
                selector.fit(X_var, y_tr)
                selected_idx = selector.get_support(indices=True)
        else:
            k = min(self.hypes.get("k_best", 20), X_var.shape[1])
            selector = SelectKBest(f_classif, k=k)
            selector.fit(X_var, y_tr)
            selected_idx = selector.get_support(indices=True)

        X_anova = X_var[:, selected_idx]
        cols_after_anova = cols_after_var[selected_idx]

        # 3. L1 Sparsity Control
        l1_sel = SelectFromModel(
            LogisticRegression(
                C=self.hypes.get("l1_c_value", 0.5),
                penalty='l1',
                solver='liblinear',
                max_iter=5000,
                random_state=42
            )
        )
        l1_sel.fit(X_anova, y_tr)
        l1_support = l1_sel.get_support()
        final_cols = cols_after_anova[l1_support].tolist()

        # Fallback if L1 eliminates everything
        if len(final_cols) == 0:
            print(f"    [Warning] L1 selected 0 features for {modality_name}, falling back to top ANOVA features.")
            final_cols = cols_after_anova[:min(10, len(cols_after_anova))].tolist()

        return final_cols

    def fit_transform(self, train_data, test_data):
        """Fits scalers and selectors on train, transforms train & test, and early-fuses."""
        p_one = self.hypes.get("p_one", 5.0)
        p_two = self.hypes.get("p_two", 95.0)

        # 1. Clinical Data (Scale only, NO feature selection)
        clin_scaler = RobustScaler(quantile_range=(p_one, p_two))
        X_tr_clin = pd.DataFrame(
            clin_scaler.fit_transform(train_data["clinical"]),
            columns=train_data["clinical"].columns
        )
        X_ts_clin = pd.DataFrame(
            clin_scaler.transform(test_data["clinical"]),
            columns=test_data["clinical"].columns
        )
        self.clinical_features = X_tr_clin.columns.tolist()

        # 2. Radiomics Data (Scale then Select)
        rad_scaler = RobustScaler(quantile_range=(p_one, p_two))
        X_tr_rad_sc = pd.DataFrame(
            rad_scaler.fit_transform(train_data["radiomics"]),
            columns=train_data["radiomics"].columns
        )
        X_ts_rad_sc = pd.DataFrame(
            rad_scaler.transform(test_data["radiomics"]),
            columns=test_data["radiomics"].columns
        )
        self.selected_radiomics = self._select_features(X_tr_rad_sc, train_data["y"], "Radiomics")
        X_tr_rad = X_tr_rad_sc[self.selected_radiomics]
        X_ts_rad = X_ts_rad_sc[self.selected_radiomics]

        # 3. RNA-seq Data (Scale then Select)
        rna_scaler = RobustScaler(quantile_range=(p_one, p_two))
        X_tr_rna_sc = pd.DataFrame(
            rna_scaler.fit_transform(train_data["rnaseq"]),
            columns=train_data["rnaseq"].columns
        )
        X_ts_rna_sc = pd.DataFrame(
            rna_scaler.transform(test_data["rnaseq"]),
            columns=test_data["rnaseq"].columns
        )
        self.selected_rnaseq = self._select_features(X_tr_rna_sc, train_data["y"], "RNA-seq")
        X_tr_rna = X_tr_rna_sc[self.selected_rnaseq]
        X_ts_rna = X_ts_rna_sc[self.selected_rnaseq]

        # 4. Early-Fusion: Concatenation
        X_tr_fused = pd.concat([X_tr_clin, X_tr_rad, X_tr_rna], axis=1)
        X_ts_fused = pd.concat([X_ts_clin, X_ts_rad, X_ts_rna], axis=1)

        self.all_fused_features = X_tr_fused.columns.tolist()

        # Build feature -> modality map for SHAP
        self.feature_modality_map = {}
        for c in self.clinical_features:
            self.feature_modality_map[c] = "Clinical"
        for c in self.selected_radiomics:
            self.feature_modality_map[c] = "Radiomics"
        for c in self.selected_rnaseq:
            self.feature_modality_map[c] = "RNA-seq"

        print("\n[Early Fusion Summary]")
        print(f"  Clinical features (no selection): {len(self.clinical_features)}")
        print(f"  Radiomics features (selected):    {len(self.selected_radiomics)}")
        print(f"  RNA-seq features (selected):      {len(self.selected_rnaseq)}")
        print(f"  Total early-fused features:       {X_tr_fused.shape[1]}")

        return X_tr_fused, X_ts_fused

    def save_selected_features_txt(self, output_dir, bias_type, training_type):
        """Saves formatted text and plain text lists of best features prior to classification."""
        os.makedirs(output_dir, exist_ok=True)
        txt_path = os.path.join(output_dir, "selected_features_prior_to_classification_"+bias_type+".txt")
        # = os.path.join(output_dir, "selected_features_list.txt")

        with open(txt_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("MULTI-MODAL EARLY-FUSION: BEST FEATURES PRIOR TO CLASSIFICATION\n")
            f.write(f"Experiment Bias:   {bias_type.upper()}\n")
            f.write(f"Training Split:    {training_type.upper()}\n")
            f.write(f"Total Features:    {len(self.all_fused_features)}\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"1. CLINICAL FEATURES ({len(self.clinical_features)} features - concatenated directly without selection):\n")
            f.write("-" * 80 + "\n")
            for i, col in enumerate(self.clinical_features, 1):
                f.write(f"  {i:3d}. {col}\n")
            f.write("\n")

            f.write(f"2. BEST SELECTED RADIOMICS FEATURES ({len(self.selected_radiomics)} features selected via ANOVA + L1):\n")
            f.write("-" * 80 + "\n")
            for i, col in enumerate(self.selected_radiomics, 1):
                short_name = shorten_feature_name(col)
                f.write(f"  {i:3d}. {col:<60s}  -->  [Display: {short_name}]\n")
            f.write("\n")

            f.write(f"3. BEST SELECTED RNA-SEQ FEATURES ({len(self.selected_rnaseq)} features selected via ANOVA + L1):\n")
            f.write("-" * 80 + "\n")
            for i, col in enumerate(self.selected_rnaseq, 1):
                f.write(f"  {i:3d}. {col}\n")
            f.write("\n")

            f.write("=" * 80 + "\n")
            f.write(f"ORDERED FUSED FEATURE VECTOR (Total: {len(self.all_fused_features)} features):\n")
            f.write("=" * 80 + "\n")
            for i, col in enumerate(self.all_fused_features, 1):
                mod = self.feature_modality_map.get(col, "Unknown")
                short_name = shorten_feature_name(col)
                f.write(f"  {i:3d}. [{mod:<9s}] {col:<60s} (Display: {short_name})\n")

        # with open(plain_path, "w") as f:
        #     for col in self.all_fused_features:
        #         f.write(f"{col}\n")

        print(f"\n[Features] Saved best features prior to classification in text form:")
        print(f"  -> Detailed Report: {txt_path}")
        # print(f"  -> Plain List:       {plain_path}")


# ==============================================================================
# 3. Comprehensive Benchmarker (Sklearn + AIF360 + SHAP)
# ==============================================================================
class ComprehensiveBenchmarker:
    """
    Evaluates standard scikit-learn models and extensive AIF360 bias mitigation methods.
    Provides inference metrics, fairness metrics, confusion matrices, and SHAP explainability.
    """
    def __init__(self, hypes):
        self.hypes = hypes
        self.results_data = {}

    def pandas_to_aif360(self, X, y, sensitive_attr_values, sensitive_attr_name):
        """Converts feature DataFrame, label, and sensitive attribute to AIF360 BinaryLabelDataset."""
        df = X.copy()
        df['target'] = np.array(y).flatten().astype(float)
        df[sensitive_attr_name] = np.array(sensitive_attr_values).flatten().astype(float)
        df = df.astype(float)
        return BinaryLabelDataset(
            df=df,
            label_names=['target'],
            protected_attribute_names=[sensitive_attr_name],
            favorable_label=1.0,
            unfavorable_label=0.0
        )

    def get_aif360_weights(self, X, y, s_values, s_attr_name):
        """Calculates instance weights for Reweighing with NaN guard."""
        ds = self.pandas_to_aif360(X, y, s_values, s_attr_name)
        rw = Reweighing(
            unprivileged_groups=[{s_attr_name: 0.0}],
            privileged_groups=[{s_attr_name: 1.0}]
        )
        ds_tr = rw.fit_transform(ds)
        weights = ds_tr.instance_weights
        if np.isnan(weights).any():
            print(f"  [Warning] NaNs detected in instance weights for {s_attr_name}. Neutralizing NaNs.")
            weights = np.nan_to_num(weights, nan=1.0)
        return weights

    def compute_fairness_metrics(self, y_true, y_pred, s_values, s_attr_name):
        """Computes comprehensive fairness metrics via AIF360 ClassificationMetric."""
        df_true = pd.DataFrame({'target': y_true.astype(float), s_attr_name: s_values.astype(float)})
        df_pred = pd.DataFrame({'target': y_pred.astype(float), s_attr_name: s_values.astype(float)})
        bld_true = BinaryLabelDataset(df=df_true, label_names=['target'], protected_attribute_names=[s_attr_name])
        bld_pred = BinaryLabelDataset(df=df_pred, label_names=['target'], protected_attribute_names=[s_attr_name])

        unprivileged = [{s_attr_name: 0.0}]
        privileged = [{s_attr_name: 1.0}]

        try:
            metric = ClassificationMetric(bld_true, bld_pred, unprivileged_groups=unprivileged, privileged_groups=privileged)
            spd = float(metric.statistical_parity_difference())
            di  = float(metric.disparate_impact())
            eod = float(metric.equal_opportunity_difference())
            aod = float(metric.average_odds_difference())
        except Exception as e:
            print(f"  [Fairness Metric Warning] {e}")
            spd, di, eod, aod = 0.0, 1.0, 0.0, 0.0

        return {
            "statistical_parity_difference": spd,
            "disparate_impact": di,
            "equal_opportunity_difference": eod,
            "average_odds_difference": aod
        }

    def compute_survival_metrics(self, y_true, probs, s_values):
        """
        Computes key clinical survival analysis metrics:
        1. Harrell's Concordance Index (C-Index): discrimination between survivors and non-survivors.
        2. Subgroup Concordance Indices (per bias group) & C-Index Disparity (|C_1 - C_0|).
        3. Brier Score: Mean squared error of survival probability forecasts (lower is better).
        4. Index of Prediction Accuracy (IPA): Scaled Brier score relative to null baseline.
        5. Expected Calibration Error (ECE): Probability reliability across binned risk tiers.
        """
        # Harrell's Concordance Index
        try:
            c_index = float(concordance_index(event_times=y_true, predicted_scores=probs))
        except Exception:
            c_index = 0.5

        # Subgroup C-indices & Disparity
        sub_c = {}
        for g in np.unique(s_values):
            mask = (s_values == g)
            if len(np.unique(y_true[mask])) >= 2:
                try:
                    sub_c[f"c_index_group_{int(g)}"] = float(concordance_index(y_true[mask], probs[mask]))
                except Exception:
                    sub_c[f"c_index_group_{int(g)}"] = 0.5
            else:
                sub_c[f"c_index_group_{int(g)}"] = 0.5

        group_keys = [k for k in sub_c if k.startswith("c_index_group_")]
        if len(group_keys) >= 2:
            c_index_disp = float(abs(sub_c[group_keys[0]] - sub_c[group_keys[1]]))
        else:
            c_index_disp = 0.0

        # Brier Score & Index of Prediction Accuracy (IPA)
        try:
            brier = float(brier_score_loss(y_true, probs))
            null_prob = float(np.mean(y_true))
            brier_null = float(brier_score_loss(y_true, np.full_like(probs, null_prob)))
            ipa = float(1.0 - (brier / brier_null)) if brier_null > 0 else 0.0
        except Exception:
            brier = 0.25
            ipa = 0.0

        # Expected Calibration Error (ECE)
        try:
            n_bins = 5
            bins = np.linspace(0.0, 1.0, n_bins + 1)
            bin_indices = np.digitize(probs, bins) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            ece = 0.0
            for i in range(n_bins):
                mask = (bin_indices == i)
                if np.sum(mask) > 0:
                    bin_acc = np.mean(y_true[mask])
                    bin_conf = np.mean(probs[mask])
                    ece += (np.sum(mask) / len(y_true)) * abs(bin_acc - bin_conf)
            ece = float(ece)
        except Exception:
            ece = 0.0

        return {
            "c_index": c_index,
            "brier_score": brier,
            "ipa": ipa,
            "ece": ece,
            "c_index_disparity": c_index_disp,
            **sub_c
        }

    def get_all_classifiers(self, s_attr_name):
        """
        Defines:
        1. All standard Scikit-Learn models (uncommented & extended).
        2. Expanded AIF360 Mitigation methods (Pre-processing, In-processing, Post-processing).
        """
        svm_c = self.hypes.get("svm_c", 1.0)

        # Base estimators
        base_lr = LogisticRegression(penalty='l2', solver='lbfgs', C=0.1, max_iter=5000, random_state=42)
        base_rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        base_svm_rbf = SVC(kernel='rbf', probability=True, C=svm_c, gamma='scale', random_state=42)
        base_svm_linear = SVC(kernel='linear', probability=True, C=svm_c, random_state=42)
        base_svm_poly = SVC(kernel='poly', degree=3, probability=True, C=svm_c, random_state=42)
        base_svm_sigmoid = SVC(kernel='sigmoid', probability=True, C=svm_c, random_state=42)
        base_boost = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)

        models = {
            # === Standard Scikit-Learn Classifiers ===
            "logistic_l1": LogisticRegression(penalty='l1', solver='liblinear', C=0.5, max_iter=5000, random_state=42),
            "logistic_l2": base_lr,
            "logistic_elastic": LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=0.1, max_iter=5000, random_state=42),
            "ridge": RidgeClassifier(alpha=1.0, random_state=42),
            "sgd": SGDClassifier(loss='log_loss', penalty='l2', alpha=0.01, max_iter=2000, random_state=42),
            "svm_linear": base_svm_linear,
            "svm_rbf": base_svm_rbf,
            "svm_poly": base_svm_poly,
            "svm_sigmoid": base_svm_sigmoid,
            "decision_tree": DecisionTreeClassifier(max_depth=5, min_samples_split=4, random_state=42),
            "random_forest": base_rf,
            "gradient_boosting": base_boost,
            "adaboost": AdaBoostClassifier(n_estimators=50, random_state=42),
            "naive_bayes": GaussianNB(),
            # === Single-Layer MLP Classifiers ===
            "mlp16": MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000, random_state=42),
            "mlp32": MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=42),
            "mlp64": MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=42),
            "mlp128": MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, random_state=42),

            # === AIF360 Pre-Processing Mitigation: Reweighing ===
            "fair_reweighted_lr": LogisticRegression(penalty='l2', solver='lbfgs', C=0.1, max_iter=5000, random_state=42),
            "fair_reweighted_rf": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
            "fair_reweighted_svm_rbf": SVC(kernel='rbf', probability=True, C=svm_c, gamma='scale', random_state=42),
            "fair_reweighted_svm_linear": SVC(kernel='linear', probability=True, C=svm_c, random_state=42),
            "fair_reweighted_svm_poly": SVC(kernel='poly', degree=3, probability=True, C=svm_c, random_state=42),
            "fair_reweighted_svm_sigmoid": SVC(kernel='sigmoid', probability=True, C=svm_c, random_state=42),
            "fair_reweighted_boost": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42),
            "fair_reweighted_ada": AdaBoostClassifier(n_estimators=50, random_state=42),
            "fair_reweighted_dt": DecisionTreeClassifier(max_depth=5, min_samples_split=4, random_state=42),
            "fair_reweighted_nb": GaussianNB(),
            "fair_reweighted_ridge": RidgeClassifier(alpha=1.0, random_state=42),
            "fair_reweighted_sgd": SGDClassifier(loss='log_loss', penalty='l2', alpha=0.01, max_iter=2000, random_state=42),
            "fair_reweighted_mlp16": MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000, random_state=42),
            "fair_reweighted_mlp32": MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=42),
            "fair_reweighted_mlp64": MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=42),
            "fair_reweighted_mlp128": MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, random_state=42),

            # === AIF360 Pre-Processing Mitigation: Disparate Impact Remover ===
            "fair_dir_lr": LogisticRegression(penalty='l2', solver='lbfgs', C=0.1, max_iter=5000, random_state=42),
            "fair_dir_rf": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
            "fair_dir_svm_rbf": SVC(kernel='rbf', probability=True, C=svm_c, gamma='scale', random_state=42),
            "fair_dir_svm_linear": SVC(kernel='linear', probability=True, C=svm_c, random_state=42),
            "fair_dir_svm_poly": SVC(kernel='poly', degree=3, probability=True, C=svm_c, random_state=42),
            "fair_dir_svm_sigmoid": SVC(kernel='sigmoid', probability=True, C=svm_c, random_state=42),
            "fair_dir_boost": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42),
            "fair_dir_ada": AdaBoostClassifier(n_estimators=50, random_state=42),
            "fair_dir_dt": DecisionTreeClassifier(max_depth=5, min_samples_split=4, random_state=42),
            "fair_dir_nb": GaussianNB(),
            "fair_dir_ridge": RidgeClassifier(alpha=1.0, random_state=42),
            "fair_dir_sgd": SGDClassifier(loss='log_loss', penalty='l2', alpha=0.01, max_iter=2000, random_state=42),
            "fair_dir_mlp16": MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000, random_state=42),
            "fair_dir_mlp32": MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=42),
            "fair_dir_mlp64": MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=42),
            "fair_dir_mlp128": MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, random_state=42),

            # === AIF360 In-Processing Mitigation: Prejudice Remover ===
            "fair_prejudice_remover": PrejudiceRemover(eta=1.0, sensitive_attr=s_attr_name),
            "fair_prejudice_remover_eta2": PrejudiceRemover(eta=2.5, sensitive_attr=s_attr_name),

            # === AIF360 In-Processing Mitigation: Reductions (Fairlearn) ===
            "fair_exp_grad_dp_lr": ExponentiatedGradientReduction(
                estimator=LogisticRegression(solver='liblinear'),
                constraints="DemographicParity",
                drop_prot_attr=False
            ),
            "fair_exp_grad_eo_lr": ExponentiatedGradientReduction(
                estimator=LogisticRegression(solver='liblinear'),
                constraints="EqualizedOdds",
                drop_prot_attr=False
            ),
            "fair_exp_grad_eo_rf": ExponentiatedGradientReduction(
                estimator=RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
                constraints="EqualizedOdds",
                drop_prot_attr=False
            ),
            "fair_exp_grad_eo_dt": ExponentiatedGradientReduction(
                estimator=DecisionTreeClassifier(max_depth=5, min_samples_split=4, random_state=42),
                constraints="EqualizedOdds",
                drop_prot_attr=False
            ),
            "fair_grid_search_dp": GridSearchReduction(
                estimator=LogisticRegression(solver='liblinear'),
                constraints="DemographicParity",
                grid_size=10,
                drop_prot_attr=False
            ),
            "fair_grid_search_eo": GridSearchReduction(
                estimator=LogisticRegression(solver='liblinear'),
                constraints="EqualizedOdds",
                grid_size=10,
                drop_prot_attr=False
            ),

            # === AIF360 Post-Processing Mitigation ===
            "fair_roc_rf": ("post_roc", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)),
            "fair_roc_lr": ("post_roc", LogisticRegression(penalty='l2', solver='lbfgs', C=0.1, max_iter=5000)),
            "fair_roc_svm_rbf": ("post_roc", SVC(kernel='rbf', probability=True, C=svm_c, gamma='scale', random_state=42)),
            "fair_roc_svm_linear": ("post_roc", SVC(kernel='linear', probability=True, C=svm_c, random_state=42)),
            "fair_roc_svm_poly": ("post_roc", SVC(kernel='poly', degree=3, probability=True, C=svm_c, random_state=42)),
            "fair_roc_svm_sigmoid": ("post_roc", SVC(kernel='sigmoid', probability=True, C=svm_c, random_state=42)),
            "fair_roc_boost": ("post_roc", GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)),
            "fair_roc_ada": ("post_roc", AdaBoostClassifier(n_estimators=50, random_state=42)),
            "fair_roc_dt": ("post_roc", DecisionTreeClassifier(max_depth=5, min_samples_split=4, random_state=42)),
            "fair_roc_nb": ("post_roc", GaussianNB()),
            "fair_roc_sgd": ("post_roc", SGDClassifier(loss='log_loss', penalty='l2', alpha=0.01, max_iter=2000, random_state=42)),
            "fair_roc_mlp16": ("post_roc", MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000, random_state=42)),
            "fair_roc_mlp32": ("post_roc", MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=42)),
            "fair_roc_mlp64": ("post_roc", MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=42)),
            "fair_roc_mlp128": ("post_roc", MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, random_state=42)),
            "fair_eq_odds_rf": ("post_eq_odds", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)),
            "fair_eq_odds_lr": ("post_eq_odds", LogisticRegression(penalty='l2', solver='lbfgs', C=0.1, max_iter=5000)),
            "fair_eq_odds_svm_rbf": ("post_eq_odds", SVC(kernel='rbf', probability=True, C=svm_c, gamma='scale', random_state=42)),
            "fair_eq_odds_svm_linear": ("post_eq_odds", SVC(kernel='linear', probability=True, C=svm_c, random_state=42)),
            "fair_eq_odds_svm_poly": ("post_eq_odds", SVC(kernel='poly', degree=3, probability=True, C=svm_c, random_state=42)),
            "fair_eq_odds_svm_sigmoid": ("post_eq_odds", SVC(kernel='sigmoid', probability=True, C=svm_c, random_state=42)),
            "fair_eq_odds_boost": ("post_eq_odds", GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)),
            "fair_eq_odds_ada": ("post_eq_odds", AdaBoostClassifier(n_estimators=50, random_state=42)),
            "fair_eq_odds_dt": ("post_eq_odds", DecisionTreeClassifier(max_depth=5, min_samples_split=4, random_state=42)),
            "fair_eq_odds_nb": ("post_eq_odds", GaussianNB()),
            "fair_eq_odds_sgd": ("post_eq_odds", SGDClassifier(loss='log_loss', penalty='l2', alpha=0.01, max_iter=2000, random_state=42)),
            "fair_eq_odds_mlp16": ("post_eq_odds", MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000, random_state=42)),
            "fair_eq_odds_mlp32": ("post_eq_odds", MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=42)),
            "fair_eq_odds_mlp64": ("post_eq_odds", MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=42)),
            "fair_eq_odds_mlp128": ("post_eq_odds", MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, random_state=42)),
            "fair_calib_eq_odds_rf": ("post_calib_eq_odds", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)),
            "fair_calib_eq_odds_lr": ("post_calib_eq_odds", LogisticRegression(penalty='l2', solver='lbfgs', C=0.1, max_iter=5000))
        }

        return models

    def save_confusion_matrix(self, y_true, y_pred, model_name, path):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix: {model_name}')
        plt.ylabel('Actual (2-Year Survival)')
        plt.xlabel('Predicted (2-Year Survival)')
        plt.tight_layout()
        plt.savefig(os.path.join(path, f"{model_name}_cm.png"), dpi=300)
        plt.close()

    def explain_with_shap(self, clf, name, X_tr, X_ts, feat_to_modality, output_dir):
        """
        Computes SHAP values on test set inference, saves summary plots,
        ranked feature importance, and multimodal attribution percentages.
        """
        try:
            print(f"    [SHAP] Generating explainability for {name}...")
            # Determine appropriate explainer
            is_tree = isinstance(clf, (RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier))
            is_linear = isinstance(clf, (LogisticRegression, RidgeClassifier, SGDClassifier))

            if is_tree:
                explainer = shap.TreeExplainer(clf)
                shap_obj = explainer(X_ts)
                shap_vals = shap_obj.values
                if len(shap_vals.shape) == 3:
                    shap_vals = shap_vals[:, :, 1]
            elif is_linear:
                explainer = shap.LinearExplainer(clf, X_tr)
                shap_obj = explainer(X_ts)
                shap_vals = shap_obj.values
            else:
                # Kernel / general Explainer fallback with background sample
                background = shap.sample(X_tr, min(20, len(X_tr)))
                if hasattr(clf, "predict_proba"):
                    predict_fn = lambda x: clf.predict_proba(x)[:, 1]
                elif hasattr(clf, "decision_function"):
                    predict_fn = clf.decision_function
                else:
                    predict_fn = clf.predict
                explainer = shap.KernelExplainer(predict_fn, background, silent=True)
                shap_vals = np.array(explainer.shap_values(X_ts, silent=True))
                if len(shap_vals.shape) == 3:
                    shap_vals = shap_vals[:, :, 1]

            # 1. Save SHAP Values array
            np.save(os.path.join(output_dir, f"{name}_shap_values.npy"), shap_vals)

            # Create shortened feature names for SHAP plots (removes rad_ and 1_original_)
            X_ts_display = X_ts.copy()
            X_ts_display.columns = [shorten_feature_name(c) for c in X_ts.columns]

            # 2. Beeswarm Summary Plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_vals, X_ts_display, show=False, max_display=15)
            plt.title(f"SHAP Summary (Test Inference): {name}", fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{name}_shap_summary.png"), dpi=300)
            plt.close()

            # 3. Bar Plot (Mean |SHAP|)
            plt.figure(figsize=(8, 6))
            shap.summary_plot(shap_vals, X_ts_display, plot_type="bar", show=False, max_display=15)
            plt.title(f"SHAP Feature Importance: {name}", fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{name}_shap_bar.png"), dpi=300)
            plt.close()

            # 4. Feature Importance CSV & Multimodal Attribution Share
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            feat_df = pd.DataFrame({
                "feature": [shorten_feature_name(c) for c in X_ts.columns],
                "raw_feature": X_ts.columns,
                "mean_abs_shap": mean_abs_shap,
                "modality": [feat_to_modality.get(c, "Unknown") for c in X_ts.columns]
            }).sort_values(by="mean_abs_shap", ascending=False)
            feat_df.to_csv(os.path.join(output_dir, f"{name}_shap_importance.csv"), index=False)

            # Modality contribution share
            mod_grouped = feat_df.groupby("modality")["mean_abs_shap"].sum()
            total_shap = mod_grouped.sum()
            if total_shap > 0:
                mod_pct = (mod_grouped / total_shap) * 100
                plt.figure(figsize=(6, 4))
                palette = {"Clinical": "#1f77b4", "Radiomics": "#ff7f0e", "RNA-seq": "#2ca02c"}
                colors = [palette.get(m, "#7f7f7f") for m in mod_pct.index]
                mod_pct.plot(kind="bar", color=colors)
                plt.ylabel("Attribution Share (%)")
                plt.title(f"Multimodal Modality Attribution: {name}")
                plt.ylim(0, 100)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{name}_modality_importance.png"), dpi=300)
                plt.close()

        except Exception as e:
            print(f"    [SHAP Warning] Skipping SHAP for {name} due to: {e}")

    def run_pipeline(self, train_data, test_data, X_tr_fused, X_ts_fused, feat_to_modality, output_dir=None, training_type="biased"):
        """Runs training, mitigation, inference evaluation, fairness auditing, and SHAP explainability."""
        y_train = train_data["y"]
        s_train = train_data["s"]
        y_test  = test_data["y"]
        s_test  = test_data["s"]
        s_attr_name = train_data["bias_col"]

        if output_dir is None:
            output_dir = f"chkp/{train_data.get('bias_name', 'experiment')}"
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n[Benchmarker] Writing all outputs and models to hierarchy: {output_dir}")

        if str(training_type).upper() == "BGMMOCE":
            # BGMMOCE dataset is already the data-mitigated benchmark (already fair).
            # No reweighing or other algorithmic fairness experiments are run on BGMMOCE.
            # Train all base classifiers on BGMMOCE to compare directly with their biased baselines and AIF360 mitigations.
            all_clfs = self.get_all_classifiers(s_attr_name)
            classifiers = {k: v for k, v in all_clfs.items() if not k.startswith("fair_")}
            reweigh_weights = None
            X_tr_repaired, X_ts_repaired = None, None
            bld_tr, bld_ts = None, None
        else:
            classifiers = self.get_all_classifiers(s_attr_name)
            unprivileged_groups = [{s_attr_name: 0.0}]
            privileged_groups   = [{s_attr_name: 1.0}]

            # Prepare base AIF360 datasets
            bld_tr = self.pandas_to_aif360(X_tr_fused, y_train, s_train, s_attr_name)
            bld_ts = self.pandas_to_aif360(X_ts_fused, y_test, s_test, s_attr_name)

            # Precompute Reweighing instance weights
            reweigh_weights = self.get_aif360_weights(X_tr_fused, y_train, s_train, s_attr_name)

            # Precompute Disparate Impact Remover repaired datasets
            dir_remover = DisparateImpactRemover(repair_level=1.0, sensitive_attribute=s_attr_name)
            bld_tr_repaired = dir_remover.fit_transform(bld_tr)
            bld_ts_repaired = dir_remover.fit_transform(bld_ts)
            # Drop the sensitive attribute from repaired feature matrix
            sens_idx = bld_tr.feature_names.index(s_attr_name)
            X_tr_repaired = np.delete(bld_tr_repaired.features, sens_idx, axis=1)
            X_ts_repaired = np.delete(bld_ts_repaired.features, sens_idx, axis=1)

        summary_rows = []

        for name, clf_item in classifiers.items():
            print(f"\nEvaluating: {name}...")

            try:
                # --------------------------------------------------------------
                # CASE A: AIF360 In-Processing (PrejudiceRemover, Reductions)
                # --------------------------------------------------------------
                if "fair_prejudice" in name or "fair_exp_grad" in name or "fair_grid_search" in name:
                    clf = clf_item
                    clf.fit(bld_tr)
                    pred_bld = clf.predict(bld_ts)
                    preds = pred_bld.labels.flatten().astype(int)
                    if hasattr(pred_bld, "scores") and pred_bld.scores is not None and len(pred_bld.scores) > 0:
                        probs = pred_bld.scores.flatten()
                    else:
                        probs = preds.astype(float)
                    fit_model_for_save = clf

                # --------------------------------------------------------------
                # CASE B: AIF360 Pre-Processing: Disparate Impact Remover
                # --------------------------------------------------------------
                elif "fair_dir_" in name:
                    clf = clf_item
                    clf.fit(X_tr_repaired, y_train)
                    preds = clf.predict(X_ts_repaired)
                    if hasattr(clf, "predict_proba"):
                        probs = clf.predict_proba(X_ts_repaired)[:, 1]
                    else:
                        probs = preds.astype(float)
                    fit_model_for_save = clf

                # --------------------------------------------------------------
                # CASE C: AIF360 Pre-Processing: Reweighing
                # --------------------------------------------------------------
                elif "fair_reweighted" in name:
                    clf = clf_item
                    clf.fit(X_tr_fused, y_train, sample_weight=reweigh_weights)
                    preds = clf.predict(X_ts_fused)
                    if hasattr(clf, "predict_proba"):
                        probs = clf.predict_proba(X_ts_fused)[:, 1]
                    else:
                        probs = preds.astype(float)
                    fit_model_for_save = clf

                # --------------------------------------------------------------
                # CASE D: AIF360 Post-Processing (ROC, EqOdds, CalibEqOdds)
                # --------------------------------------------------------------
                elif isinstance(clf_item, tuple) and clf_item[0].startswith("post_"):
                    post_type, base_clf = clf_item
                    # 1. Fit base model on training data
                    base_clf.fit(X_tr_fused, y_train)

                    # 2. Get predictions on training set for post-processor fitting
                    tr_probs = base_clf.predict_proba(X_tr_fused)[:, 1]
                    tr_preds = (tr_probs >= 0.5).astype(float)
                    bld_tr_pred = bld_tr.copy()
                    bld_tr_pred.scores = tr_probs.reshape(-1, 1)
                    bld_tr_pred.labels = tr_preds.reshape(-1, 1)

                    # 3. Fit specific post-processor
                    if post_type == "post_roc":
                        post_proc = RejectOptionClassification(
                            unprivileged_groups=unprivileged_groups,
                            privileged_groups=privileged_groups,
                            metric_name="Statistical parity difference"
                        )
                    elif post_type == "post_eq_odds":
                        post_proc = EqOddsPostprocessing(
                            unprivileged_groups=unprivileged_groups,
                            privileged_groups=privileged_groups
                        )
                    elif post_type == "post_calib_eq_odds":
                        post_proc = CalibratedEqOddsPostprocessing(
                            unprivileged_groups=unprivileged_groups,
                            privileged_groups=privileged_groups,
                            cost_constraint="weighted"
                        )
                    post_proc.fit(bld_tr, bld_tr_pred)

                    # 4. Apply to test set
                    ts_probs = base_clf.predict_proba(X_ts_fused)[:, 1]
                    ts_preds = (ts_probs >= 0.5).astype(float)
                    bld_ts_pred = bld_ts.copy()
                    bld_ts_pred.scores = ts_probs.reshape(-1, 1)
                    bld_ts_pred.labels = ts_preds.reshape(-1, 1)

                    bld_ts_transformed = post_proc.predict(bld_ts_pred)
                    preds = bld_ts_transformed.labels.flatten().astype(int)
                    probs = bld_ts_transformed.scores.flatten() if bld_ts_transformed.scores is not None else preds.astype(float)
                    fit_model_for_save = (base_clf, post_proc)

                # --------------------------------------------------------------
                # CASE E: Standard Scikit-Learn Classifiers
                # --------------------------------------------------------------
                else:
                    clf = clf_item
                    # Optional SMOTE balancing on training set if requested
                    if self.hypes.get("use_smote", False):
                        try:
                            sm = SMOTE(
                                sampling_strategy=self.hypes.get("smote_strategy", 1.0),
                                k_neighbors=min(5, np.sum(y_train == 0) - 1),
                                random_state=42
                            )
                            X_tr_fit, y_tr_fit = sm.fit_resample(X_tr_fused, y_train)
                        except Exception as e:
                            print(f"  [SMOTE Warning] {e}. Falling back to unresampled data.")
                            X_tr_fit, y_tr_fit = X_tr_fused, y_train
                    else:
                        X_tr_fit, y_tr_fit = X_tr_fused, y_train

                    clf.fit(X_tr_fit, y_tr_fit)
                    preds = clf.predict(X_ts_fused)
                    if hasattr(clf, "predict_proba"):
                        probs = clf.predict_proba(X_ts_fused)[:, 1]
                    elif hasattr(clf, "decision_function"):
                        df_scores = clf.decision_function(X_ts_fused)
                        # Min-max scale decision function to [0, 1]
                        probs = (df_scores - df_scores.min()) / (df_scores.max() - df_scores.min() + 1e-8)
                    else:
                        probs = preds.astype(float)
                    fit_model_for_save = clf

                # --------------------------------------------------------------
                # Metrics Calculation
                # --------------------------------------------------------------
                try:
                    auc = float(roc_auc_score(y_test, probs))
                except Exception:
                    auc = 0.5

                acc = float(accuracy_score(y_test, preds))
                f1  = float(f1_score(y_test, preds, zero_division=0))
                prec = float(precision_score(y_test, preds, zero_division=0))
                rec = float(recall_score(y_test, preds, zero_division=0))

                # Survival Metrics (Concordance Index, Brier, IPA, ECE, Subgroup C-index)
                survival_metrics = self.compute_survival_metrics(y_test, probs, s_test)

                # Fairness Metrics
                fairness_metrics = self.compute_fairness_metrics(y_test, preds, s_test, s_attr_name)
                fpr, tpr, _ = roc_curve(y_test, probs)

                # Log results structure compatible with getResultsV2.py & plotClassFairness.py
                self.results_data[name] = {
                    "mean_auc": auc,
                    "std_auc": 0.0,
                    "c_index": survival_metrics["c_index"],
                    "brier_score": survival_metrics["brier_score"],
                    "ipa": survival_metrics["ipa"],
                    "ece": survival_metrics["ece"],
                    "accuracy": acc,
                    "f1": f1,
                    "precision": prec,
                    "recall": rec,
                    "survival": survival_metrics,
                    "fairness": fairness_metrics,
                    "folds": [{
                        "auc": auc,
                        "c_index": survival_metrics["c_index"],
                        "brier_score": survival_metrics["brier_score"],
                        "f1": f1,
                        "accuracy": acc,
                        "fpr": fpr.tolist(),
                        "tpr": tpr.tolist()
                    }]
                }

                summary_rows.append({
                    "Model": name,
                    "AUC": round(auc, 4),
                    "C_Index": round(survival_metrics["c_index"], 4),
                    "Brier_Score": round(survival_metrics["brier_score"], 4),
                    "IPA": round(survival_metrics["ipa"], 4),
                    "ECE": round(survival_metrics["ece"], 4),
                    "Accuracy": round(acc, 4),
                    "F1": round(f1, 4),
                    "Precision": round(prec, 4),
                    "Recall": round(rec, 4),
                    "Statistical_Parity_Diff": round(fairness_metrics["statistical_parity_difference"], 4),
                    "Disparate_Impact": round(fairness_metrics["disparate_impact"], 4),
                    "Equal_Opportunity_Diff": round(fairness_metrics["equal_opportunity_difference"], 4),
                    "Average_Odds_Diff": round(fairness_metrics["average_odds_difference"], 4),
                    "C_Index_Disparity": round(survival_metrics["c_index_disparity"], 4)
                })

                print(f"  -> AUC: {auc:.4f} | C-Index: {survival_metrics['c_index']:.4f} | Brier: {survival_metrics['brier_score']:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | StatParityDiff: {fairness_metrics['statistical_parity_difference']:.4f}")

                # Save Model and Selected Features into classifier/variant directory
                clf_family, variant_name = get_classifier_family_and_variant(name, training_type)
                variant_dir = os.path.join(output_dir, clf_family, variant_name)
                os.makedirs(variant_dir, exist_ok=True)

                with open(os.path.join(variant_dir, f"{name}_clf.pkl"), "wb") as f:
                    pkl.dump(fit_model_for_save, f, protocol=pkl.HIGHEST_PROTOCOL)
                np.save(os.path.join(variant_dir, f"{name}_selected_feats.npy"), np.array(X_tr_fused.columns))
                
                # Also save standard names for cross-script compatibility
                with open(os.path.join(variant_dir, "model.pkl"), "wb") as f:
                    pkl.dump(fit_model_for_save, f, protocol=pkl.HIGHEST_PROTOCOL)
                np.save(os.path.join(variant_dir, "selected_feats.npy"), np.array(X_tr_fused.columns))

                # Save predictions and true test labels for downstream disparity evaluation
                np.save(os.path.join(variant_dir, "preds.npy"), np.array(preds))
                np.save(os.path.join(variant_dir, "probs.npy"), np.array(probs))
                np.save(os.path.join(variant_dir, "y_test.npy"), np.array(y_test))
                np.save(os.path.join(variant_dir, "s_test.npy"), np.array(s_test))

                # Save metrics JSON for this specific model
                with open(os.path.join(variant_dir, f"{name}_metrics.json"), "w") as f:
                    json.dump(self.results_data[name], f, indent=4)

                # Save confusion matrix in variant_dir
                self.save_confusion_matrix(y_test, preds, name, variant_dir)

                # SHAP Explainability (Run for standard models, reweighted models, and tree ensembles)
                run_shap = self.hypes.get("run_shap", True)
                shap_targets = self.hypes.get("shap_models", [
                    "random_forest", "gradient_boosting", "logistic_l2",
                    "svm_rbf", "svm_linear", "svm_poly", "svm_sigmoid",
                    "decision_tree",
                    "fair_reweighted_rf", "fair_reweighted_lr", "fair_reweighted_boost", "fair_reweighted_dt",
                    "fair_reweighted_svm_rbf", "fair_reweighted_svm_linear", "fair_reweighted_svm_poly", "fair_reweighted_svm_sigmoid",
                    "adaboost", "naive_bayes",
                    "mlp16", "mlp32", "mlp64", "mlp128",
                    "fair_reweighted_mlp16", "fair_reweighted_mlp32", "fair_reweighted_mlp64", "fair_reweighted_mlp128"
                ])
                if run_shap and (shap_targets == "all" or name in shap_targets):
                    # For DIR models, pass repaired features; otherwise fused features
                    if "fair_dir_" in name:
                        feat_map_rep = {f"rep_feat_{i}": "Repaired" for i in range(X_tr_repaired.shape[1])}
                        df_tr_rep = pd.DataFrame(X_tr_repaired, columns=list(feat_map_rep.keys()))
                        df_ts_rep = pd.DataFrame(X_ts_repaired, columns=list(feat_map_rep.keys()))
                        self.explain_with_shap(clf, name, df_tr_rep, df_ts_rep, feat_map_rep, variant_dir)
                    elif not ("fair_prejudice" in name or "fair_exp_grad" in name or "fair_grid_search" in name or isinstance(clf_item, tuple)):
                        self.explain_with_shap(clf, name, X_tr_fused, X_ts_fused, feat_to_modality, variant_dir)

            except Exception as e:
                print(f"  [!] Failed evaluation for {name}: {e}")

        # Update / merge benchmark_results.json at bias level
        json_path = os.path.join(output_dir, "benchmark_results.json")
        combined_results = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    combined_results = json.load(f)
            except Exception:
                combined_results = {}
        for k, v in self.results_data.items():
            key_tag = f"{k}_BGMMOCE" if training_type.upper() == "BGMMOCE" else k
            combined_results[key_tag] = v
        with open(json_path, "w") as f:
            json.dump(combined_results, f, indent=4)

        # Update / merge summary CSV at bias level
        csv_path = os.path.join(output_dir, "benchmark_summary.csv")
        new_summary_df = pd.DataFrame(summary_rows)
        new_summary_df["Training_Split"] = training_type
        if os.path.exists(csv_path):
            try:
                old_df = pd.read_csv(csv_path)
                combined_df = pd.concat([old_df[old_df.get("Training_Split", "") != training_type], new_summary_df], ignore_index=True)
            except Exception:
                combined_df = new_summary_df
        else:
            combined_df = new_summary_df
        combined_df.to_csv(csv_path, index=False)

        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY (Top 10 by AUC):")
        print("=" * 80)
        if not combined_df.empty:
            print(combined_df.sort_values(by="AUC", ascending=False).head(10).to_string(index=False))
        print(f"\nAll results, plots, models, and SHAP explainability saved to:\n  {output_dir}\n")


def get_experiment_dir(base_dir="chkp", exp_name=None):
    """
    Returns the target experiment folder path: chkp/exp_[day.month]_[23h59m]/
    If exp_name is provided, uses that name.
    Otherwise, reuses a session created within the last 45 minutes or generates
    a new timestamped name: exp_<DD.MM>_<HHhMMm>.
    """
    os.makedirs(base_dir, exist_ok=True)
    if exp_name:
        target = os.path.join(base_dir, exp_name)
        os.makedirs(target, exist_ok=True)
        return target

    now = datetime.now()
    today_prefix = f"exp_{now.strftime('%d.%m')}_"
    existing_today = sorted(glob.glob(os.path.join(base_dir, f"{today_prefix}*")))
    if existing_today:
        latest = existing_today[-1]
        # If created within the last 45 minutes, reuse it so biased and BGMMOCE runs join the same session
        if (time.time() - os.path.getmtime(latest)) < 2700:
            return latest

    default_name = f"exp_{now.strftime('%d.%m')}_{now.strftime('%Hh%Mm')}"
    target = os.path.join(base_dir, default_name)
    os.makedirs(target, exist_ok=True)
    return target


# ==============================================================================
# 4. Main Execution Block
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="NSCLC 2-Year Survival Multimodal Fairness Benchmark")
    parser.add_argument("--bias", type=str, default="affiliation", choices=["affiliation", "gender", "ethnicity", "all"],
                        help="Bias/Sensitive attribute to evaluate (affiliation, gender, ethnicity, or all)")
    parser.add_argument("--training-type", type=str, default="biased", choices=["biased", "BGMMOCE"],
                        help="Training dataset split: 'biased' or 'BGMMOCE'")
    parser.add_argument("--use-smote", action="store_true", default=False,
                        help="Apply SMOTE resampling to training data")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Experiment folder name under chkp/, e.g. exp_18.10_23h59m")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Custom output directory path")
    args = parser.parse_args()

    hypes = {
        "p_one": 5.0,
        "p_two": 95.0,
        "variance_threshold": 0.0,
        "use_custom_anova": True,
        "p_value": 0.005,
        "k_best": 40,
        "min_features_per_modality": 5,
        "l1_c_value": 0.5,
        "use_smote": args.use_smote,
        "smote_strategy": 1.0,
        "svm_c": 1.0,
        "run_shap": True,
        "shap_models": [
            "random_forest", "gradient_boosting", "logistic_l2",
            "svm_rbf", "svm_linear", "svm_poly", "svm_sigmoid",
            "decision_tree",
            "fair_reweighted_rf", "fair_reweighted_lr", "fair_reweighted_boost", "fair_reweighted_dt",
            "fair_reweighted_svm_rbf", "fair_reweighted_svm_linear", "fair_reweighted_svm_poly", "fair_reweighted_svm_sigmoid",
            "adaboost", "naive_bayes",
            "mlp16", "mlp32", "mlp64", "mlp128",
            "fair_reweighted_mlp16", "fair_reweighted_mlp32", "fair_reweighted_mlp64", "fair_reweighted_mlp128"
        ]
    }

    loader = MultimodalDataLoader(base_dir="dataset")
    biases_to_run = ["affiliation", "gender", "ethnicity"] if args.bias == "all" else [args.bias]

    # Resolve target experiment root folder: chkp/exp_[day.month]_[23h59m]/
    if args.output_dir:
        exp_dir = args.output_dir
    else:
        exp_dir = get_experiment_dir("chkp", args.exp_name)

    print("\n" + "=" * 80)
    print(f"EXPERIMENT ROOT DIRECTORY: {exp_dir}")
    print("=" * 80)

    for bias_type in biases_to_run:
        print("\n" + "#" * 80)
        print(f"RUNNING BENCHMARK: Bias={bias_type.upper()} | TrainingSet={args.training_type} | Endpoint=2_Year_Survival")
        print("#" * 80)

        # 1. Load Data
        train_data, test_data = loader.load_dataset(bias_type=bias_type, training_set_type=args.training_type)

        # 2. Multi-Modal Early-Fusion & Feature Selection
        fusion_selector = MultimodalEarlyFusionSelector(hypes)
        X_tr_fused, X_ts_fused = fusion_selector.fit_transform(train_data, test_data)

        # 3. Setup output folder: bias -> classifier -> variant inside experiment folder
        out_dir = os.path.join(exp_dir, bias_type)
        os.makedirs(out_dir, exist_ok=True)

        # Save selected features list in TXT form prior to classification
        fusion_selector.save_selected_features_txt(out_dir, bias_type, args.training_type)

        # 4. Run Benchmark & Explainability
        benchmarker = ComprehensiveBenchmarker(hypes)
        benchmarker.run_pipeline(
            train_data=train_data,
            test_data=test_data,
            X_tr_fused=X_tr_fused,
            X_ts_fused=X_ts_fused,
            feat_to_modality=fusion_selector.feature_modality_map,
            output_dir=out_dir,
            training_type=args.training_type
        )

    # 5. Check if both 'biased' and 'BGMMOCE' variants exist in exp_dir to automatically generate reports and disparity plots
    has_biased = any(glob.glob(os.path.join(exp_dir, "*", "*", "biased")))
    has_bgmmoce = any(glob.glob(os.path.join(exp_dir, "*", "*", "BGMMOCE")))
    if has_biased and has_bgmmoce:
        print("\n" + "=" * 80)
        print("DETECTED BOTH BIASED AND BGMMOCE RESULTS: Running Automated Post-Processing")
        print("=" * 80)
        py_bin = sys.executable
        try:
            import subprocess
            print(">>> Running getResultsV2.py...")
            subprocess.run([py_bin, "getResultsV2.py", "--base-dir", exp_dir], check=False)
            print(">>> Running plotClassFairness.py (standard + best_performing_models)...")
            subprocess.run([py_bin, "plotClassFairness.py", "--base-dir", exp_dir], check=False)
            print(">>> Automated Post-Processing complete.")
        except Exception as e:
            print(f"  [!] Note: Automated post-processing encountered an issue: {e}")


if __name__ == '__main__':
    main()