#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class Fairness & Disparity Visualizer for Lung Cancer 2-Year Survival Multi-Modal Benchmark

Visualizes per bias (affiliation, gender, ethnicity):
1. Fairness Gap Reduction (Difference between Privileged & Unprivileged groups)
2. Selection Rate Disparity by subgroup
3. Error Rate by bias subgroup (Misclassification disparity)

Generates two sets of figures:
A. Standard figures across all classifier families:
   - fairness_gap_reduction_[bias].png
   - fairness_selection_rate_disparity_[bias].png
   - fairness_error_rate_by_subgroup_[bias].png
   Saved in: chkp/exp_[date]_[time]/ (and chkp/exp_[date]_[time]/[bias]/)

B. Best-performing models figures (only triplets with at least one model > 0.70 AUC):
   - fairness_gap_reduction_[bias].png
   - fairness_selection_rate_disparity_[bias].png
   - fairness_error_rate_by_subgroup_[bias].png
   Saved in: chkp/exp_[date]_[time]/best_performing_models/ (and best_performing_models/[bias]/)

@author: Eleftherios Trivizakis & Google DeepMind Antigravity
"""

import os
import glob
import json
import argparse
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import RobustScaler

COLOR_PALETTE = {
    "Biased": "#d62728",   # Red
    "Fair": "#2ca02c",     # Green
    "BGMMOCE": "#1f77b4"   # Blue
}

CLF_DISPLAY_NAMES = {
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "svm_rbf": "SVM (RBF)",
    "svm_linear": "SVM (Linear)",
    "svm_poly": "SVM (Poly)",
    "svm_sigmoid": "SVM (Sigmoid)",
    "svm": "SVM",
    "gradient_boosting": "Gradient Boosting",
    "adaboost": "AdaBoost",
    "mlp16": "MLP (16)",
    "mlp32": "MLP (32)",
    "mlp64": "MLP (64)",
    "mlp128": "MLP (128)",
    "mlp": "MLP",
    "decision_tree": "Decision Tree",
    "naive_bayes": "Naive Bayes",
    "ridge": "Ridge",
    "sgd": "SGD"
}

def clean_display_name(clf_key):
    return CLF_DISPLAY_NAMES.get(clf_key, str(clf_key).replace("_", " ").title())

def get_model_category(clf_name):
    """Maps raw classifier names to visualization categories."""
    name = clf_name.lower()
    if "bgmmoce" in name or "mitigated" in name:
        return "BGMMOCE"
    elif "fair" in name or "reweigh" in name:
        return "Fair"
    else:
        return "Biased"

def filter_best_performing_triplets(bias_data, auc_threshold=0.70):
    """
    Filters bias_data to only include classifier families (triplets) where
    at least one model variant (Biased, Fair, BGMMOCE) has AUC > auc_threshold.
    """
    filtered = {}
    for clf_fam, variants in bias_data.items():
        aucs = [v.get("auc") for v in variants.values() if v.get("auc") is not None]
        if aucs and max(aucs) > auc_threshold:
            filtered[clf_fam] = variants
    return filtered

def plot_fairness_gap_reduction_per_bias(bias_name, bias_data, output_dir, is_best_performing=False):
    """
    Generates fairness_gap_reduction_<bias>.png
    Shows Accuracy Disparity Gap (|Acc_Privileged - Acc_Unprivileged|) across classifiers for:
    - Biased (Red)
    - Fair (Green)
    - BGMMOCE (Blue)
    """
    plot_rows = []
    for clf_fam, variants in bias_data.items():
        clf_label = clean_display_name(clf_fam)
        for cat in ["Biased", "Fair", "BGMMOCE"]:
            if cat not in variants:
                continue
            data = variants[cat]
            y_pred = np.array(data["preds"])
            y_true = np.array(data["y_true"])
            biases = np.array(data["biases"])

            subgroup_accs = []
            for g in np.unique(biases):
                mask = (biases == g)
                if np.sum(mask) > 0:
                    subgroup_accs.append(accuracy_score(y_true[mask], y_pred[mask]))

            if len(subgroup_accs) >= 2:
                gap = abs(max(subgroup_accs) - min(subgroup_accs))
                plot_rows.append({
                    "Classifier": clf_label,
                    "Model Type": cat,
                    "Accuracy Gap": gap
                })

    if not plot_rows:
        print(f"  [!] No disparity gap data available for bias: {bias_name}")
        return

    df = pd.DataFrame(plot_rows)
    n_clfs = len(df["Classifier"].unique())
    fig_width = max(7.0, min(14.0, n_clfs * 1.5 + 2.0))

    plt.figure(figsize=(fig_width, 5.5))
    ax = sns.barplot(
        data=df,
        x="Classifier", y="Accuracy Gap", hue="Model Type",
        hue_order=["Biased", "Fair", "BGMMOCE"],
        palette=COLOR_PALETTE, edgecolor="black", alpha=0.9
    )
    plt.title("Fairness Gap Reduction", fontsize=14, fontweight='bold', pad=14)
    plt.ylabel("Accuracy Gap", fontsize=11)
    plt.xlabel("Classifier Family", fontsize=11)
    plt.xticks(rotation=20 if n_clfs > 4 else 0, ha='right' if n_clfs > 4 else 'center', fontsize=10)
    plt.ylim(0, max(0.4, df["Accuracy Gap"].max() * 1.15))
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title="Model Type", frameon=True)
    plt.tight_layout()

    # Save at root output dir and inside bias folder if bias subfolder exists
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"fairness_gap_reduction_{bias_name}.png")
    plt.savefig(save_path, dpi=300)
    bias_subpath = os.path.join(output_dir, bias_name, f"fairness_gap_reduction_{bias_name}.png")
    if os.path.exists(os.path.join(output_dir, bias_name)):
        plt.savefig(bias_subpath, dpi=300)
    plt.close()
    print(f"  -> Saved: {save_path}")

def plot_selection_rate_disparity_per_bias(bias_name, bias_data, output_dir, is_best_performing=False):
    """
    Generates fairness_selection_rate_disparity_<bias>.png
    Shows Selection Rate across subgroups for each classifier.
    """
    plot_rows = []
    for clf_fam, variants in bias_data.items():
        clf_label = clean_display_name(clf_fam)
        for cat in ["Biased", "Fair", "BGMMOCE"]:
            if cat not in variants:
                continue
            data = variants[cat]
            y_pred = np.array(data["preds"])
            biases = np.array(data["biases"])

            for g in np.unique(biases):
                mask = (biases == g)
                if np.sum(mask) > 0:
                    sel_rate = np.mean(y_pred[mask])
                    plot_rows.append({
                        "Classifier": clf_label,
                        "Model Type": cat,
                        "Subgroup": f"Group {int(g)}",
                        "Selection Rate": sel_rate
                    })

    if not plot_rows:
        print(f"  [!] No selection rate data available for bias: {bias_name}")
        return

    df = pd.DataFrame(plot_rows)
    n_clfs = len(df["Classifier"].unique())
    col_wrap = min(4, max(1, n_clfs))

    g = sns.FacetGrid(df, col="Classifier", col_wrap=col_wrap, height=3.5, aspect=1.1, sharey=True)
    g.map_dataframe(
        sns.pointplot,
        x="Subgroup", y="Selection Rate", hue="Model Type",
        hue_order=["Biased", "Fair", "BGMMOCE"],
        palette=COLOR_PALETTE,
        markers=["o", "s", "D"], linestyles=["-", "--", "-."],
        dodge=0.2, scale=0.9
    )
    g.add_legend(title="Model Type")
    g.set_titles(col_template="{col_name}", size=11)
    g.set_ylabels("Selection Rate", fontsize=11)
    g.set_xlabels("Subgroup", fontsize=11)
    plt.subplots_adjust(top=0.88)
    g.fig.suptitle("Fairness Selection Rate", fontsize=14, fontweight='bold')

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"fairness_selection_rate_disparity_{bias_name}.png")
    plt.savefig(save_path, dpi=300)
    bias_subpath = os.path.join(output_dir, bias_name, f"fairness_selection_rate_disparity_{bias_name}.png")
    if os.path.exists(os.path.join(output_dir, bias_name)):
        plt.savefig(bias_subpath, dpi=300)
    plt.close()
    print(f"  -> Saved: {save_path}")

def plot_error_rate_by_subgroup_per_bias(bias_name, bias_data, output_dir, is_best_performing=False):
    """
    Generates fairness_error_rate_by_subgroup_<bias>.png
    Shows Misclassification Error Rate across subgroups for each classifier.
    """
    plot_rows = []
    for clf_fam, variants in bias_data.items():
        clf_label = clean_display_name(clf_fam)
        for cat in ["Biased", "Fair", "BGMMOCE"]:
            if cat not in variants:
                continue
            data = variants[cat]
            y_pred = np.array(data["preds"])
            y_true = np.array(data["y_true"])
            biases = np.array(data["biases"])

            for g in np.unique(biases):
                mask = (biases == g)
                if np.sum(mask) > 0:
                    acc = accuracy_score(y_true[mask], y_pred[mask])
                    plot_rows.append({
                        "Classifier": clf_label,
                        "Model Type": cat,
                        "Subgroup": f"Group {int(g)}",
                        "Error Rate": 1.0 - acc
                    })

    if not plot_rows:
        print(f"  [!] No error rate data available for bias: {bias_name}")
        return

    df = pd.DataFrame(plot_rows)
    n_clfs = len(df["Classifier"].unique())
    col_wrap = min(4, max(1, n_clfs))

    g = sns.catplot(
        data=df,
        x="Subgroup", y="Error Rate", hue="Model Type", col="Classifier",
        hue_order=["Biased", "Fair", "BGMMOCE"],
        kind="bar", col_wrap=col_wrap, height=3.5, aspect=1.1,
        palette=COLOR_PALETTE, edgecolor="black", alpha=0.85, sharey=True
    )
    g.set_titles(col_template="{col_name}", size=11)
    g.set_ylabels("Error Rate", fontsize=11)
    g.set_xlabels("Subgroup", fontsize=11)
    plt.subplots_adjust(top=0.88)
    g.fig.suptitle("Fairness Error Rate", fontsize=14, fontweight='bold')

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"fairness_error_rate_by_subgroup_{bias_name}.png")
    plt.savefig(save_path, dpi=300)
    bias_subpath = os.path.join(output_dir, bias_name, f"fairness_error_rate_by_subgroup_{bias_name}.png")
    if os.path.exists(os.path.join(output_dir, bias_name)):
        plt.savefig(bias_subpath, dpi=300)
    plt.close()
    print(f"  -> Saved: {save_path}")

def load_all_scaled_test_features(loader, bias, training_type="biased"):
    """Loads raw train and test data and robust-scales all features across modalities."""
    train_data, test_data = loader.load_dataset(bias_type=bias, training_set_type=training_type)
    scaler = RobustScaler(quantile_range=(5.0, 95.0))
    
    X_tr_clin = pd.DataFrame(scaler.fit_transform(train_data["clinical"]), columns=train_data["clinical"].columns)
    X_ts_clin = pd.DataFrame(scaler.transform(test_data["clinical"]), columns=test_data["clinical"].columns)
    
    X_tr_rad = pd.DataFrame(scaler.fit_transform(train_data["radiomics"]), columns=train_data["radiomics"].columns)
    X_ts_rad = pd.DataFrame(scaler.transform(test_data["radiomics"]), columns=test_data["radiomics"].columns)
    
    X_tr_rna = pd.DataFrame(scaler.fit_transform(train_data["rnaseq"]), columns=train_data["rnaseq"].columns)
    X_ts_rna = pd.DataFrame(scaler.transform(test_data["rnaseq"]), columns=test_data["rnaseq"].columns)
    
    X_ts_full = pd.concat([X_ts_clin, X_ts_rad, X_ts_rna], axis=1)
    return X_ts_full, test_data["y"], test_data["s"]

def find_target_dir(base_chkp="chkp", exp_name=None):
    if exp_name:
        path = os.path.join(base_chkp, exp_name)
        if os.path.exists(path):
            return path
        return exp_name

    exp_folders = sorted(glob.glob(os.path.join(base_chkp, "exp_*")))
    if exp_folders:
        return exp_folders[-1]

    return base_chkp

def main():
    parser = argparse.ArgumentParser(description="Disparity and Fairness Visualization per Bias")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Specific experiment folder under chkp/, e.g. exp_18.10_23h59m")
    parser.add_argument("--base-dir", type=str, default=None,
                        help="Base directory containing bias folders")
    parser.add_argument("--auc-threshold", type=float, default=0.70,
                        help="AUC threshold for best performing models (default: 0.70)")
    args = parser.parse_args()

    target_dir = args.base_dir if args.base_dir else find_target_dir("chkp", args.exp_name)
    print("=" * 80)
    print(f"DISPARITY & FAIRNESS PLOT GENERATOR")
    print(f"Target Directory: {target_dir}")
    print(f"Best Performing Models AUC Threshold: > {args.auc_threshold:.2f}")
    print("=" * 80)

    from analysisv3 import MultimodalDataLoader
    loader = MultimodalDataLoader(base_dir="dataset")
    biases = ["affiliation", "gender", "ethnicity"]
    target_clfs = [
        "random_forest", "logistic_regression",
        "svm_rbf", "svm_linear", "svm_poly", "svm_sigmoid",
        "gradient_boosting", "decision_tree", "adaboost", "naive_bayes",
        "mlp16", "mlp32", "mlp64", "mlp128",
        "ridge", "sgd"
    ]

    best_dir = os.path.join(target_dir, "best_performing_models")
    os.makedirs(best_dir, exist_ok=True)
    all_summary_records = []

    for bias in biases:
        bias_dir = os.path.join(target_dir, bias)
        if not os.path.exists(bias_dir):
            print(f"\n[Skip] No results directory for bias '{bias}' at: {bias_dir}")
            continue

        print(f"\nProcessing Fairness & Disparity Visualizations for: {bias.upper()}")
        bias_data = {}  # {clf_fam: {"Biased": {...}, "Fair": {...}, "BGMMOCE": {...}}}

        # Preload fallback test features if needed
        cached_ts_biased = None
        cached_ts_bgmmoce = None

        # Dynamically discover all classifier folders present
        available_clf_folders = [
            d for d in os.listdir(bias_dir)
            if os.path.isdir(os.path.join(bias_dir, d)) and not d.startswith(".")
        ]
        # Sort so standard target clfs appear first
        sorted_clfs = [c for c in target_clfs if c in available_clf_folders] + [c for c in available_clf_folders if c not in target_clfs]

        for clf_fam in sorted_clfs:
            clf_fam_dir = os.path.join(bias_dir, clf_fam)
            if not os.path.isdir(clf_fam_dir):
                continue

            variant_map = {
                "Biased": os.path.join(clf_fam_dir, "biased"),
                "Fair": os.path.join(clf_fam_dir, "reweighing"),
                "BGMMOCE": os.path.join(clf_fam_dir, "BGMMOCE")
            }

            # If reweighing doesn't exist, check alternative fair mitigations
            if not os.path.exists(variant_map["Fair"]):
                for alt_fair in ["disparate_impact_remover", "postprocessing", "reductions", "inprocessing"]:
                    cand = os.path.join(clf_fam_dir, alt_fair)
                    if os.path.exists(cand):
                        variant_map["Fair"] = cand
                        break

            clf_records = {}

            for cat, v_dir in variant_map.items():
                if not os.path.isdir(v_dir):
                    continue

                # Option 1: Fast direct load from saved npy arrays
                preds_file = os.path.join(v_dir, "preds.npy")
                probs_file = os.path.join(v_dir, "probs.npy")
                y_file = os.path.join(v_dir, "y_test.npy")
                s_file = os.path.join(v_dir, "s_test.npy")

                if os.path.exists(preds_file) and os.path.exists(y_file) and os.path.exists(s_file):
                    preds = np.load(preds_file, allow_pickle=True)
                    y_true = np.load(y_file, allow_pickle=True)
                    biases_arr = np.load(s_file, allow_pickle=True)

                    auc_val = None
                    if os.path.exists(probs_file):
                        try:
                            probs = np.load(probs_file, allow_pickle=True)
                            if len(probs) == len(y_true):
                                if probs.ndim == 2 and probs.shape[1] == 2:
                                    auc_val = float(roc_auc_score(y_true, probs[:, 1]))
                                else:
                                    auc_val = float(roc_auc_score(y_true, probs))
                        except Exception:
                            auc_val = None

                    if auc_val is None:
                        # Fallback to metrics JSON
                        m_files = glob.glob(os.path.join(v_dir, "*metrics.json"))
                        if m_files:
                            try:
                                with open(m_files[0], "r") as mf:
                                    m_data = json.load(mf)
                                    auc_val = float(m_data.get("mean_auc", 0.0))
                            except Exception:
                                auc_val = None

                    clf_records[cat] = {
                        "preds": preds,
                        "y_true": y_true,
                        "biases": biases_arr,
                        "auc": auc_val
                    }
                else:
                    # Option 2: Fallback to model evaluation
                    pkl_files = glob.glob(os.path.join(v_dir, "*clf.pkl")) + glob.glob(os.path.join(v_dir, "model.pkl"))
                    feat_files = glob.glob(os.path.join(v_dir, "*selected_feats.npy")) + glob.glob(os.path.join(v_dir, "selected_feats.npy"))
                    if pkl_files and feat_files:
                        try:
                            split_type = "BGMMOCE" if cat == "BGMMOCE" else "biased"
                            if split_type == "BGMMOCE":
                                if cached_ts_bgmmoce is None:
                                    cached_ts_bgmmoce = load_all_scaled_test_features(loader, bias, "BGMMOCE")
                                X_ts, y_ts, s_ts = cached_ts_bgmmoce
                            else:
                                if cached_ts_biased is None:
                                    cached_ts_biased = load_all_scaled_test_features(loader, bias, "biased")
                                X_ts, y_ts, s_ts = cached_ts_biased

                            with open(pkl_files[0], "rb") as f:
                                clf = pkl.load(f)
                            sel_cols = np.load(feat_files[0], allow_pickle=True)
                            avail = [c for c in sel_cols if c in X_ts.columns]
                            if len(avail) == len(sel_cols) and hasattr(clf, "predict"):
                                preds = clf.predict(X_ts[avail])
                                auc_val = None
                                if hasattr(clf, "predict_proba"):
                                    try:
                                        p_proba = clf.predict_proba(X_ts[avail])
                                        auc_val = float(roc_auc_score(y_ts, p_proba[:, 1]))
                                    except Exception:
                                        auc_val = None
                                if auc_val is None:
                                    try:
                                        auc_val = float(roc_auc_score(y_ts, preds))
                                    except Exception:
                                        auc_val = None

                                clf_records[cat] = {
                                    "preds": preds,
                                    "y_true": y_ts,
                                    "biases": s_ts,
                                    "auc": auc_val
                                }
                        except Exception as e:
                            print(f"    [!] Error during model fallback for {cat} in {v_dir}: {e}")

            if clf_records:
                bias_data[clf_fam] = clf_records

        if bias_data:
            # 1. Standard plots (All models)
            print(f"  Generating standard fairness plots (all {len(bias_data)} classifier families)...")
            plot_fairness_gap_reduction_per_bias(bias, bias_data, target_dir, is_best_performing=False)
            plot_selection_rate_disparity_per_bias(bias, bias_data, target_dir, is_best_performing=False)
            plot_error_rate_by_subgroup_per_bias(bias, bias_data, target_dir, is_best_performing=False)
            print(f"  [Done] Standard plots saved for {bias.upper()} in: {target_dir}")

            # 2. Filtered plots: Only triplets with at least one model > auc_threshold (default 0.70)
            bias_best_dir = os.path.join(best_dir, bias)
            os.makedirs(bias_best_dir, exist_ok=True)
            best_bias_data = filter_best_performing_triplets(bias_data, auc_threshold=args.auc_threshold)

            # Record summary across families
            for clf_fam, variants in bias_data.items():
                b_auc = variants.get("Biased", {}).get("auc")
                f_auc = variants.get("Fair", {}).get("auc")
                m_auc = variants.get("BGMMOCE", {}).get("auc")
                valid_aucs = [a for a in [b_auc, f_auc, m_auc] if a is not None]
                max_auc = max(valid_aucs) if valid_aucs else 0.0
                all_summary_records.append({
                    "Bias": bias,
                    "Classifier": clf_fam,
                    "Display Name": clean_display_name(clf_fam),
                    "Biased_AUC": round(b_auc, 3) if b_auc is not None else "N/A",
                    "Fair_AUC": round(f_auc, 3) if f_auc is not None else "N/A",
                    "BGMMOCE_AUC": round(m_auc, 3) if m_auc is not None else "N/A",
                    "Max_AUC": round(max_auc, 3),
                    f"Qualifies (>{args.auc_threshold:.2f})": (max_auc > args.auc_threshold)
                })

            if best_bias_data:
                qual_names = [
                    f"{clean_display_name(k)} (max={max([v['auc'] for v in best_bias_data[k].values() if v.get('auc') is not None]):.3f})"
                    for k in best_bias_data
                ]
                print(f"  [+] {len(best_bias_data)} qualifying triplets (AUC > {args.auc_threshold:.2f}) for {bias.upper()}: {', '.join(qual_names)}")
                plot_fairness_gap_reduction_per_bias(bias, best_bias_data, best_dir, is_best_performing=True)
                plot_selection_rate_disparity_per_bias(bias, best_bias_data, best_dir, is_best_performing=True)
                plot_error_rate_by_subgroup_per_bias(bias, best_bias_data, best_dir, is_best_performing=True)
                print(f"  [Done] Best-performing models plots saved for {bias.upper()} in: {best_dir}")
            else:
                print(f"  [!] No classifier triplets with AUC > {args.auc_threshold:.2f} found for bias: {bias.upper()} (Skipped best_performing_models plots for this bias)")
        else:
            print(f"[!] No valid model data found for bias: {bias}")

    # Save summary table in best_performing_models
    if all_summary_records:
        sum_df = pd.DataFrame(all_summary_records)
        csv_path = os.path.join(best_dir, "best_performing_models_summary.csv")
        sum_df.to_csv(csv_path, index=False)
        print(f"\n[Saved] Best-performing models summary table: {csv_path}")

    print("\n" + "=" * 80)
    print(f"[SUCCESS] Disparity and fairness visualizations complete in:")
    print(f"  - All models: {target_dir}")
    print(f"  - Best performing (AUC > {args.auc_threshold:.2f}): {best_dir}")
    print("=" * 80)

if __name__ == '__main__':
    main()
