#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUROC Analysis & Global Results Aggregator for Lung Cancer 2-Year Survival Fairness Benchmark

Reads from the structured results hierarchy:
    chkp/<bias>/<classifier>/<variant>/

Generates:
1. AUROC_comparison.png inside each classifier folder (comparing Biased, BGMMOCE, Reweighing, DIR, Reductions, Postprocessing)
2. metrics_comparison.csv inside each classifier folder
3. Comprehensive benchmark_summary_global.csv in chkp/ taking into account ALL classifiers and ALL mitigation methods.

@author: Eleftherios Trivizakis & Google DeepMind Antigravity
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Style dictionary for comparable variants
VARIANT_STYLES = {
    "biased":                    {"color": "#d62728", "style": "-",  "label": "Biased (Baseline)"},
    "BGMMOCE":                   {"color": "#1f77b4", "style": "--", "label": "BGMMOCE"},
    "reweighing":                {"color": "#2ca02c", "style": "-",  "label": "Fair: Reweighing"},
    "disparate_impact_remover":  {"color": "#9467bd", "style": "-.", "label": "Fair: Disparate Impact Remover"},
    "reductions":                {"color": "#ff7f0e", "style": ":",  "label": "Fair: Reduction (EO/DP)"},
    "inprocessing":              {"color": "#17becf", "style": "-",  "label": "Fair: Prejudice Remover"},
    "postprocessing":            {"color": "#bcbd22", "style": "--", "label": "Fair: Post-Processing"}
}

def clean_display_name(name):
    special = {
        "svm_rbf": "SVM (RBF)",
        "svm_linear": "SVM (Linear)",
        "svm_poly": "SVM (Poly)",
        "svm_sigmoid": "SVM (Sigmoid)",
        "svm": "SVM",
        "mlp16": "MLP (16 Neurons)",
        "mlp32": "MLP (32 Neurons)",
        "mlp64": "MLP (64 Neurons)",
        "mlp128": "MLP (128 Neurons)",
        "mlp": "MLP",
        "random_forest": "Random Forest",
        "logistic_regression": "Logistic Regression",
        "gradient_boosting": "Gradient Boosting",
        "adaboost": "AdaBoost",
        "decision_tree": "Decision Tree",
        "naive_bayes": "Naive Bayes",
        "ridge": "Ridge",
        "sgd": "SGD"
    }
    if name.lower() in special:
        return special[name.lower()]
    return str(name).replace("_", " ").title()

def process_classifier_folder(bias_name, clf_name, clf_dir, global_rows):
    """
    Processes all variants inside a specific classifier folder:
    chkp/<bias>/<classifier>/<variant>/
    Generates AUROC_comparison.png and metrics_comparison.csv.
    """
    variant_subdirs = [
        d for d in os.listdir(clf_dir)
        if os.path.isdir(os.path.join(clf_dir, d)) and not d.startswith(".")
    ]

    if not variant_subdirs:
        return

    clf_metrics = []
    # Collect best curve candidates per variant for clean, publication-grade comparative AUROC plot
    variant_best_curves = {}

    # Define standard variant display order
    variant_order = [
        "biased", "BGMMOCE", "reweighing", "disparate_impact_remover",
        "reductions", "inprocessing", "postprocessing"
    ]

    for variant in sorted(variant_subdirs):
        var_path = os.path.join(clf_dir, variant)
        json_files = glob.glob(os.path.join(var_path, "*_metrics.json"))
        if not json_files:
            continue

        for jf in sorted(json_files):
            model_key = os.path.basename(jf).replace("_metrics.json", "")
            try:
                with open(jf, "r") as f:
                    mdata = json.load(f)
            except Exception as e:
                print(f"    [!] Error reading {jf}: {e}")
                continue

            folds = mdata.get("folds", [])
            auc = mdata.get("mean_auc", 0.0)
            c_index = mdata.get("c_index", auc)
            brier = mdata.get("brier_score", np.nan)
            ipa = mdata.get("ipa", np.nan)
            ece = mdata.get("ece", np.nan)
            acc = mdata.get("accuracy", 0.0)
            f1 = mdata.get("f1", 0.0)
            prec = mdata.get("precision", 0.0)
            rec = mdata.get("recall", 0.0)
            fairness = mdata.get("fairness", {})
            spd = fairness.get("statistical_parity_difference", np.nan)
            di = fairness.get("disparate_impact", np.nan)
            eod = fairness.get("equal_opportunity_difference", np.nan)
            aod = fairness.get("average_odds_difference", np.nan)
            c_disp = mdata.get("survival", {}).get("c_index_disparity", np.nan)

            training_split = "BGMMOCE" if "bgmmoce" in variant.lower() else "biased"

            row = {
                "Bias": bias_name.capitalize(),
                "Classifier_Family": clf_name,
                "Model_Name": model_key,
                "Variant": variant,
                "Training_Split": training_split,
                "AUC": round(auc, 4) if not np.isnan(auc) else None,
                "C_Index": round(c_index, 4) if not np.isnan(c_index) else None,
                "Brier_Score": round(brier, 4) if not np.isnan(brier) else None,
                "IPA": round(ipa, 4) if not np.isnan(ipa) else None,
                "ECE": round(ece, 4) if not np.isnan(ece) else None,
                "Accuracy": round(acc, 4) if not np.isnan(acc) else None,
                "F1": round(f1, 4) if not np.isnan(f1) else None,
                "Precision": round(prec, 4) if not np.isnan(prec) else None,
                "Recall": round(rec, 4) if not np.isnan(rec) else None,
                "Statistical_Parity_Diff": round(spd, 4) if not np.isnan(spd) else None,
                "Disparate_Impact": round(di, 4) if not np.isnan(di) else None,
                "Equal_Opportunity_Diff": round(eod, 4) if not np.isnan(eod) else None,
                "Average_Odds_Diff": round(aod, 4) if not np.isnan(aod) else None,
                "C_Index_Disparity": round(c_disp, 4) if not np.isnan(c_disp) else None
            }

            clf_metrics.append(row)
            global_rows.append(row)

            # Check if this model is candidate for best ROC curve for this variant
            if folds and "fpr" in folds[0] and "tpr" in folds[0]:
                curr_best = variant_best_curves.get(variant)
                if curr_best is None or auc > curr_best["auc"]:
                    variant_best_curves[variant] = {
                        "model_key": model_key,
                        "auc": auc,
                        "fpr": np.array(folds[0]["fpr"]),
                        "tpr": np.array(folds[0]["tpr"])
                    }

    # Generate Comparative AUROC Figure
    if variant_best_curves:
        plt.figure(figsize=(8, 7))
        # Sort variants according to standard comparison hierarchy
        sorted_variants = sorted(
            variant_best_curves.keys(),
            key=lambda v: variant_order.index(v) if v in variant_order else 99
        )

        for variant in sorted_variants:
            cdata = variant_best_curves[variant]
            style_info = VARIANT_STYLES.get(variant, {"color": None, "style": "-", "label": clean_display_name(variant)})
            base_label = style_info.get("label", variant)
            
            # If multiple models could exist in this variant, show model name for transparency
            if cdata["model_key"] != variant and cdata["model_key"] not in ["random_forest", "mlp16", "mlp32", "mlp64", "mlp", "adaboost", "gradient_boosting"]:
                label = f"{base_label} [{cdata['model_key']}] (AUC={cdata['auc']:.2f})"
            else:
                label = f"{base_label} (AUC={cdata['auc']:.2f})"

            lw = 2.5 if variant in ["biased", "BGMMOCE"] else 1.8
            plt.plot(
                cdata["fpr"], cdata["tpr"], label=label,
                color=style_info.get("color"),
                linestyle=style_info.get("style", "-"),
                linewidth=lw
            )

        readable_clf = clean_display_name(clf_name)
        plt.plot([0, 1], [0, 1], 'k:', alpha=0.4, label='Random Chance')
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11)
        plt.title(f'AUROC Comparison: {readable_clf} | Bias: {bias_name.capitalize()}', fontsize=12)
        plt.legend(loc='lower right', fontsize=9)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()

        plot_path = os.path.join(clf_dir, "AUROC_comparison.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"  -> Generated AUROC: {plot_path}")

    if clf_metrics:
        metrics_df = pd.DataFrame(clf_metrics)
        metrics_csv = os.path.join(clf_dir, "metrics_comparison.csv")
        metrics_df.to_csv(metrics_csv, index=False)
        print(f"  -> Generated Metrics CSV: {metrics_csv}")

def aggregate_and_generate_reports(base_dir="chkp"):
    """
    Main aggregator: crawls <base_dir>/<bias>/<classifier>/<variant>/
    Generates comparison plots and writes split benchmark_summary_<bias>.csv files.
    """
    print("=" * 80)
    print("RUNNING RESULTS AGGREGATION & AUROC COMPARISONS")
    print(f"Target Directory: {base_dir}")
    print("=" * 80)

    global_rows = []
    known_biases = ["affiliation", "gender", "ethnicity"]

    # 1. Discover structured bias folders in base_dir
    found_biases = []
    for item in os.listdir(base_dir):
        bias_path = os.path.join(base_dir, item)
        if os.path.isdir(bias_path) and item.lower() in known_biases:
            found_biases.append(item.lower())

    for bias_name in sorted(found_biases):
        bias_dir = os.path.join(base_dir, bias_name)
        print(f"\nProcessing Bias: {bias_name.upper()} ({bias_dir})")

        # Discover classifier folders
        clf_folders = [
            d for d in os.listdir(bias_dir)
            if os.path.isdir(os.path.join(bias_dir, d)) and not d.startswith(".")
        ]

        for clf_name in sorted(clf_folders):
            clf_dir = os.path.join(bias_dir, clf_name)
            process_classifier_folder(bias_name, clf_name, clf_dir, global_rows)

        # Split benchmark summary: save one dedicated CSV per bias
        bias_rows = [r for r in global_rows if r["Bias"].lower() == bias_name]
        if bias_rows:
            bias_df = pd.DataFrame(bias_rows)
            bias_df = bias_df.drop_duplicates(subset=["Bias", "Classifier_Family", "Model_Name", "Variant", "Training_Split"])
            # Save inside base experiment directory
            split_csv_path = os.path.join(base_dir, f"benchmark_summary_{bias_name}.csv")
            bias_df.to_csv(split_csv_path, index=False)
            # Also save inside bias subfolder
            bias_summary_path = os.path.join(bias_dir, f"benchmark_summary_{bias_name}.csv")
            bias_df.to_csv(bias_summary_path, index=False)
            print(f"[Bias Summary] Saved {split_csv_path} ({len(bias_df)} models)")

    # 2. Write Master Global Summary as well
    if global_rows:
        global_df = pd.DataFrame(global_rows)
        global_df = global_df.drop_duplicates(subset=["Bias", "Classifier_Family", "Model_Name", "Variant", "Training_Split"])
        global_csv = os.path.join(base_dir, "benchmark_summary_global.csv")
        global_df.to_csv(global_csv, index=False)

        print("\n" + "=" * 80)
        print(f"[SUCCESS] Global Benchmark Summary Saved: {global_csv}")
        print(f"Total Model Evaluations Recorded: {len(global_df)}")
        print(f"Distinct Biases:      {global_df['Bias'].unique().tolist()}")
        print(f"Distinct Classifiers: {len(global_df['Classifier_Family'].unique())} families")
        print(f"Distinct Variants:    {global_df['Variant'].unique().tolist()}")
        print("=" * 80)
        print(global_df.groupby(["Bias", "Variant"])["AUC"].agg(["count", "mean"]).to_string())
    else:
        print("\n[!] No structured classifier results found. Run analysisv3.py to generate models.")

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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AUROC Analysis & Global Results Aggregator")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Specific experiment folder under chkp/, e.g. exp_18.10_23h59m")
    parser.add_argument("--base-dir", type=str, default=None,
                        help="Base directory containing bias folders")
    args = parser.parse_args()

    target_dir = args.base_dir if args.base_dir else find_target_dir("chkp", args.exp_name)
    aggregate_and_generate_reports(target_dir)