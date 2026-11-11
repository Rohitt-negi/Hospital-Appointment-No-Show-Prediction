"""
Comprehensive Evaluation and Visualization Suite
Author: Rohit Singh Negi (Reg No: 12311143)
Project: Hospital Appointment No-Show Prediction

Generates publication-quality evaluation figures:
1. Class distribution & appointment volume
2. No-show probability by lead time & age
3. ROC Curves & Precision-Recall Curves across all models
4. Confusion Matrix heatmap at optimal decision threshold
5. Calibration Reliability Diagram & Brier score
6. Operational Business Cost vs Threshold tradeoff curve
7. Global Feature Importances
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, brier_score_loss
from sklearn.calibration import calibration_curve
from src.utils.metrics import sweep_thresholds, COST_FALSE_POSITIVE, COST_FALSE_NEGATIVE

# Set visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, sans-serif'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def generate_evaluation_artifacts(data_dir='data/processed', models_dir='models', figures_dir='reports/figures'):
    os.makedirs(figures_dir, exist_ok=True)
    print("\n" + "="*60)
    print("GENERATING EVALUATION FIGURES AND ARTIFACTS")
    print("="*60)
    
    test_path = os.path.join(data_dir, 'test.csv')
    train_path = os.path.join(data_dir, 'train.csv')
    full_path = os.path.join(data_dir, 'full_features.csv')
    
    test_df = pd.read_csv(test_path)
    train_df = pd.read_csv(train_path)
    full_df = pd.read_csv(full_path) if os.path.exists(full_path) else pd.concat([train_df, test_df])
    
    y_test = test_df['no_show'].values
    
    # Load models
    best_pipe = joblib.load(os.path.join(models_dir, 'best_model.joblib'))
    baseline_pipe = joblib.load(os.path.join(models_dir, 'baseline_model.joblib'))
    
    with open(os.path.join(models_dir, 'model_metadata.json'), 'r') as f:
        metadata = json.load(f)
        
    optimal_threshold = metadata.get('operational_decision_threshold', 0.35)
    
    # Compute test probabilities
    y_proba_best = best_pipe.predict_proba(test_df)[:, 1]
    y_proba_baseline = baseline_pipe.predict_proba(test_df)[:, 1]

    # -------------------------------------------------------------
    # 1. Class Distribution
    # -------------------------------------------------------------
    print("Generating Figure 1: Class Distribution...")
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = full_df['no_show'].value_counts()
    labels = ['Show Up (0)', 'No-Show (1)']
    colors = ['#2b7bba', '#e74c3c']
    bars = ax.bar(labels, counts.values, color=colors, width=0.5, edgecolor='#333333', linewidth=0.5)
    ax.set_ylabel('Number of Appointments', fontsize=11, fontweight='bold')
    ax.set_title('Dataset Class Distribution (No-Show vs Show-Up)', fontsize=12, fontweight='bold', pad=12)
    for bar in bars:
        yval = bar.get_height()
        pct = yval / len(full_df) * 100
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 150, f'{yval:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylim(0, max(counts.values) * 1.18)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'class_distribution.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 2. Lead Time vs No-Show Rate
    # -------------------------------------------------------------
    print("Generating Figure 2: Lead Time vs No-Show Rate...")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    full_df['lead_bucket'] = pd.cut(
        full_df['lead_days'], 
        bins=[-1, 0, 3, 7, 14, 28, 60, 100], 
        labels=['Same Day (0)', '1-3 Days', '4-7 Days', '8-14 Days', '15-28 Days', '29-60 Days', '60+ Days']
    )
    lead_stats = full_df.groupby('lead_bucket', observed=False)['no_show'].agg(['count', 'mean']).reset_index()
    
    ax.bar(lead_stats['lead_bucket'].astype(str), lead_stats['mean'] * 100, color='#3498db', edgecolor='#2980b9', width=0.6)
    ax.set_ylabel('No-Show Rate (%)', fontsize=11, fontweight='bold', color='#1a5276')
    ax.set_xlabel('Booking Lead Time Window', fontsize=11, fontweight='bold')
    ax.set_title('Impact of Lead Time on Appointment No-Show Rate', fontsize=12, fontweight='bold', pad=12)
    ax.set_ylim(0, 45)
    for i, row in lead_stats.iterrows():
        ax.text(i, row['mean']*100 + 1.0, f"{row['mean']*100:.1f}%\n(n={row['count']})", ha='center', va='bottom', fontsize=8.5)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'lead_time_vs_noshow.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 3. Age Distribution by Attendance
    # -------------------------------------------------------------
    print("Generating Figure 3: Age Distribution by Attendance...")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.kdeplot(data=full_df[full_df['no_show'] == 0], x='age', label='Attended (0)', color='#27ae60', fill=True, alpha=0.35, ax=ax)
    sns.kdeplot(data=full_df[full_df['no_show'] == 1], x='age', label='No-Show (1)', color='#e74c3c', fill=True, alpha=0.35, ax=ax)
    ax.set_title('Patient Age Density Distribution by Attendance Status', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Patient Age (Years)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'age_distribution.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 4. ROC Curves
    # -------------------------------------------------------------
    print("Generating Figure 4: ROC Curves...")
    fig, ax = plt.subplots(figsize=(6.5, 5))
    fpr_b, tpr_b, _ = roc_curve(y_test, y_proba_best)
    fpr_base, tpr_base, _ = roc_curve(y_test, y_proba_baseline)
    
    from sklearn.metrics import roc_auc_score
    auc_best = roc_auc_score(y_test, y_proba_best)
    auc_base = roc_auc_score(y_test, y_proba_baseline)
    
    ax.plot(fpr_b, tpr_b, color='#2980b9', lw=2.5, label=f'Best Model ({metadata["model_name"]}) (AUC = {auc_best:.3f})')
    ax.plot(fpr_base, tpr_base, color='#7f8c8d', lw=2, linestyle='--', label=f'Baseline Logistic Reg (AUC = {auc_base:.3f})')
    ax.plot([0, 1], [0, 1], color='#bdc3c7', linestyle=':', label='Random Chance (AUC = 0.500)')
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Comparison', fontsize=12, fontweight='bold', pad=12)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'roc_curves.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 5. Precision-Recall Curves
    # -------------------------------------------------------------
    print("Generating Figure 5: Precision-Recall Curves...")
    fig, ax = plt.subplots(figsize=(6.5, 5))
    p_b, r_b, _ = precision_recall_curve(y_test, y_proba_best)
    p_base, r_base, _ = precision_recall_curve(y_test, y_proba_baseline)
    
    from sklearn.metrics import average_precision_score
    ap_b = average_precision_score(y_test, y_proba_best)
    ap_base = average_precision_score(y_test, y_proba_baseline)
    
    ax.plot(r_b, p_b, color='#8e44ad', lw=2.5, label=f'{metadata["model_name"]} (AP = {ap_b:.3f})')
    ax.plot(r_base, p_base, color='#7f8c8d', lw=2, linestyle='--', label=f'Baseline Logistic Reg (AP = {ap_base:.3f})')
    ax.axhline(y=y_test.mean(), color='#e74c3c', linestyle=':', label=f'No-Skill Baseline ({y_test.mean():.3f})')
    
    ax.set_xlabel('Recall', fontsize=11, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=11, fontweight='bold')
    ax.set_title('Precision-Recall Curves (Addressing Class Imbalance)', fontsize=12, fontweight='bold', pad=12)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'pr_curves.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 6. Confusion Matrix at Optimal Decision Threshold
    # -------------------------------------------------------------
    print("Generating Figure 6: Confusion Matrix...")
    fig, ax = plt.subplots(figsize=(6, 5))
    y_pred_opt = (y_proba_best >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_opt)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Pred: Attended', 'Pred: No-Show'],
                yticklabels=['Actual: Attended', 'Actual: No-Show'],
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title(f'Test Confusion Matrix at Decision Threshold $\\tau = {optimal_threshold:.2f}$', fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 7. Calibration Curve (Reliability Diagram)
    # -------------------------------------------------------------
    print("Generating Figure 7: Calibration Curve...")
    fig, ax = plt.subplots(figsize=(6.5, 5))
    prob_true, prob_pred = calibration_curve(y_test, y_proba_best, n_bins=8, strategy='uniform')
    brier = brier_score_loss(y_test, y_proba_best)
    
    ax.plot([0, 1], [0, 1], linestyle='--', color='#7f8c8d', label='Perfect Calibration')
    ax.plot(prob_pred, prob_true, marker='o', lw=2, color='#e67e22', label=f'Model Probabilities (Brier={brier:.4f})')
    ax.set_xlabel('Mean Predicted Probability', fontsize=11, fontweight='bold')
    ax.set_ylabel('Empirical Fraction of Positives', fontsize=11, fontweight='bold')
    ax.set_title('Probability Calibration Reliability Diagram', fontsize=12, fontweight='bold', pad=12)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'calibration_curve.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 8. Operational Cost vs Threshold Sweep Curve
    # -------------------------------------------------------------
    print("Generating Figure 8: Operational Cost vs Threshold Sweep Curve...")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sweep_df = sweep_thresholds(y_test, y_proba_best)['sweep_table']
    
    ax.plot(sweep_df['threshold'], sweep_df['cost_per_appointment'], color='#c0392b', lw=2.5, label='Operational Cost ($/Appt)')
    ax.axvline(x=optimal_threshold, color='#27ae60', linestyle='--', lw=2, label=f'Optimal Operational Threshold ($\\tau = {optimal_threshold:.2f}$)')
    ax.axvline(x=0.50, color='#95a5a6', linestyle=':', lw=1.5, label='Default Naive Threshold ($\\tau = 0.50$)')
    
    # Annotate minimum cost
    min_cost = sweep_df.loc[sweep_df['threshold'] == optimal_threshold, 'cost_per_appointment'].values
    if len(min_cost) > 0:
        ax.scatter([optimal_threshold], [min_cost[0]], color='#27ae60', s=80, zorder=5)
        ax.annotate(f"Min Cost: ${min_cost[0]:.2f}",
                    xy=(optimal_threshold, min_cost[0]),
                    xytext=(optimal_threshold + 0.08, min_cost[0] + 6),
                    arrowprops=dict(facecolor='#27ae60', shrink=0.08, width=1.5, headwidth=6),
                    fontsize=10, fontweight='bold')
        
    ax.set_xlabel('Classification Decision Threshold ($\\tau$)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Expected Cost per Appointment ($)', fontsize=11, fontweight='bold')
    ax.set_title('Operational Financial Cost Curve ($C_{FP}=\\$15, C_{FN}=\\$180$)', fontsize=12, fontweight='bold', pad=12)
    ax.legend(loc="upper center", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'cost_threshold_curve.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 9. Feature Importance Bar Chart
    # -------------------------------------------------------------
    print("Generating Figure 9: Feature Importance Bar Chart...")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    top_feats = metadata.get('top_feature_importances', {})
    if top_feats:
        feat_df = pd.DataFrame(list(top_feats.items()), columns=['Feature', 'Importance']).head(10)
        feat_df = feat_df.sort_values(by='Importance', ascending=True)
        ax.barh(feat_df['Feature'], feat_df['Importance'], color='#16a085', edgecolor='#0e6655')
        ax.set_xlabel('Relative Importance (Standardized Absolute Weight / Gini)', fontsize=11, fontweight='bold')
        ax.set_title('Top 10 Most Influential Operational Predictors of No-Show', fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'feature_importance.png'), dpi=300)
    plt.close()

    print(f"\nAll 9 evaluation figures generated successfully in: {figures_dir}/")
    print("="*60 + "\n")


if __name__ == '__main__':
    generate_evaluation_artifacts()

