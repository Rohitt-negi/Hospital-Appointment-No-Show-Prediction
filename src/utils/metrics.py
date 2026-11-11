"""
Evaluation Metrics and Business Cost Utilities
Author: Rohit Singh Negi (Reg No: 12311143)
Project: Hospital Appointment No-Show Prediction

Provides comprehensive evaluation routines:
- Standard classification metrics (Precision, Recall, F1, PR-AUC, ROC-AUC)
- Healthcare operational cost function (FP = staff reminder cost vs FN = empty doctor slot cost)
- Threshold optimization curves
- Expected Calibration Error (ECE) and Brier score
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss
)
from sklearn.calibration import calibration_curve


# Operational healthcare unit costs (in USD / operational currency equivalent)
COST_FALSE_POSITIVE = 15.0   # Operational overhead: staff manual call / overbook buffer friction
COST_FALSE_NEGATIVE = 180.0  # Provider lost opportunity: empty clinic slot / idle clinical staff


def calculate_business_cost(y_true, y_pred, c_fp=COST_FALSE_POSITIVE, c_fn=COST_FALSE_NEGATIVE):
    """
    Computes total operational cost of errors.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = (fp * c_fp) + (fn * c_fn)
    cost_per_appointment = total_cost / len(y_true)
    return {
        'total_cost': float(total_cost),
        'cost_per_appointment': float(cost_per_appointment),
        'fp_cost': float(fp * c_fp),
        'fn_cost': float(fn * c_fn),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }


def compute_comprehensive_metrics(y_true, y_proba, threshold=0.50, c_fp=COST_FALSE_POSITIVE, c_fn=COST_FALSE_NEGATIVE):
    """
    Calculates statistical, discrimination, calibration, and business cost metrics.
    """
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calibration error (ECE with 10 bins)
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy='uniform')
    brier = brier_score_loss(y_true, y_proba)
    
    # Cost analysis
    cost_stats = calculate_business_cost(y_true, y_pred, c_fp=c_fp, c_fn=c_fn)
    
    metrics = {
        'threshold': float(threshold),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, y_proba)),
        'pr_auc': float(average_precision_score(y_true, y_proba)),
        'brier_score': float(brier),
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        },
        'business_cost': cost_stats
    }
    return metrics


def sweep_thresholds(y_true, y_proba, thresholds=None, c_fp=COST_FALSE_POSITIVE, c_fn=COST_FALSE_NEGATIVE):
    """
    Evaluates metrics across a grid of classification thresholds to determine
    the optimal operational operating point.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 37)
        
    records = []
    for t in thresholds:
        m = compute_comprehensive_metrics(y_true, y_proba, threshold=t, c_fp=c_fp, c_fn=c_fn)
        records.append({
            'threshold': t,
            'precision': m['precision'],
            'recall': m['recall'],
            'f1': m['f1'],
            'total_cost': m['business_cost']['total_cost'],
            'cost_per_appointment': m['business_cost']['cost_per_appointment'],
            'fp': m['confusion_matrix']['fp'],
            'fn': m['confusion_matrix']['fn'],
            'tp': m['confusion_matrix']['tp'],
            'tn': m['confusion_matrix']['tn']
        })
        
    df_sweep = pd.DataFrame(records)
    best_cost_row = df_sweep.loc[df_sweep['total_cost'].idxmin()]
    best_f1_row = df_sweep.loc[df_sweep['f1'].idxmax()]
    
    return {
        'sweep_table': df_sweep,
        'optimal_cost_threshold': float(best_cost_row['threshold']),
        'min_cost_per_appointment': float(best_cost_row['cost_per_appointment']),
        'optimal_f1_threshold': float(best_f1_row['threshold']),
        'max_f1': float(best_f1_row['f1'])
    }
