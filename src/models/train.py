"""
Model Training and Experimentation Pipeline
Author: Rohit Singh Negi (Reg No: 12311143)
Project: Hospital Appointment No-Show Prediction

Trains baseline and candidate classifiers with class imbalance handling,
evaluates on held-out test set, tunes operational threshold, and serializes artifacts.
"""

import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV

from src.data.make_dataset import prepare_dataset
from src.features.build_features import get_preprocessor
from src.utils.metrics import (
    compute_comprehensive_metrics,
    sweep_thresholds,
    COST_FALSE_POSITIVE,
    COST_FALSE_NEGATIVE
)


def load_data(processed_dir='data/processed'):
    train_path = os.path.join(processed_dir, 'train.csv')
    val_path = os.path.join(processed_dir, 'val.csv')
    test_path = os.path.join(processed_dir, 'test.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Processed splits not found. Running dataset preparation pipeline...")
        prepare_dataset(processed_dir=processed_dir)
        
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, val_df, test_df


def train_and_evaluate(processed_dir='data/processed', models_dir='models'):
    print("\n" + "="*60)
    print("HOSPITAL APPOINTMENT NO-SHOW PREDICTION - TRAINING PIPELINE")
    print("Author: Rohit Singh Negi (Reg No: 12311143)")
    print("="*60 + "\n")
    
    train_df, val_df, test_df = load_data(processed_dir)
    
    target_col = 'no_show'
    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values
    
    print(f"Dataset summary:")
    print(f"  Train records: {len(train_df):5d} (No-shows: {y_train.sum()} / {y_train.mean()*100:.2f}%)")
    print(f"  Val records  : {len(val_df):5d} (No-shows: {y_val.sum()} / {y_val.mean()*100:.2f}%)")
    print(f"  Test records : {len(test_df):5d} (No-shows: {y_test.sum()} / {y_test.mean()*100:.2f}%)")
    
    # Preprocessor
    preprocessor, num_cols, cat_cols = get_preprocessor()
    print(f"\nFeature pipeline: {len(num_cols)} numeric + {len(cat_cols)} categorical inputs.")
    
    # Class weighting ratio for handling class imbalance
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    imbalance_ratio = neg_count / max(1, pos_count)
    print(f"Class imbalance ratio (Negative/Positive): {imbalance_ratio:.2f}:1")
    
    # Define model suite
    models = {
        'Dummy_MostFrequent': DummyClassifier(strategy='most_frequent'),
        'Baseline_LogisticRegression': LogisticRegression(
            penalty='l2', C=1.0, max_iter=1000, random_state=42
        ),
        'Balanced_LogisticRegression': LogisticRegression(
            penalty='l2', C=0.5, class_weight='balanced', max_iter=1000, random_state=42
        ),
        'RandomForest_Balanced': RandomForestClassifier(
            n_estimators=150, max_depth=10, min_samples_split=8,
            class_weight='balanced_subsample', random_state=42, n_jobs=-1
        ),
        'HistGradientBoosting': HistGradientBoostingClassifier(
            max_iter=120, max_depth=6, learning_rate=0.08,
            class_weight='balanced', random_state=42
        )
    }
    
    results = {}
    fitted_pipelines = {}
    
    print("\n" + "-"*60)
    print("EXPERIMENTATION: TRAINING & VALIDATION PERFORMANCE")
    print("-"*60)
    
    for name, clf in models.items():
        print(f"\nTraining [{name}]...")
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        pipe.fit(train_df, y_train)
        fitted_pipelines[name] = pipe
        
        # Validation probabilities
        if hasattr(pipe, 'predict_proba'):
            val_proba = pipe.predict_proba(val_df)[:, 1]
        else:
            val_proba = np.full(len(val_df), y_train.mean())
            
        # Default 0.50 threshold metrics
        val_metrics_default = compute_comprehensive_metrics(y_val, val_proba, threshold=0.50)
        
        # Sweep threshold to find cost-optimal operating point on validation set
        sweep_res = sweep_thresholds(y_val, val_proba)
        opt_thresh = sweep_res['optimal_cost_threshold']
        val_metrics_opt = compute_comprehensive_metrics(y_val, val_proba, threshold=opt_thresh)
        
        results[name] = {
            'pipeline': pipe,
            'val_proba': val_proba,
            'val_metrics_default': val_metrics_default,
            'optimal_threshold': opt_thresh,
            'val_metrics_opt': val_metrics_opt,
            'sweep_data': sweep_res
        }
        
        print(f"  Val ROC-AUC : {val_metrics_default['roc_auc']:.4f} | PR-AUC: {val_metrics_default['pr_auc']:.4f}")
        print(f"  @ Thresh 0.50 -> Prec: {val_metrics_default['precision']:.3f} | Rec: {val_metrics_default['recall']:.3f} | F1: {val_metrics_default['f1']:.3f} | Cost/appt: ${val_metrics_default['business_cost']['cost_per_appointment']:.2f}")
        print(f"  @ Opt Thresh ({opt_thresh:.2f}) -> Prec: {val_metrics_opt['precision']:.3f} | Rec: {val_metrics_opt['recall']:.3f} | F1: {val_metrics_opt['f1']:.3f} | Cost/appt: ${val_metrics_opt['business_cost']['cost_per_appointment']:.2f}")

    # Select Best Model based on Validation PR-AUC and Operational Cost
    candidate_names = ['Balanced_LogisticRegression', 'RandomForest_Balanced', 'HistGradientBoosting']
    best_candidate_name = min(candidate_names, key=lambda n: results[n]['val_metrics_opt']['business_cost']['cost_per_appointment'])
    print(f"\n>>> Best Performing Candidate Model: [{best_candidate_name}]")
    
    best_pipeline = results[best_candidate_name]['pipeline']
    optimal_threshold = results[best_candidate_name]['optimal_threshold']
    
    # Final Evaluation on Held-Out Test Set (strictly unseen)
    print("\n" + "="*60)
    print(f"FINAL TEST SET EVALUATION (Held-out Test, N={len(test_df)})")
    print("="*60)
    
    test_evaluations = {}
    for name, p in fitted_pipelines.items():
        if hasattr(p, 'predict_proba'):
            test_proba = p.predict_proba(test_df)[:, 1]
        else:
            test_proba = np.full(len(test_df), y_train.mean())
            
        m_def = compute_comprehensive_metrics(y_test, test_proba, threshold=0.50)
        m_opt = compute_comprehensive_metrics(y_test, test_proba, threshold=optimal_threshold)
        
        test_evaluations[name] = {
            'test_proba': test_proba,
            'metrics_default': m_def,
            'metrics_optimal': m_opt
        }
        
        print(f"\nModel: {name}")
        print(f"  ROC-AUC: {m_def['roc_auc']:.4f} | PR-AUC: {m_def['pr_auc']:.4f} | Brier: {m_def['brier_score']:.4f}")
        print(f"  [Threshold = 0.50]  Precision: {m_def['precision']:.3f} | Recall: {m_def['recall']:.3f} | F1: {m_def['f1']:.3f} | Cost/appt: ${m_def['business_cost']['cost_per_appointment']:.2f}")
        print(f"  [Threshold = {optimal_threshold:.2f}]  Precision: {m_opt['precision']:.3f} | Recall: {m_opt['recall']:.3f} | F1: {m_opt['f1']:.3f} | Cost/appt: ${m_opt['business_cost']['cost_per_appointment']:.2f}")
        print(f"  Confusion Matrix (@{optimal_threshold:.2f}): TN={m_opt['confusion_matrix']['tn']}, FP={m_opt['confusion_matrix']['fp']}, FN={m_opt['confusion_matrix']['fn']}, TP={m_opt['confusion_matrix']['tp']}")

    # Save artifacts
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, 'best_model.joblib')
    baseline_model_path = os.path.join(models_dir, 'baseline_model.joblib')
    metadata_path = os.path.join(models_dir, 'model_metadata.json')
    
    joblib.dump(best_pipeline, best_model_path)
    joblib.dump(fitted_pipelines['Baseline_LogisticRegression'], baseline_model_path)
    
    # Save preprocessor separately for modular testing
    joblib.dump(best_pipeline.named_steps['preprocessor'], os.path.join(models_dir, 'preprocessor.joblib'))
    
    # Get feature names after one-hot encoding
    prep = best_pipeline.named_steps['preprocessor']
    feature_names = list(prep.get_feature_names_out())
    
    # Feature importance extraction
    classifier_step = best_pipeline.named_steps['classifier']
    feature_importances = {}
    if hasattr(classifier_step, 'feature_importances_'):
        imp = classifier_step.feature_importances_
        sorted_idx = np.argsort(imp)[::-1]
        for idx in sorted_idx[:15]:
            clean_name = feature_names[idx].replace('num__', '').replace('cat__', '')
            feature_importances[clean_name] = float(imp[idx])
    elif hasattr(classifier_step, 'coef_'):
        coef = np.abs(classifier_step.coef_[0])
        sorted_idx = np.argsort(coef)[::-1]
        for idx in sorted_idx[:15]:
            clean_name = feature_names[idx].replace('num__', '').replace('cat__', '')
            feature_importances[clean_name] = float(coef[idx])
            
    best_test_opt = test_evaluations[best_candidate_name]['metrics_optimal']
    baseline_test_opt = test_evaluations['Baseline_LogisticRegression']['metrics_default']
    
    metadata = {
        'model_name': best_candidate_name,
        'candidate_models_evaluated': list(models.keys()),
        'trained_by': 'Rohit Singh Negi',
        'registration_number': '12311143',
        'operational_decision_threshold': float(optimal_threshold),
        'cost_parameters': {
            'cost_fp_usd': COST_FALSE_POSITIVE,
            'cost_fn_usd': COST_FALSE_NEGATIVE
        },
        'test_metrics_at_optimal_threshold': {
            'roc_auc': float(best_test_opt['roc_auc']),
            'pr_auc': float(best_test_opt['pr_auc']),
            'precision': float(best_test_opt['precision']),
            'recall': float(best_test_opt['recall']),
            'f1': float(best_test_opt['f1']),
            'brier_score': float(best_test_opt['brier_score']),
            'cost_per_appointment': float(best_test_opt['business_cost']['cost_per_appointment']),
            'confusion_matrix': best_test_opt['confusion_matrix']
        },
        'baseline_comparison': {
            'baseline_model': 'Baseline_LogisticRegression',
            'baseline_cost_per_appointment': float(baseline_test_opt['business_cost']['cost_per_appointment']),
            'cost_reduction_percent': float(
                (baseline_test_opt['business_cost']['cost_per_appointment'] - best_test_opt['business_cost']['cost_per_appointment'])
                / baseline_test_opt['business_cost']['cost_per_appointment'] * 100
            )
        },
        'top_feature_importances': feature_importances,
        'feature_names': feature_names
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"\nArtifacts successfully serialized to: {models_dir}/")
    print(f"  - Best Model : {best_model_path}")
    print(f"  - Baseline   : {baseline_model_path}")
    print(f"  - Metadata   : {metadata_path}")
    print("="*60 + "\n")
    
    return best_pipeline, test_evaluations, metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train hospital appointment no-show prediction models.")
    parser.add_argument('--processed_dir', type=str, default='data/processed', help="Directory containing processed CSVs.")
    parser.add_argument('--models_dir', type=str, default='models', help="Directory to save model artifacts.")
    args = parser.parse_args()
    
    train_and_evaluate(processed_dir=args.processed_dir, models_dir=args.models_dir)

