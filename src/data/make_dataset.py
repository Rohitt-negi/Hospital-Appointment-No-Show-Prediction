"""
Data Preparation and Chronological Split Pipeline
Author: Rohit Singh Negi (Reg No: 12311143)
Project: Hospital Appointment No-Show Prediction

Reads raw appointment dataset, performs quality validation, computes leakage-safe
features, and executes a chronological train/val/test split.
"""

import os
import sys
import argparse
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.generate_dataset import generate_synthetic_appointments
from src.features.build_features import compute_historical_features, engineer_appointment_features


def prepare_dataset(raw_path='data/raw/hospital_appointments.csv', processed_dir='data/processed'):
    """
    Loads, cleans, engineers features, and chronologically splits hospital appointment data.
    """
    if not os.path.exists(raw_path):
        print(f"Raw data not found at {raw_path}. Generating realistic dataset...")
        generate_synthetic_appointments(num_records=15000, seed=42, output_path=raw_path)
        
    print(f"Loading raw appointment records from: {raw_path}")
    df = pd.read_csv(raw_path)
    print(f"Raw record count: {len(df)}")
    
    # Data Quality Sanity Checks
    print("Performing Data Quality Sanity Checks...")
    initial_len = len(df)
    
    # 1. Filter out impossible ages (< 0 or > 115)
    df = df[(df['age'] >= 0) & (df['age'] <= 115)]
    
    # 2. Filter out negative lead time (scheduled after appointment)
    df = df[df['lead_days'] >= 0]
    
    # 3. Ensure no missing values in critical columns
    critical_cols = ['appointment_id', 'patient_id', 'scheduled_day', 'appointment_day', 'no_show']
    df = df.dropna(subset=critical_cols)
    
    dropped_count = initial_len - len(df)
    print(f"Sanity checks complete. Dropped records: {dropped_count} (Remaining: {len(df)})")
    
    # Chronological Ordering by Appointment Date & Scheduled Time
    print("Sorting chronologically to prevent temporal data leakage...")
    df['temp_appt_dt'] = pd.to_datetime(df['appointment_day'])
    df['temp_sched_dt'] = pd.to_datetime(df['scheduled_day'])
    df = df.sort_values(by=['temp_appt_dt', 'temp_sched_dt']).reset_index(drop=True)
    df = df.drop(columns=['temp_appt_dt', 'temp_sched_dt'])
    
    # Compute historical features sequentially (strictly past visits)
    print("Computing patient historical attendance metrics strictly from prior appointments...")
    df = compute_historical_features(df)
    
    # Feature engineering for operational prediction
    print("Extracting booking-time operational features...")
    df = engineer_appointment_features(df)
    
    # Chronological Split (70% Train, 15% Validation, 15% Test)
    total_records = len(df)
    train_end = int(total_records * 0.70)
    val_end = int(total_records * 0.85)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    os.makedirs(processed_dir, exist_ok=True)
    
    train_path = os.path.join(processed_dir, 'train.csv')
    val_path = os.path.join(processed_dir, 'val.csv')
    test_path = os.path.join(processed_dir, 'test.csv')
    full_path = os.path.join(processed_dir, 'full_features.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    df.to_csv(full_path, index=False)
    
    print("\n" + "="*50)
    print("DATASET PREPARATION COMPLETE")
    print("="*50)
    print(f"Total Processed Records: {len(df)}")
    print(f"Train Set : {len(train_df):5d} records ({len(train_df)/len(df)*100:.1f}%) | "
          f"Dates: {train_df['appointment_day'].min()} to {train_df['appointment_day'].max()} | "
          f"No-show rate: {train_df['no_show'].mean()*100:.2f}%")
    print(f"Val Set   : {len(val_df):5d} records ({len(val_df)/len(df)*100:.1f}%) | "
          f"Dates: {val_df['appointment_day'].min()} to {val_df['appointment_day'].max()} | "
          f"No-show rate: {val_df['no_show'].mean()*100:.2f}%")
    print(f"Test Set  : {len(test_df):5d} records ({len(test_df)/len(df)*100:.1f}%) | "
          f"Dates: {test_df['appointment_day'].min()} to {test_df['appointment_day'].max()} | "
          f"No-show rate: {test_df['no_show'].mean()*100:.2f}%")
    print(f"Saved to: {processed_dir}/ [train.csv, val.csv, test.csv, full_features.csv]")
    print("="*50 + "\n")
    
    return train_df, val_df, test_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare dataset and execute chronological split.")
    parser.add_argument('--raw', type=str, default='data/raw/hospital_appointments.csv', help="Path to raw CSV.")
    parser.add_argument('--outdir', type=str, default='data/processed', help="Path to processed output dir.")
    args = parser.parse_args()
    
    prepare_dataset(raw_path=args.raw, processed_dir=args.outdir)
