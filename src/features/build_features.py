"""
Feature Engineering and Preprocessing Pipeline
Author: Rohit Singh Negi (Reg No: 12311143)
Project: Hospital Appointment No-Show Prediction

This module handles:
1. Leakage-free historical patient feature calculation (cumulative past appointments/no-shows strictly prior to current appointment date).
2. Temporal feature extraction (lead days, log lead days, appointment day-of-week, booking day-of-week, same-day flag).
3. Demographic and administrative binning (age groups, risk categories).
4. Scikit-learn preprocessing pipeline for categorical encoding and numeric scaling.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def compute_historical_features(df):
    """
    Computes prior appointment attendance metrics for each patient strictly using
    prior events to prevent future leakage. Assumes df is sorted chronologically.
    
    Returns df with added features:
    - prior_appointments: int
    - prior_noshows: int
    - prior_noshow_rate: float
    - is_new_patient: int
    """
    df = df.copy()
    
    # Ensure chronological order
    if 'appointment_day' in df.columns:
        df['temp_appt_dt'] = pd.to_datetime(df['appointment_day'])
        df = df.sort_values(by=['temp_appt_dt', 'scheduled_day']).reset_index(drop=True)
        df = df.drop(columns=['temp_appt_dt'])

    # Track historical counts per patient
    prior_appts = []
    prior_noshow_list = []
    prior_rate_list = []
    is_new = []
    
    # Dictionary tracking {patient_id: [total_visits, total_noshows]}
    history = {}
    
    for _, row in df.iterrows():
        p_id = row['patient_id']
        if p_id not in history:
            prior_appts.append(0)
            prior_noshow_list.append(0)
            prior_rate_list.append(0.0)
            is_new.append(1)
            # Initialize history
            if 'no_show' in row:
                history[p_id] = [1, int(row['no_show'])]
            else:
                history[p_id] = [1, 0]
        else:
            prev_total, prev_ns = history[p_id]
            prior_appts.append(prev_total)
            prior_noshow_list.append(prev_ns)
            prior_rate_list.append(prev_ns / prev_total if prev_total > 0 else 0.0)
            is_new.append(0)
            
            # Update history with this appointment outcome if present
            if 'no_show' in row:
                history[p_id] = [prev_total + 1, prev_ns + int(row['no_show'])]
            else:
                history[p_id] = [prev_total + 1, prev_ns]
                
    df['prior_appointments'] = prior_appts
    df['prior_noshows'] = prior_noshow_list
    df['prior_noshow_rate'] = prior_rate_list
    df['is_new_patient'] = is_new
    return df


def engineer_appointment_features(df):
    """
    Extracts operational and temporal features known strictly at booking time.
    """
    df = df.copy()
    
    # Parse dates
    sched_dt = pd.to_datetime(df['scheduled_day'])
    appt_dt = pd.to_datetime(df['appointment_day'])
    
    # Lead time in days
    if 'lead_days' not in df.columns:
        df['lead_days'] = (appt_dt.dt.date - sched_dt.dt.date).apply(lambda d: max(0, d.days))
        
    df['log_lead_days'] = np.log1p(df['lead_days'])
    df['is_same_day'] = (df['lead_days'] == 0).astype(int)
    
    # Day of week features (0 = Monday, 6 = Sunday)
    df['appointment_dow'] = appt_dt.dt.day_name()
    df['scheduled_dow'] = sched_dt.dt.day_name()
    
    # Age group segmentation
    bins = [-1, 12, 25, 45, 65, 120]
    labels = ['Child', 'Youth', 'Adult', 'MiddleAge', 'Senior']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
    
    # Distance buckets
    df['is_long_distance'] = (df['distance_km'] > 12.0).astype(int)
    
    return df


def get_preprocessor():
    """
    Builds a Scikit-Learn ColumnTransformer for scaling numeric features
    and one-hot encoding categorical features.
    """
    numeric_features = [
        'age', 
        'lead_days', 
        'log_lead_days', 
        'distance_km', 
        'prior_appointments', 
        'prior_noshows', 
        'prior_noshow_rate'
    ]
    
    categorical_features = [
        'gender', 
        'scholarship', 
        'clinic_department', 
        'appointment_slot', 
        'appointment_dow', 
        'scheduled_dow', 
        'age_group', 
        'sms_received', 
        'is_same_day', 
        'is_new_patient',
        'is_long_distance'
    ]
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor, numeric_features, categorical_features

