"""
Hospital Appointment No-Show Prediction Dataset Generator
Author: Rohit Singh Negi (Reg No: 12311143)
Project: Hospital Appointment No-Show Prediction
Assessment: AI & Machine Learning - Algions AI Labs

This script generates a realistic, de-identified synthetic outpatient appointment dataset
modeled after real-world hospital operational dynamics (e.g. Kaggle Medical Appointment No-Shows benchmark).
Strictly adheres to data governance:
- Zero sensitive clinical data (no diagnoses, symptoms, treatments, or medication)
- Clear point-in-time reference (booking timestamp vs appointment date)
- Realistic distributions: lead time, patient age, SMS reminders, distance, and prior visit history
"""

import os
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def generate_synthetic_appointments(num_records=15000, seed=42, output_path='data/raw/hospital_appointments.csv'):
    """
    Generates realistic hospital appointment records for operational no-show modeling.
    
    Parameters:
    -----------
    num_records : int
        Number of appointment rows to generate (default: 15,000)
    seed : int
        Random seed for exact reproducibility
    output_path : str
        Path where raw CSV will be saved
    """
    np.random.seed(seed)
    print(f"Generating {num_records} synthetic appointment records with seed={seed}...")

    # Realistic pool of patients (some repeat visitors to simulate longitudinal behavior)
    num_unique_patients = int(num_records * 0.60)
    patient_pool = [100000 + i for i in range(num_unique_patients)]
    
    # Assign persistent attributes to patients: gender, age, base reliability factor
    patient_genders = np.random.choice(['F', 'M'], size=num_unique_patients, p=[0.62, 0.38])
    
    # Age distribution: mixture of infants/children, working adults, and seniors
    # Common in general outpatient clinics
    n1 = int(num_unique_patients * 0.18)
    n2 = int(num_unique_patients * 0.32)
    n3 = int(num_unique_patients * 0.30)
    n4 = num_unique_patients - (n1 + n2 + n3)
    age_dist = np.concatenate([
        np.random.randint(0, 18, size=n1),
        np.random.randint(18, 40, size=n2),
        np.random.randint(40, 65, size=n3),
        np.random.randint(65, 92, size=n4)
    ])
    np.random.shuffle(age_dist)
    
    # Patient social assistance / scholarship (welfare)
    scholarship_status = np.random.binomial(n=1, p=0.095, size=num_unique_patients)
    
    # Distance to facility in kilometers (gamma distribution: majority 1-8 km, long tail to 25 km)
    distances = np.round(np.clip(np.random.gamma(shape=2.5, scale=2.8, size=num_unique_patients), 0.5, 30.0), 1)
    
    # Inherent patient reliability / punctuality baseline (latent score 0 to 1)
    # Higher score = more reliable attendee
    patient_reliability = np.clip(np.random.beta(a=5, b=1.5, size=num_unique_patients), 0.1, 0.98)
    
    patient_df = pd.DataFrame({
        'patient_id': patient_pool,
        'gender': patient_genders,
        'age': age_dist,
        'scholarship': scholarship_status,
        'distance_km': distances,
        'latent_reliability': patient_reliability
    })

    # Start date of scheduling: Jan 1, 2026 to May 31, 2026 (5 months)
    start_date = datetime(2026, 1, 5, 8, 0, 0)
    
    records = []
    departments = ['General Practice', 'Pediatrics', 'Cardiology', 'Orthopedics', 'Dermatology', 'Ophthalmology', 'Dental']
    dept_weights = [0.35, 0.15, 0.12, 0.12, 0.10, 0.08, 0.08]
    
    # Generate appointments
    patient_selection = np.random.choice(patient_pool, size=num_records, replace=True)
    
    # Lead time in days:
    # 25% same-day (lead_time = 0)
    # 35% short lead (1 - 7 days)
    # 25% medium lead (8 - 21 days)
    # 15% long lead (22 - 60 days)
    lead_time_segments = np.random.choice([0, 1, 2, 3], size=num_records, p=[0.25, 0.35, 0.25, 0.15])
    
    # Track patient booking timeline
    for i in range(num_records):
        p_id = patient_selection[i]
        p_info = patient_df.loc[patient_df['patient_id'] == p_id].iloc[0]
        
        # Schedule day between day 0 and day 140
        sched_offset_days = int(np.random.uniform(0, 140))
        sched_offset_secs = int(np.random.uniform(28800, 61200)) # 8:00 AM to 5:00 PM
        scheduled_dt = start_date + timedelta(days=sched_offset_days, seconds=sched_offset_secs)
        
        seg = lead_time_segments[i]
        if seg == 0:
            lead_days = 0
        elif seg == 1:
            lead_days = int(np.random.randint(1, 8))
        elif seg == 2:
            lead_days = int(np.random.randint(8, 22))
        else:
            lead_days = int(np.random.randint(22, 61))
            
        appointment_date = (scheduled_dt + timedelta(days=lead_days)).date()
        
        # Clinic operates Mon-Sat (1 to 6). Skip Sundays (if Sunday, push to Monday)
        if appointment_date.weekday() == 6: # Sunday
            lead_days += 1
            appointment_date = (scheduled_dt + timedelta(days=lead_days)).date()
            
        # Appointment time slot
        time_slot = np.random.choice(['Morning', 'Afternoon'], p=[0.58, 0.42])
        
        # Department
        dept = np.random.choice(departments, p=dept_weights)
        if p_info['age'] <= 14:
            dept = 'Pediatrics'
        elif dept == 'Pediatrics' and p_info['age'] > 18:
            dept = 'General Practice'
            
        # SMS reminder:
        # Same day appointments don't need SMS.
        # For lead_days >= 3, SMS is dispatched ~65% of the time.
        if lead_days < 3:
            sms_received = 0
        else:
            sms_received = int(np.random.binomial(n=1, p=0.68))
            
        # Calculate realistic probability of NO-SHOW:
        # Base log-odds calculated from operational variables
        # Target average no-show rate ~ 20.5%
        
        # 1. Lead time effect: strong positive correlation with no-show
        if lead_days == 0:
            lead_effect = -2.2 # Same-day bookings show up ~95% of the time
        elif lead_days <= 3:
            lead_effect = -0.5
        elif lead_days <= 7:
            lead_effect = -0.1
        elif lead_days <= 14:
            lead_effect = 0.3
        elif lead_days <= 30:
            lead_effect = 0.7
        else:
            lead_effect = 1.1 # High lead time increases risk
            
        # 2. Age effect: young adults (18-30) miss more; seniors and infants show up more
        age = p_info['age']
        if age < 12:
            age_effect = -0.25 # Parents bring children
        elif 18 <= age <= 30:
            age_effect = 0.45  # Higher missed rates
        elif 31 <= age <= 60:
            age_effect = 0.05
        else:
            age_effect = -0.40 # Seniors have lower no-show rates
            
        # 3. SMS reminder effect: reducing no-show risk
        sms_effect = -0.35 if sms_received == 1 else 0.15
        
        # 4. Day of week effect (Friday & Monday have slightly more no-shows)
        dow = appointment_date.weekday() # 0: Mon, 4: Fri, 5: Sat
        dow_effect = 0.20 if dow in [0, 4] else (-0.10 if dow in [1, 2, 3] else 0.05)
        
        # 5. Slot effect: Afternoon has higher friction
        slot_effect = 0.18 if time_slot == 'Afternoon' else -0.10
        
        # 6. Scholarship (socioeconomic factor): modest increase due to transport/work flexibility issues
        scholarship_effect = 0.30 if p_info['scholarship'] == 1 else 0.0
        
        # 7. Distance effect: >15km increases no-show likelihood
        dist = p_info['distance_km']
        dist_effect = 0.03 * (dist - 5.0)
        
        # 8. Patient intrinsic reliability factor
        rel_effect = -2.5 * (p_info['latent_reliability'] - 0.7)
        
        # Combine log-odds
        base_intercept = -1.25
        log_odds = (base_intercept + lead_effect + age_effect + sms_effect +
                    dow_effect + slot_effect + scholarship_effect + dist_effect + rel_effect)
        
        prob_no_show = 1.0 / (1.0 + np.exp(-log_odds))
        no_show_label = int(np.random.binomial(n=1, p=prob_no_show))
        
        records.append({
            'appointment_id': 1000000 + i,
            'patient_id': p_id,
            'gender': p_info['gender'],
            'age': int(p_info['age']),
            'scholarship': int(p_info['scholarship']),
            'distance_km': float(p_info['distance_km']),
            'clinic_department': dept,
            'scheduled_day': scheduled_dt.strftime('%Y-%m-%d %H:%M:%S'),
            'appointment_day': appointment_date.strftime('%Y-%m-%d'),
            'appointment_slot': time_slot,
            'lead_days': lead_days,
            'sms_received': sms_received,
            'no_show': no_show_label
        })
        
    df = pd.DataFrame(records)
    
    # Sort chronologically by appointment_day then scheduled_day
    df['appointment_day_dt'] = pd.to_datetime(df['appointment_day'])
    df['scheduled_day_dt'] = pd.to_datetime(df['scheduled_day'])
    df = df.sort_values(by=['appointment_day_dt', 'scheduled_day_dt']).reset_index(drop=True)
    df = df.drop(columns=['appointment_day_dt', 'scheduled_day_dt'])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Dataset successfully created and saved to: {output_path}")
    print(f"Total records: {len(df)}")
    print(f"Target distribution (no_show):")
    print(df['no_show'].value_counts(normalize=True).apply(lambda x: f"{x*100:.2f}%"))
    print(f"Average lead time: {df['lead_days'].mean():.1f} days")
    print(f"Unique patients: {df['patient_id'].nunique()}")
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic hospital appointments dataset.")
    parser.add_argument('--records', type=int, default=15000, help="Number of records to generate.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--output', type=str, default='data/raw/hospital_appointments.csv', help="Output CSV path.")
    args = parser.parse_args()
    
    generate_synthetic_appointments(num_records=args.records, seed=args.seed, output_path=args.output)
