"""
Unit Tests - Data Pipeline and Leakage Prevention
Author: Rohit Singh Negi (Reg No: 12311143)
Project: Hospital Appointment No-Show Prediction
"""

import os
import unittest
import pandas as pd
from datetime import datetime, timedelta

from src.data.generate_dataset import generate_synthetic_appointments
from src.features.build_features import compute_historical_features, engineer_appointment_features


class TestDataPipeline(unittest.TestCase):

    def setUp(self):
        self.test_data_path = 'data/test_synthetic_sample.csv'
        self.df_sample = generate_synthetic_appointments(num_records=200, seed=123, output_path=self.test_data_path)

    def tearDown(self):
        if os.path.exists(self.test_data_path):
            os.remove(self.test_data_path)

    def test_synthetic_data_generation(self):
        """Test dataset shape, required columns, and value bounds."""
        self.assertEqual(len(self.df_sample), 200)
        required_cols = [
            'appointment_id', 'patient_id', 'gender', 'age', 'scholarship',
            'distance_km', 'clinic_department', 'scheduled_day', 'appointment_day',
            'appointment_slot', 'lead_days', 'sms_received', 'no_show'
        ]
        for col in required_cols:
            self.assertIn(col, self.df_sample.columns, f"Missing expected column: {col}")

        # Check domain constraints
        self.assertTrue((self.df_sample['age'] >= 0).all(), "Found negative age.")
        self.assertTrue((self.df_sample['lead_days'] >= 0).all(), "Found negative lead time.")
        self.assertTrue(set(self.df_sample['no_show'].unique()).issubset({0, 1}), "Target must be binary 0/1.")

    def test_chronological_leakage_prevention(self):
        """Verify that historical feature extraction uses strictly past events."""
        df_hist = compute_historical_features(self.df_sample)
        
        # New patients must have 0 prior appointments and 0 prior noshows
        new_patients = df_hist[df_hist['is_new_patient'] == 1]
        self.assertTrue((new_patients['prior_appointments'] == 0).all())
        self.assertTrue((new_patients['prior_noshows'] == 0).all())
        self.assertTrue((new_patients['prior_noshow_rate'] == 0.0).all())

        # Patients with prior visits must have valid rate between 0 and 1
        repeat_patients = df_hist[df_hist['is_new_patient'] == 0]
        if len(repeat_patients) > 0:
            self.assertTrue((repeat_patients['prior_appointments'] > 0).all())
            self.assertTrue((repeat_patients['prior_noshow_rate'] >= 0.0).all())
            self.assertTrue((repeat_patients['prior_noshow_rate'] <= 1.0).all())

    def test_feature_engineering_transforms(self):
        """Verify engineered operational features at booking time."""
        df_feat = engineer_appointment_features(self.df_sample)
        self.assertIn('log_lead_days', df_feat.columns)
        self.assertIn('is_same_day', df_feat.columns)
        self.assertIn('appointment_dow', df_feat.columns)
        self.assertIn('age_group', df_feat.columns)
        self.assertIn('is_long_distance', df_feat.columns)

        # Same day check
        same_day_records = df_feat[df_feat['lead_days'] == 0]
        if len(same_day_records) > 0:
            self.assertTrue((same_day_records['is_same_day'] == 1).all())


if __name__ == '__main__':
    unittest.main()
