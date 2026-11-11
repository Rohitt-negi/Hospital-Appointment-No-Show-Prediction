"""
Unit Tests - Model Loading and Prediction Robustness
Author: Rohit Singh Negi (Reg No: 12311143)
Project: Hospital Appointment No-Show Prediction
"""

import os
import unittest
import joblib
import json
import numpy as np
import pandas as pd

from app.app import transform_single_record, get_recommendation_tier


class TestModelInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_path = 'models/best_model.joblib'
        cls.metadata_path = 'models/model_metadata.json'
        cls.assertTrue(os.path.exists(cls.model_path), f"Artifact missing: {cls.model_path}")
        cls.model = joblib.load(cls.model_path)
        with open(cls.metadata_path, 'r') as f:
            cls.metadata = json.load(f)

    def test_model_loading(self):
        """Verify model pipeline artifact loads and contains expected steps."""
        self.assertIsNotNone(self.model)
        self.assertTrue(hasattr(self.model, 'predict_proba'))
        self.assertIn('preprocessor', self.model.named_steps)
        self.assertIn('classifier', self.model.named_steps)

    def test_prediction_output_contract(self):
        """Ensure model returns valid probability between 0 and 1."""
        sample_input = {
            'patient_id': 105555,
            'age': 32,
            'gender': 'F',
            'scholarship': 0,
            'distance_km': 6.5,
            'clinic_department': 'General Practice',
            'scheduled_day': '2026-05-01 10:00:00',
            'appointment_day': '2026-05-15',
            'appointment_slot': 'Morning',
            'prior_appointments': 3,
            'prior_noshows': 0,
            'sms_received': 1
        }
        df_feat = transform_single_record(sample_input)
        probas = self.model.predict_proba(df_feat)[0]

        # Valid 2-class probability distribution
        self.assertEqual(len(probas), 2)
        self.assertAlmostEqual(probas.sum(), 1.0, places=4)
        self.assertGreaterEqual(probas[1], 0.0)
        self.assertLessEqual(probas[1], 1.0)

    def test_risk_direction_monotonicity(self):
        """
        Verify domain monotonicity:
        A high lead-time patient with prior missed visits and no SMS should have
        higher no-show probability than an identical patient with 0 lead time and past attendance.
        """
        high_risk_input = {
            'patient_id': 105555,
            'age': 22,
            'gender': 'M',
            'scholarship': 1,
            'distance_km': 20.0,
            'clinic_department': 'General Practice',
            'scheduled_day': '2026-04-01 10:00:00',
            'appointment_day': '2026-05-15', # 44 days lead time
            'appointment_slot': 'Afternoon',
            'prior_appointments': 2,
            'prior_noshows': 2, # 100% past no-show
            'sms_received': 0
        }
        
        low_risk_input = {
            'patient_id': 105555,
            'age': 70,
            'gender': 'F',
            'scholarship': 0,
            'distance_km': 2.0,
            'clinic_department': 'Cardiology',
            'scheduled_day': '2026-05-14 10:00:00',
            'appointment_day': '2026-05-15', # 1 day lead time
            'appointment_slot': 'Morning',
            'prior_appointments': 6,
            'prior_noshows': 0, # 0% past no-show
            'sms_received': 1
        }

        df_high = transform_single_record(high_risk_input)
        df_low = transform_single_record(low_risk_input)

        p_high = self.model.predict_proba(df_high)[0, 1]
        p_low = self.model.predict_proba(df_low)[0, 1]

        self.assertGreater(p_high, p_low, "High risk profile must produce higher probability than low risk profile.")

    def test_operational_tier_mapping(self):
        """Test business tier and action assignments across probabilities."""
        thresh = 0.35
        low_rec = get_recommendation_tier(0.12, thresh)
        self.assertEqual(low_rec['tier'], 'Low Risk')

        high_rec = get_recommendation_tier(0.48, thresh)
        self.assertEqual(high_rec['tier'], 'High Risk')

        critical_rec = get_recommendation_tier(0.75, thresh)
        self.assertEqual(critical_rec['tier'], 'Critical Risk')


if __name__ == '__main__':
    unittest.main()
