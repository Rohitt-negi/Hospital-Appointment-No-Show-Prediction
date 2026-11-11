"""
Unit Tests - REST API Endpoints
Author: Rohit Singh Negi (Reg No: 12311143)
Project: Hospital Appointment No-Show Prediction
"""

import unittest
import json
from app.app import app


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_health_endpoint(self):
        """Test GET /health returns 200, healthy status and candidate details."""
        res = self.app.get('/health')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertEqual(data['candidate_name'], 'Rohit Singh Negi')
        self.assertEqual(data['registration_number'], '12311143')
        self.assertTrue(data['model_loaded'])

    def test_metrics_endpoint(self):
        """Test GET /metrics returns model evaluation metadata."""
        res = self.app.get('/metrics')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertIn('test_metrics_at_optimal_threshold', data)
        self.assertIn('top_feature_importances', data)

    def test_predict_single_endpoint(self):
        """Test POST /predict with a realistic single appointment payload."""
        payload = {
            'patient_id': 109999,
            'age': 25,
            'gender': 'M',
            'scholarship': 1,
            'distance_km': 15.0,
            'clinic_department': 'General Practice',
            'scheduled_day': '2026-05-01 10:00:00',
            'appointment_day': '2026-05-20',
            'appointment_slot': 'Afternoon',
            'prior_appointments': 2,
            'prior_noshows': 1,
            'sms_received': 0
        }
        res = self.app.post('/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        
        self.assertIn('no_show_probability', data)
        self.assertIn('prediction', data)
        self.assertIn('risk_tier', data)
        self.assertIn('recommended_action', data)
        self.assertIn('risk_drivers', data)
        self.assertIn('governance_notice', data)
        self.assertGreater(len(data['risk_drivers']), 0)

    def test_predict_invalid_payload(self):
        """Test POST /predict handles malformed or empty body gracefully."""
        res = self.app.post('/predict', data="not-json", content_type='application/json')
        self.assertIn(res.status_code, [400, 500])

    def test_batch_predict_endpoint(self):
        """Test POST /batch-predict scores a list of appointments."""
        batch_payload = [
            {
                'appointment_id': 101,
                'patient_id': 201,
                'age': 65,
                'gender': 'F',
                'scholarship': 0,
                'distance_km': 3.0,
                'clinic_department': 'Cardiology',
                'scheduled_day': '2026-05-10 09:00:00',
                'appointment_day': '2026-05-12',
                'appointment_slot': 'Morning',
                'prior_appointments': 4,
                'prior_noshows': 0,
                'sms_received': 1
            },
            {
                'appointment_id': 102,
                'patient_id': 202,
                'age': 21,
                'gender': 'M',
                'scholarship': 1,
                'distance_km': 18.0,
                'clinic_department': 'General Practice',
                'scheduled_day': '2026-04-10 14:00:00',
                'appointment_day': '2026-05-12',
                'appointment_slot': 'Afternoon',
                'prior_appointments': 2,
                'prior_noshows': 2,
                'sms_received': 0
            }
        ]
        res = self.app.post('/batch-predict', data=json.dumps(batch_payload), content_type='application/json')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data['total_scored'], 2)
        self.assertEqual(len(data['results']), 2)


if __name__ == '__main__':
    unittest.main()
