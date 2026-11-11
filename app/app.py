"""
Hospital Appointment No-Show Prediction - Inference API & Dashboard
Author: Rohit Singh Negi (Reg No: 12311143)
Project: Hospital Appointment No-Show Prediction
Assessment: AI & Machine Learning - Algions AI Labs

This Flask application serves:
1. Operational Web Dashboard: interactive appointment risk assessment for clinic staff
2. REST API: /predict, /batch-predict, /health, /metrics
3. Human-in-the-loop decision-support guidelines with local explainability
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.build_features import engineer_appointment_features

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.joblib')
METADATA_PATH = os.path.join(BASE_DIR, 'models', 'model_metadata.json')

# Global model artifact cache
model = None
metadata = {}


def load_artifacts():
    global model, metadata
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}. Please run training script first.")
    
    model = joblib.load(MODEL_PATH)
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {'operational_decision_threshold': 0.35, 'model_name': 'Balanced_LogisticRegression'}
        
    print(f"Loaded model artifact successfully: {metadata.get('model_name', 'TrainedModel')}")
    print(f"Active Operational Decision Threshold: {metadata.get('operational_decision_threshold', 0.35)}")


# Load at startup
try:
    load_artifacts()
except Exception as e:
    print(f"Warning during artifact load: {e}")


def transform_single_record(data_dict):
    """
    Transforms a dictionary of appointment details into a DataFrame ready for model inference.
    Strictly uses information available at booking time.
    """
    # Defaults and validation
    age = float(data_dict.get('age', 35))
    gender = str(data_dict.get('gender', 'F')).upper()
    scholarship = int(data_dict.get('scholarship', 0))
    distance_km = float(data_dict.get('distance_km', 5.0))
    clinic_dept = str(data_dict.get('clinic_department', 'General Practice'))
    appt_slot = str(data_dict.get('appointment_slot', 'Morning'))
    sms_received = int(data_dict.get('sms_received', 0))
    
    sched_day_str = str(data_dict.get('scheduled_day', datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    appt_day_str = str(data_dict.get('appointment_day', datetime.now().strftime('%Y-%m-%d')))
    
    prior_appts = int(data_dict.get('prior_appointments', 0))
    prior_ns = int(data_dict.get('prior_noshows', 0))
    prior_rate = float(prior_ns / prior_appts) if prior_appts > 0 else 0.0
    is_new = 1 if prior_appts == 0 else 0
    
    row = {
        'patient_id': int(data_dict.get('patient_id', 999999)),
        'gender': gender,
        'age': age,
        'scholarship': scholarship,
        'distance_km': distance_km,
        'clinic_department': clinic_dept,
        'scheduled_day': sched_day_str,
        'appointment_day': appt_day_str,
        'appointment_slot': appt_slot,
        'sms_received': sms_received,
        'prior_appointments': prior_appts,
        'prior_noshows': prior_ns,
        'prior_noshow_rate': prior_rate,
        'is_new_patient': is_new
    }
    
    df = pd.DataFrame([row])
    df = engineer_appointment_features(df)
    return df


def generate_local_explanation(df_record, proba, top_importances):
    """
    Generates human-interpretable risk drivers for a single patient appointment.
    """
    row = df_record.iloc[0]
    drivers = []
    
    # 1. Lead time driver
    lead = row['lead_days']
    if lead == 0:
        drivers.append({"factor": "Same-Day Booking", "effect": "Decreases Risk", "impact": "High Positive", "detail": "Patient booked today and is highly likely to attend."})
    elif lead > 21:
        drivers.append({"factor": f"Long Lead Time ({lead} days)", "effect": "Increases Risk", "impact": "High Negative", "detail": f"Booked {lead} days in advance. High forgetting rate."})
    elif lead > 7:
        drivers.append({"factor": f"Moderate Lead Time ({lead} days)", "effect": "Increases Risk", "impact": "Medium Negative", "detail": "Extended booking window requires reminder reinforcement."})
        
    # 2. Prior history driver
    prior_appts = row['prior_appointments']
    prior_ns = row['prior_noshows']
    if prior_appts > 0:
        rate = row['prior_noshow_rate'] * 100
        if prior_ns > 0:
            drivers.append({"factor": f"Prior Missed Visits ({prior_ns}/{prior_appts}, {rate:.0f}%)", "effect": "Increases Risk", "impact": "High Negative", "detail": "Patient has previous history of unannounced no-shows."})
        else:
            drivers.append({"factor": f"Consistent Past Attendance ({prior_appts} visits)", "effect": "Decreases Risk", "impact": "Positive", "detail": "Patient consistently attends scheduled appointments."})
    else:
        drivers.append({"factor": "New Patient", "effect": "Neutral", "impact": "Neutral", "detail": "No historical clinic visit record."})
        
    # 3. SMS reminder
    if row['sms_received'] == 1:
        drivers.append({"factor": "SMS Reminder Dispatched", "effect": "Decreases Risk", "impact": "Positive", "detail": "Automated notification sent to mobile phone."})
    else:
        if lead >= 3:
            drivers.append({"factor": "No SMS Reminder Sent", "effect": "Increases Risk", "impact": "Medium Negative", "detail": "No automated text message reminder logged."})

    # 4. Age factor
    age = row['age']
    if 18 <= age <= 30:
        drivers.append({"factor": f"Age Demographic ({int(age)} yrs)", "effect": "Increases Risk", "impact": "Moderate Negative", "detail": "Young adult demographic exhibits higher baseline scheduling conflicts."})
    elif age >= 65:
        drivers.append({"factor": f"Senior Patient ({int(age)} yrs)", "effect": "Decreases Risk", "impact": "Moderate Positive", "detail": "Senior patients show higher punctuality and attendance reliability."})

    # 5. Distance factor
    dist = row['distance_km']
    if dist > 15.0:
        drivers.append({"factor": f"Long Transit Distance ({dist:.1f} km)", "effect": "Increases Risk", "impact": "Moderate Negative", "detail": "Significant commute distance creates transit vulnerability."})
        
    return drivers


def get_recommendation_tier(proba, threshold):
    """
    Maps predicted probability to operational risk tiers and actions.
    """
    if proba < 0.20:
        return {
            "tier": "Low Risk",
            "badge_color": "green",
            "action": "Standard Routine",
            "operational_workflow": "Send standard automated SMS reminder 24 hours prior. Keep standard schedule allocation. No staff intervention needed."
        }
    elif proba < threshold:
        return {
            "tier": "Moderate Risk",
            "badge_color": "yellow",
            "action": "Enhanced Digital Reminder",
            "operational_workflow": "Send two-way interactive SMS requesting confirmation (Reply YES/NO) 48 hours prior. Alert waitlisted patients for rapid fill if canceled."
        }
    elif proba < 0.55:
        return {
            "tier": "High Risk",
            "badge_color": "orange",
            "action": "Direct Staff Outreach",
            "operational_workflow": "Administrative staff phone call 48 hours prior to confirm attendance or offer rescheduling. Note in schedule as potential buffer slot."
        }
    else:
        return {
            "tier": "Critical Risk",
            "badge_color": "red",
            "action": "Priority Confirmation & Scheduling Buffer",
            "operational_workflow": "Priority direct phone contact. If unconfirmed 24 hours before, schedule an urgent walk-in buffer patient in parallel to prevent physician idle time."
        }


# -------------------------------------------------------------
# Routes
# -------------------------------------------------------------

@app.route('/')
def home():
    """Serves the interactive web dashboard."""
    threshold = metadata.get('operational_decision_threshold', 0.35)
    model_name = metadata.get('model_name', 'Balanced_LogisticRegression')
    return render_template('index.html', threshold=threshold, model_name=model_name)


@app.route('/reports/figures/<path:filename>')
def serve_figure(filename):
    """Serves generated evaluation figures."""
    figures_dir = os.path.join(BASE_DIR, 'reports', 'figures')
    return send_from_directory(figures_dir, filename)


@app.route('/health', methods=['GET'])
def health():
    """System health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Hospital Appointment No-Show Prediction Decision-Support API',
        'candidate_name': 'Rohit Singh Negi',
        'registration_number': '12311143',
        'model_loaded': model is not None,
        'model_name': metadata.get('model_name', 'Unknown'),
        'active_threshold': metadata.get('operational_decision_threshold', 0.35),
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Returns model evaluation metrics from metadata."""
    return jsonify(metadata), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Inference endpoint: evaluates probability of appointment no-show.
    Returns real model output, decision tier, cost-optimized classification, and explanations.
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({'error': f'Invalid JSON payload: {str(e)}'}), 400
        
    try:
        df_input = transform_single_record(data)
        proba_no_show = float(model.predict_proba(df_input)[0, 1])
        threshold = float(metadata.get('operational_decision_threshold', 0.35))
        
        # Decision-support binary recommendation
        prediction_binary = int(proba_no_show >= threshold)
        
        # Risk tier and clinical operations recommendations
        rec_info = get_recommendation_tier(proba_no_show, threshold)
        
        # Local explainability drivers
        drivers = generate_local_explanation(df_input, proba_no_show, metadata.get('top_feature_importances', {}))
        
        response = {
            'prediction': prediction_binary,
            'prediction_label': 'High Risk / Potential No-Show' if prediction_binary == 1 else 'Likely to Attend',
            'no_show_probability': round(proba_no_show, 4),
            'risk_percentage': round(proba_no_show * 100, 1),
            'operational_threshold': threshold,
            'risk_tier': rec_info['tier'],
            'badge_color': rec_info['badge_color'],
            'recommended_action': rec_info['action'],
            'operational_workflow': rec_info['operational_workflow'],
            'input_summary': {
                'lead_days': int(df_input['lead_days'].iloc[0]),
                'patient_age': int(df_input['age'].iloc[0]),
                'appointment_slot': str(df_input['appointment_slot'].iloc[0]),
                'clinic_department': str(df_input['clinic_department'].iloc[0]),
                'sms_received': int(df_input['sms_received'].iloc[0]),
                'prior_visits': int(df_input['prior_appointments'].iloc[0]),
                'prior_missed': int(df_input['prior_noshows'].iloc[0])
            },
            'risk_drivers': drivers,
            'governance_notice': (
                "DECISION-SUPPORT NOTICE: This predictive score is intended solely to prioritize "
                "administrative reminders and operational scheduling buffers. It must NEVER be used "
                "for clinical diagnosis, medical triage, or denial of healthcare services."
            )
        }
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({'error': f'Inference error: {str(e)}'}), 500


@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """
    Batch scoring endpoint: processes multiple appointment records.
    Accepts JSON array of appointments or CSV file.
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    records = []
    if request.is_json:
        payload = request.get_json()
        if isinstance(payload, list):
            records = payload
        elif 'appointments' in payload:
            records = payload['appointments']
    elif 'file' in request.files:
        uploaded_file = request.files['file']
        df_uploaded = pd.read_csv(uploaded_file)
        records = df_uploaded.to_dict(orient='records')
        
    if not records:
        return jsonify({'error': 'No appointment records provided in request.'}), 400
        
    threshold = float(metadata.get('operational_decision_threshold', 0.35))
    results = []
    
    for item in records:
        try:
            df_single = transform_single_record(item)
            p = float(model.predict_proba(df_single)[0, 1])
            is_flagged = int(p >= threshold)
            rec = get_recommendation_tier(p, threshold)
            
            results.append({
                'appointment_id': item.get('appointment_id', 'N/A'),
                'patient_id': item.get('patient_id', 'N/A'),
                'lead_days': int(df_single['lead_days'].iloc[0]),
                'no_show_probability': round(p, 4),
                'risk_percentage': round(p * 100, 1),
                'flagged_for_intervention': is_flagged,
                'risk_tier': rec['tier'],
                'recommended_action': rec['action']
            })
        except Exception as ex:
            results.append({
                'appointment_id': item.get('appointment_id', 'N/A'),
                'error': str(ex)
            })
            
    summary = {
        'total_scored': len(results),
        'high_risk_flagged': sum(1 for r in results if r.get('flagged_for_intervention') == 1),
        'operational_threshold': threshold,
        'results': results
    }
    return jsonify(summary), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"\n=======================================================")
    print(f"Hospital Appointment No-Show Prediction Dashboard")
    print(f"Candidate: Rohit Singh Negi (Reg No: 12311143)")
    print(f"Starting server on http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host='127.0.0.1', port=port, debug=False)
