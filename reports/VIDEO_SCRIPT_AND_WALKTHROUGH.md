# Demonstration Video Script & Interview Walkthrough Guide

**Candidate Name:** Rohit Singh Negi  
**Registration Number:** 12311143  
**Role:** AI & Machine Learning  
**Project:** Hospital Appointment No-Show Prediction  
**Assessment:** Advanced Individual Assessment &bull; Algions AI Labs Private Limited  

---

## Part 1: 3–5 Minute Video Demonstration Script

> **Recording Tip:** Use screen recorder (QuickTime on Mac, OBS, or Loom). Ensure full screen capture of your browser and terminal. Speak with a steady, confident pace.

### Timeline & Screen Action Overview
- **0:00 – 0:45 (45s):** Introduction, Business Problem & Cost Asymmetry Framing
- **0:45 – 1:45 (60s):** EDA, Zero-Leakage Chronological Split & Feature Engineering
- **1:45 – 2:45 (60s):** Model Evaluation, Imbalance Handling & Threshold Cost Optimization
- **2:45 – 4:00 (75s):** Live Inference Demo (Single Prediction, Explainability & Batch Scoring)
- **4:00 – 4:30 (30s):** Responsible AI, Clinical Boundary & Conclusion

---

### Step-by-Step Script

#### [0:00 - 0:45] Intro & Business Problem Framing
- **Screen:** Open project README or Web Dashboard homepage showing your name: **Rohit Singh Negi**, Reg No: **12311143**.
- **Voiceover:**
  > "Hello everyone. My name is Rohit Singh Negi, registration number 12311143. Today, I am presenting my end-to-end machine learning decision-support solution for Hospital Appointment No-Show Prediction.
  >
  > In outpatient clinical operations, missed appointments create massive resource waste. An empty doctor slot costs the facility roughly \$180 in lost revenue and idle provider time, whereas an administrative reminder or staff confirmation phone call costs only about \$15. 
  >
  > Because false negatives are 12 times more costly than false positives, standard 50-50 accuracy models fail in production. My solution introduces a cost-sensitive, threshold-optimized ML pipeline designed strictly for administrative reminder prioritization, never for clinical care denial."

#### [0:45 - 1:45] Data Pipeline, EDA & Zero Leakage
- **Screen:** Switch to Jupyter Notebook or show `reports/figures/lead_time_vs_noshow.png` and `age_distribution.png`.
- **Voiceover:**
  > "Let's examine the data pipeline. In strict compliance with HIPAA and healthcare governance, the dataset contains zero sensitive clinical diagnoses—focusing purely on operational booking signals.
  >
  > Through exploratory analysis across 15,000 records, we uncovered three dominant signals:
  > First, booking lead time: same-day appointments have an empirical no-show rate under 5%, whereas appointments booked more than three weeks in advance exceed a 30% no-show rate.
  > Second, automated SMS notifications cut no-show rates by over 8 percentage points.
  > Third, longitudinal patient attendance: a patient's historical missed-visit ratio is the single strongest behavioral predictor.
  >
  > Critically, to prevent temporal target leakage, I implemented a strict chronological split—70% training, 15% validation, and 15% held-out test—and calculated patient historical visit metrics sequentially using only events that occurred prior to each index booking date."

#### [1:45 - 2:45] Model Comparison & Threshold Cost Optimization
- **Screen:** Show the Model Analytics tab in the dashboard or `cost_threshold_curve.png` and `confusion_matrix.png`.
- **Voiceover:**
  > "To address the 4:1 class imbalance, I benchmarked an unweighted logistic regression baseline against balanced logistic regression, weighted random forest, and histogram gradient boosting.
  >
  > On our held-out test set of 2,250 unseen future appointments, our candidate models achieve an ROC-AUC of 0.698 and PR-AUC of 0.437.
  >
  > Here is where engineering rigor matters: looking at our operational cost curve, the naive 0.50 threshold catches only 12% of no-shows in the baseline, leaving \$43 in wasted capacity per appointment. By sweeping decision thresholds on the validation set, we identify the cost-optimal operating threshold at tau = 0.35.
  >
  > At tau = 0.35, the model achieves 97.7% recall, catching 593 out of 607 no-shows and slashing expected operational losses from \$43.01 down to \$10.18 per appointment—a 76% financial saving."

#### [2:45 - 4:00] Live Application & REST API Demonstration
- **Screen:** Switch to the live web browser running at `http://127.0.0.1:5001`.
- **Action 1 (High Risk Case):**
  - Click button **'Load High Risk Sample'** (Lead time: 28 days, age: 23, 2 past no-shows, no SMS).
  - Click **'Evaluate No-Show Probability'**.
- **Voiceover:**
  > "Now let's demonstrate the working inference interface. The web dashboard dynamically loads our serialized model pipeline.
  > 
  > I'll click 'Load High Risk Sample': a 23-year-old patient booking 28 days in advance, with 2 previous missed appointments and no SMS reminder.
  > 
  > When I click evaluate, the model returns a real-time probability of 67.8%, categorized as 'High Risk'. Notice the local explainability drivers: the long lead time and past missed appointments push the risk upward. The system recommends: 'Staff telephone outreach 48 hours prior to offer flexible rescheduling'."
- **Action 2 (Low Risk Case):**
  - Click button **'Load Low Risk Sample'** (Lead time: 1 day, age: 68, perfect past attendance, SMS sent).
  - Click **'Evaluate No-Show Probability'**.
- **Voiceover:**
  > "Next, I click 'Load Low Risk Sample': a senior patient with same-week booking and perfect past attendance. The probability drops to 11.2% (Low Risk), recommending standard automated SMS only."
- **Action 3 (Batch Scoring Tab):**
  - Click **'Batch Schedule Scoring'** tab.
  - Click **'Run Sample Clinic Day'**.
- **Voiceover:**
  > "For clinic workflow efficiency, the 'Batch Schedule Scoring' tab allows staff to process the entire clinic roster simultaneously, automatically categorizing patients into priority contact tiers."

#### [4:00 - 4:30] Responsible AI & Conclusion
- **Screen:** Click **'Governance & Limitations'** tab or project summary.
- **Voiceover:**
  > "Finally, responsible AI governance: this system is strictly a decision-support tool. It contains hardcoded ethical boundaries—it is never permitted to triage clinical urgency, deny care, or charge punitive fees. Demographic risk flags prompt supportive interventions, such as transit assistance, rather than penalties.
  >
  > The repository includes automated tests with a 100% pass rate, full Docker readiness, and complete reproducibility. Thank you."

---

## Part 2: 15-Minute Technical Walkthrough Q&A Preparation

During the 15-minute live walkthrough, senior evaluators test your depth of understanding and ownership. Here are the core questions they will ask and how to answer them with authority:

### Q1: "Why did you choose a chronological split instead of Stratified K-Fold Cross-Validation?"
> **Your Answer:**  
> "In hospital scheduling, appointments arrive as a time series. If we use random or stratified K-Fold, appointments from May and June are placed in the training set while appointments from February are in the test set. This creates severe temporal leakage because future attendance patterns and seasonal clinic habits bleed into past predictions. Chronological splitting (Train: Jan–Apr, Val: Apr–May, Test: May–Jul) faithfully replicates the real deployment environment where we train on the past and predict into the future."

### Q2: "Why is the Precision relatively low (~30-34%) at your chosen threshold?"
> **Your Answer:**  
> "That is a deliberate, mathematically justified engineering trade-off governed by our asymmetric cost matrix. In hospital operations, a False Negative costs \$180 in idle doctor capacity, while a False Positive costs only \$15 for a quick phone call. If we tuned for high precision (e.g. threshold 0.70), we would miss 80% of all no-shows, causing tens of thousands of dollars in wasted clinical time. At threshold 0.35, our recall is 97.7%. Yes, we have more false positives, but a polite reminder phone call to an attending patient has almost zero downside and patients appreciate it."

### Q3: "How did you ensure there is no future data leakage in patient history?"
> **Your Answer:**  
> "In `src/features/build_features.py`, in `compute_historical_features()`, the dataframe is ordered chronologically by appointment date and booking timestamp. We maintain a sequential state dictionary of past appointments and missed visits per patient. When processing patient $i$ at appointment date $T$, only events strictly prior to $T$ are aggregated. For first-time visitors, `is_new_patient` is flagged as 1, and prior counts are safely initialized to 0."

### Q4: "Why did you pick Balanced Logistic Regression as the primary deployed model instead of a deep neural network or complex ensemble?"
> **Your Answer:**  
> "Three reasons:
> 1. **Calibration & Interpretability:** Logistic regression produces smooth, well-calibrated probabilities across the range, which are essential for reliable threshold tuning. Complex tree ensembles often distort probabilities toward 0 and 1.
> 2. **Competitive Performance:** Balanced Logistic Regression achieved an ROC-AUC of 0.698 and PR-AUC of 0.437, on par with HistGradientBoosting (0.442).
> 3. **Latency and Robustness:** In outpatient clinic systems, lightweight inference with transparent linear log-odds enables instantaneous local explainability with zero latency and high clinical auditability."

---

## Part 3: 10-Minute Live Coding / Debugging Task Cheat Sheet

During the 10-minute live coding task, the interviewer will typically ask you to make a quick modification to your codebase. Here is how to handle the common live tasks effortlessly:

### Task Scenario A: "Change the False Positive cost from \$15 to \$30 and re-evaluate the optimal threshold."
1. Open `src/utils/metrics.py`.
2. Edit lines 26–27:
   ```python
   COST_FALSE_POSITIVE = 30.0   # Updated from 15.0
   COST_FALSE_NEGATIVE = 180.0
   ```
3. Run in terminal:
   ```bash
   python3 src/models/train.py
   ```
4. Explain: "Notice how doubling the cost of a reminder phone call causes the optimal threshold to shift upwards from 0.35 to ~0.42, reducing staff call volume while maintaining strong recall."

### Task Scenario B: "Add a new feature: 'Is Weekend Booking'."
1. Open `src/features/build_features.py`.
2. In `engineer_appointment_features(df)`:
   ```python
   df['is_weekend_booking'] = sched_dt.dt.dayofweek.isin([5, 6]).astype(int)
   ```
3. In `get_preprocessor()`: add `'is_weekend_booking'` to `categorical_features`.
4. Re-run tests:
   ```bash
   python3 run_tests.py
   ```

### Task Scenario C: "Demonstrate a curl prediction against the running REST API."
In terminal:
```bash
curl -X POST http://127.0.0.1:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": 109999,
    "age": 25,
    "gender": "M",
    "scholarship": 1,
    "distance_km": 15.0,
    "clinic_department": "General Practice",
    "scheduled_day": "2026-05-01 10:00:00",
    "appointment_day": "2026-05-25",
    "appointment_slot": "Afternoon",
    "prior_appointments": 2,
    "prior_noshows": 2,
    "sms_received": 0
  }'
```
Show the formatted JSON response with risk tier and drivers.

