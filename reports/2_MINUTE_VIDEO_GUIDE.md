# 2-Minute Video Presentation Guide & Script
**Hospital Outpatient Appointment No-Show Prediction**  
**Candidate:** Rohit Singh Negi | **Registration Number:** 12311143  
**Assessment:** AI & Machine Learning &bull; Algions AI Labs Private Limited  
**Target Duration:** Exactly 2 Minutes (120 Seconds)

> **PDF Version Available:** You can also open the formatted PDF cheat-sheet directly at:  
> [`reports/2_Minute_Video_Walkthrough_Guide.pdf`](file:///Users/rohitsinghnegi/Desktop/Ai_assianment/reports/2_Minute_Video_Walkthrough_Guide.pdf)

---

## 1. Quick Project Overview (What You Need to Know)

| Component | High-Level Overview (What to Keep in Mind) |
| :--- | :--- |
| **What is this project?** | An AI decision-support tool for hospital receptionists and clinic operations. It predicts if a patient is likely to miss a scheduled outpatient appointment so staff can send targeted reminders or manage backup schedule buffers. |
| **Where did the dataset come from?** | A de-identified synthetic dataset of **15,000 appointment records** modeled on the standard public **Kaggle Medical Appointment No-Shows** benchmark. Strictly HIPAA compliant: **zero sensitive clinical data** (no diagnoses, medical histories, or medications). Uses only booking-time variables (lead time, age, distance, SMS sent, and past attendance history). |
| **What ML models were used?** | Evaluated 4 algorithms: **Baseline Logistic Regression** (unweighted), **Balanced Logistic Regression** (class-weighted), **Random Forest**, and **Histogram Gradient Boosting**. Balanced Logistic Regression was chosen as the primary model for its clean probability calibration, zero latency, and explainability. |
| **Key Result & Business Cost** | Because an empty doctor slot costs **~$180** while a reminder call costs only **~$15**, missing a no-show is 12x worse! By adjusting our decision threshold from 0.50 to **0.35**, our model captures **97.7% of all no-shows**, cutting wasted clinic capacity costs by **76%** (\$43.01 down to \$10.18 per appointment). |
| **Governance & Ethics** | Purely a **decision-support tool** for administrative reminders. It is **never used to deny care**, alter treatments, or triage clinical urgency. High risk triggers proactive support (like transport help or rescheduling). |

---

## 2. Quick Setup Before You Press Record

1. **Start the local server:**
   In Terminal, run:
   ```bash
   python3 app/app.py
   ```
   *(Server starts at `http://127.0.0.1:5001`)*
2. **Open browser:**
   Open `http://127.0.0.1:5001` in Chrome or Safari. Make sure the top navbar clearly shows your name (**Rohit Singh Negi**) and Reg No (**12311143**).
3. **Screen recorder:**
   Open **QuickTime Player** (`File -> New Screen Recording`), **Loom**, or **OBS**. Record the browser window and your microphone.

---

## 3. Second-by-Second 2-Minute (120-Second) Spoken Script

| Time | Screen Action (What to Show) | Exact Spoken Script (What to Say) |
| :--- | :--- | :--- |
| **0:00 – 0:25**<br/>*(25s)* | **Website Home Screen**<br/>Show browser at `http://127.0.0.1:5001` with navbar displaying your name and Reg No. | *"Hello everyone. My name is **Rohit Singh Negi**, registration number **12311143**. Today I am presenting my project: **Hospital Appointment No-Show Prediction**.<br/><br/>In outpatient hospital operations, when a patient misses an appointment without notice, doctors sit idle and valuable clinic time is wasted. Our goal is to build an operational decision-support tool that predicts no-show risks at the time of booking, helping clinic staff direct phone reminders and organize scheduling buffers."* |
| **0:25 – 0:50**<br/>*(25s)* | **Model Analytics Tab**<br/>Click the *'Model Analytics & Validation'* tab. Hover over Lead Time and Age plots. | *"For the dataset, respecting healthcare privacy and HIPAA standards, we used a de-identified outpatient dataset of 15,000 records modeled after the public Kaggle Medical Appointment benchmark. It strictly excludes sensitive clinical diagnoses.<br/><br/>We engineered booking-time features like booking lead time, patient age, distance from clinic, SMS reminder status, and prior visit history. We used a strict **chronological split** to ensure zero data leakage from future visits into training."* |
| **0:50 – 1:15**<br/>*(25s)* | **Cost Curve & Metrics**<br/>Point to the *'Cost Optimization Curve'* and metric cards (ROC-AUC 0.698, Recall 97.7%). | *"To handle the 4:1 class imbalance, we trained multiple models: baseline Logistic Regression, Balanced Random Forest, HistGradientBoosting, and Balanced Logistic Regression.<br/><br/>Because an empty doctor slot costs about \$180 while a quick phone call costs only \$15, missing a no-show is 12 times more expensive! By optimizing our decision threshold to **0.35**, our model catches **97.7% of all no-shows**, cutting wasted clinic costs by **76%**."* |
| **1:15 – 1:45**<br/>*(30s)* | **Live Demo: Single & Batch**<br/>1. Switch to *'Single Patient'* tab.<br/>2. Click **'Load High Risk Sample'** &rarr; **'Evaluate'**.<br/>3. Switch to *'Batch Schedule'* tab &rarr; Click **'Run Sample Clinic Day'**. | *"Now let's see the live web dashboard. If I load a high-risk sample—a 23-year-old booking 28 days ahead with past missed visits—the model outputs a **68% risk probability**, explaining the top risk drivers and recommending a direct telephone call 48 hours prior.<br/><br/>In the **Batch Scoring** tab, clinic staff can evaluate an entire day's appointment roster in one click to prioritize who needs reminders."* |
| **1:45 – 2:00**<br/>*(15s)* | **Governance Tab & Closing**<br/>Click *'Governance & Limitations'* tab, then show VS Code / terminal briefly. | *"Finally, this system is strictly a decision-support aid—it is never used to deny care or triage medical urgency. The project includes 12 automated unit tests passing at 100% and is published on GitHub. Thank you!"* |

---

## 4. Quick 3-Point Mental Summary (If You Need a Quick Reminder)
1. **The Problem:** Hospital no-shows waste physician slots (\$180 loss). A reminder phone call is cheap (\$15).
2. **Data & Model:** 15,000 de-identified records, no clinical data. Balanced Logistic Regression catches 97.7% of no-shows.
3. **App Demo:** Live Flask dashboard with single evaluation, risk gauge, local explainability, and batch roster scoring.
