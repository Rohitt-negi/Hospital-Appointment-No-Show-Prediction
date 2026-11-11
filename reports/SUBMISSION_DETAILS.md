# Algions AI Labs - Assessment Submission Details Form

This document contains all official submission form answers prepared for **Rohit Singh Negi** for the AI & Machine Learning Advanced Assessment.

---

### 1. Candidate Identification
| Field | Candidate Response |
| :--- | :--- |
| **Full Name** | Rohit Singh Negi |
| **Registration Number** | 12311143 |
| **University Email** | *(Enter your institutional student email address)* |

---

### 2. Assignment Confirmation
| Field | Candidate Response |
| :--- | :--- |
| **Internship Role** | AI & Machine Learning |
| **Assigned Project Title** | Hospital Appointment No-Show Prediction |
| **Submission Deadline** | As specified in official invitation |

---

### 3. Code and Version Information
| Field | Candidate Response |
| :--- | :--- |
| **Repository URL** | https://github.com/Rohitt-negi/Hospital-Appointment-No-Show-Prediction |
| **Primary Branch** | `main` |
| **Final Commit ID** | *(Run `git rev-parse HEAD` and insert commit hash)* |
| **Reviewer Access** | Public (or provide read access to reviewer handle if private) |

---

### 4. Demonstration and Interfaces
| Field | Candidate Response |
| :--- | :--- |
| **Demonstration Video Link** | *(Insert Google Drive / Loom / YouTube unlisted video link, 3-5 minutes)* |
| **Live Demo URL** | Local Demonstration on `http://127.0.0.1:5001` (Complete instructions provided in README) |
| **API Endpoints** | `POST /predict`, `POST /batch-predict`, `GET /health`, `GET /metrics` |

---

### 5. Documentation and Technical Architecture
| Field | Candidate Response |
| :--- | :--- |
| **Technical Report Link** | `reports/TECHNICAL_REPORT.md` (Included in repository root) |
| **Jupyter Notebook Link** | `notebooks/01_hospital_no_show_analysis.ipynb` |
| **Architecture Diagram** | Documented via Mermaid in `README.md` and `reports/TECHNICAL_REPORT.md` |

---

### 6. Technology Stack and Limitations
| Field | Candidate Response |
| :--- | :--- |
| **Technology Stack** | Python 3.11+, Scikit-Learn, Pandas, NumPy, Flask, Flask-CORS, Joblib, Matplotlib, Seaborn, Unittest |
| **Core Models Compared** | 1. Dummy Classifier (Baseline)<br>2. Unweighted Logistic Regression (Baseline standard)<br>3. Balanced Logistic Regression (Class-weighted)<br>4. Balanced Random Forest Classifier<br>5. Histogram Gradient Boosting Classifier |
| **Chosen Primary Model** | Balanced Logistic Regression (`models/best_model.joblib`), optimized at operating decision threshold $\tau^* = 0.35$ |
| **Evaluation Metrics** | Test ROC-AUC: 0.6977 &bull; PR-AUC: 0.4365 &bull; Recall (@ $\tau=0.35$): 97.7% &bull; Cost per appointment: \$10.18 (76.3% reduction vs baseline) |
| **Known Assumptions & Limitations** | 1. Model operates strictly at booking confirmation time using administrative variables; clinical diagnosis and medical urgency are deliberately excluded.<br>2. External macro-shocks (severe weather storms, public transit strikes) cannot be predicted from booking features.<br>3. First-time patients have no historical record, leading to wider epistemic uncertainty. |

---

### 7. AI and Attribution Disclosure
| Field | Candidate Response |
| :--- | :--- |
| **AI Tools Used** | Antigravity AI Pair Programmer |
| **Purpose of Use** | Assisted with boilerplate scaffolding, rapid metric visualization code generation, and test suite structure. All domain framing, cost formulation, pipeline architecture, and validation logic were engineered, audited, and verified by Rohit Singh Negi. |
| **Dataset Source & Attribution** | Synthetic de-identified outpatient appointment dataset generated based on empirical distributions of the public Brazilian Vitória Hospital No-Shows benchmark (Kaggle). Strictly contains zero protected health information (PHI) or clinical data. |

---

### 8. Candidate Declarations
| Declaration | Status |
| :--- | :--- |
| **Originality** | I declare that this submission represents my own work and engineering design. |
| **Technical Walkthrough** | I am fully prepared to explain every function, class, metric, and architectural choice during the 15-minute live technical walkthrough. |
| **Live Debugging Readiness** | I am prepared to execute live modifications to the codebase (e.g. adding features, altering cost weights, testing endpoints) during the 10-minute live coding session. |
| **Data Privacy & Secrets** | I confirm that no real patient identifiable data, credentials, passwords, or live tokens are committed to this repository. |
| **Link Integrity** | All video and repository links provided are functional and publicly accessible to reviewers. |

