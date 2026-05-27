# vehicle-loan-default-prediction
End-to-End Credit Risk ML System — Logistic Regression, Random Forest, XGBoost, SHAP, Optuna, Streamlit
# Vehicle Loan Default Risk Prediction System

## Business Problem
Financial institutions face massive NPA losses when vehicle 
loans default. This end-to-end ML system predicts first EMI 
default risk using 2,33,154 historical L&T Finance loan records 
enabling data-driven loan approval decisions.
## App Screenshots

### Single Applicant Assessment
![Tab1](screenshots/tab1.png)

### Batch Portfolio Assessment  
![Tab2](screenshots/tab2.png)
## Live Demo
Run locally using steps below.

## Tech Stack
- **Python** — Data cleaning and feature engineering
- **XGBoost** — Champion classification model
- **Optuna** — Bayesian hyperparameter optimization
- **SHAP** — Model explainability
- **Streamlit** — Interactive risk assessment app

## Project Structure
vehicle-loan-default/
├── data/                    # Dataset (not included)
├── notebooks/               # EDA notebook
├── src/                     # Processing scripts
│   ├── 01_data_cleaning.py
│   ├── 02_feature_engineering.py
│   ├── 04_model_training.py
│   ├── 05_optuna_tuning.py
│   └── 06_shap_explainability.py
├── models/                  # Saved models
├── app/
│   └── streamlit_app.py     # Streamlit app
├── reports/                 # Charts and outputs
└── README.md


## Key Results
| Model | Recall | ROC-AUC | PR-AUC |
|---|---|---|---|
| Logistic Regression | 0.6384 | 0.6306 | 0.3128 |
| Random Forest | 0.5237 | 0.6467 | 0.3292 |
| XGBoost (Champion) | 0.6131 | 0.6600 | 0.3433 |

## Key Insights From EDA
- Self employed customers default 2.34% more than salaried
- Credit score category 7 (Very High Risk) defaults at 31%
- Customers with document score 3 default least at 16.72%
- LTV ratio is the strongest default predictor (SHAP: 0.193)

## Feature Engineering
Created 7 domain-driven features:
1. age_at_disbursal — customer age at loan time
2. loan_to_asset_ratio — financing proportion
3. doc_score — identity verification strength
4. total_existing_debt — current debt burden
5. pri_overdue_ratio — repayment quality ratio
6. total_emi_burden — monthly obligation
7. credit_utilization — financial stretch indicator

## SHAP Top Risk Factors
1. LTV Ratio (0.193) — strongest predictor
2. Credit Bureau Score (0.134)
3. Geographic Location/Pincode (0.121)
4. Disbursed Amount (0.119)
5. Employment Type (0.090)

## App Features
**Tab 1 — Single Applicant Assessment**
- Dropdown form with all loan application fields
- Instant risk verdict (Low/Medium/High)
- SHAP waterfall explanation per applicant

**Tab 2 — Batch Portfolio Assessment**
- CSV upload with column validation guard
- Portfolio risk summary metrics
- Downloadable Excel report with 3 sheets

## How to Run
```bash
# Clone repository
git clone https://github.com/koteshwarkatkuri/vehicle-loan-default-prediction.git
cd vehicle-loan-default-prediction

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app/streamlit_app.py
```

## Author
Koteshwar Katkuri | Mumbai, India
[LinkedIn] | [GitHub]
