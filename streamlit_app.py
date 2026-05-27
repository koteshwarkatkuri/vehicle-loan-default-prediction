# ================================================
# Vehicle Loan Default Risk Prediction
# Streamlit Application
# Author: Koteshwar Katkuri
# ================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
from datetime import date
import io
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------
st.set_page_config(
    page_title="Vehicle Loan Risk Assessor",
    page_icon="🚗",
    layout="wide"
)

# ------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .risk-high {
        background-color: #FF4B4B;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #FFA500;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .risk-low {
        background-color: #00CC44;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# LOAD MODEL AND EXPLAINER
# ------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load('C:/Users/kotes/OneDrive/Documents/Desktop/cv_loan_project/models/xgboost_best_final.pkl')
    explainer = joblib.load('C:/Users/kotes/OneDrive/Documents/Desktop/cv_loan_project/notebooks/models/shap_explainer.pkl')
    return model, explainer

model, explainer = load_model()

# ------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------
def get_risk_category(probability):
    if probability < 0.30:
        return "LOW RISK", "✅ RECOMMEND APPROVE", "risk-low"
    elif probability < 0.60:
        return "MEDIUM RISK", "⚠️ RECOMMEND REVIEW", "risk-medium"
    else:
        return "HIGH RISK", "❌ RECOMMEND REJECT", "risk-high"

def calculate_age(dob):
    today = date.today()
    age = today.year - dob.year
    if (today.month, today.day) < (dob.month, dob.day):
        age -= 1
    return age

# ------------------------------------------------
# APP TITLE
# ------------------------------------------------
st.title("🚗 Vehicle Loan Default Risk Assessment System")
st.markdown("**Powered by XGBoost + SHAP Explainability | L&T Finance Dataset**")
st.markdown("---")

# ------------------------------------------------
# TWO TABS
# ------------------------------------------------
tab1, tab2 = st.tabs([
    "🧑 Single Applicant Assessment",
    "📊 Batch Portfolio Assessment"
])

# ================================================
# TAB 1 — SINGLE APPLICANT
# ================================================
with tab1:
    st.subheader("Enter Applicant Details")
    st.markdown("Fill in the loan application details below:")

    # Three columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Loan Details**")
        disbursed_amount = st.number_input(
            "Loan Amount (₹)",
            min_value=13000,
            max_value=1000000,
            value=50000,
            step=1000
        )
        asset_cost = st.number_input(
            "Asset Cost (₹)",
            min_value=37000,
            max_value=1628992,
            value=70000,
            step=1000
        )
        ltv = st.number_input(
            "LTV Ratio (%)",
            min_value=10.0,
            max_value=95.0,
            value=75.0,
            step=0.5
        )
        dob = st.date_input(
            "Date of Birth",
            min_value=date(1960, 1, 1),
            max_value=date(2005, 1, 1),
            value=date(1990, 1, 1)
        )

    with col2:
        st.markdown("**Credit Information**")
        perform_cns_score = st.number_input(
            "Credit Bureau Score",
            min_value=0,
            max_value=900,
            value=600,
            step=10
        )
        credit_category = st.selectbox(
            "Credit Score Category",
            options=[
                "No Bureau History Available",
                "Not Scored: No Activity seen on the customer (Inactive)",
                "Not Scored: No Updates available in last 36 months",
                "Not Scored: Only a Guarantor",
                "Not Scored: More than 50 active Accounts found",
                "Not Scored: Sufficient History Not Available",
                "Not Scored: Not Enough Info available on the customer",
                "M-Very High Risk",
                "L-Very High Risk",
                "K-High Risk",
                "J-High Risk",
                "I-Medium Risk",
                "H-Medium Risk",
                "G-Low Risk",
                "F-Low Risk",
                "E-Low Risk",
                "D-Very Low Risk",
                "C-Very Low Risk",
                "B-Very Low Risk",
                "A-Very Low Risk"
            ]
        )
        employment_type = st.selectbox(
            "Employment Type",
            options=["Salaried", "Self Employed"]
        )
        no_of_inquiries = st.number_input(
            "Credit Inquiries (Last 6 months)",
            min_value=0,
            max_value=20,
            value=1,
            step=1
        )

    with col3:
        st.markdown("**Account History**")
        pri_no_of_accts = st.number_input(
            "Primary Accounts",
            min_value=0,
            max_value=50,
            value=2,
            step=1
        )
        pri_active_accts = st.number_input(
            "Active Primary Accounts",
            min_value=0,
            max_value=50,
            value=1,
            step=1
        )
        pri_overdue_accts = st.number_input(
            "Overdue Primary Accounts",
            min_value=0,
            max_value=20,
            value=0,
            step=1
        )
        delinquent_accts = st.number_input(
            "Delinquent Accounts (Last 6 months)",
            min_value=0,
            max_value=20,
            value=0,
            step=1
        )
        new_accts = st.number_input(
            "New Accounts (Last 6 months)",
            min_value=0,
            max_value=20,
            value=0,
            step=1
        )

    st.markdown("---")

    # ------------------------------------------------
    # ASSESS RISK BUTTON
    # ------------------------------------------------
    if st.button("🚀 Assess Risk", use_container_width=True):

        # Calculate engineered features
        age_at_disbursal = calculate_age(dob)
        loan_to_asset_ratio = round(disbursed_amount / asset_cost, 4)
        pri_overdue_ratio = round(
            pri_overdue_accts / max(pri_no_of_accts, 1), 4
        )

        # Credit score category mapping
        credit_score_map = {
            'No Bureau History Available': 0,
            'Not Scored: No Activity seen on the customer (Inactive)': 1,
            'Not Scored: No Updates available in last 36 months': 2,
            'Not Scored: Only a Guarantor': 3,
            'Not Scored: More than 50 active Accounts found': 4,
            'Not Scored: Sufficient History Not Available': 5,
            'Not Scored: Not Enough Info available on the customer': 6,
            'M-Very High Risk': 7,
            'L-Very High Risk': 8,
            'K-High Risk': 9,
            'J-High Risk': 10,
            'I-Medium Risk': 11,
            'H-Medium Risk': 12,
            'G-Low Risk': 13,
            'F-Low Risk': 14,
            'E-Low Risk': 15,
            'D-Very Low Risk': 16,
            'C-Very Low Risk': 17,
            'B-Very Low Risk': 18,
            'A-Very Low Risk': 19
        }
        credit_score_category = credit_score_map[credit_category]
        emp_encoded = 0 if employment_type == "Salaried" else 1

        # Build input — must match training feature order
        # Load feature names
        import json
        with open('C:/Users/kotes/OneDrive/Documents/Desktop/cv_loan_project/notebooks/models/feature_names.json', 'r') as f:
            feature_names = json.load(f)

        # Create input with zeros for all features
        input_dict = {feat: 0 for feat in feature_names}

        # Fill known values
        input_dict['disbursed_amount'] = disbursed_amount
        input_dict['asset_cost'] = asset_cost
        input_dict['ltv'] = ltv
        input_dict['Employment.Type'] = emp_encoded
        input_dict['PERFORM_CNS.SCORE'] = perform_cns_score
        input_dict['CREDIT_SCORE_CATEGORY'] = credit_score_category
        input_dict['NO.OF_INQUIRIES'] = no_of_inquiries
        input_dict['PRI.NO.OF.ACCTS'] = pri_no_of_accts
        input_dict['PRI.ACTIVE.ACCTS'] = pri_active_accts
        input_dict['PRI.OVERDUE.ACCTS'] = pri_overdue_accts
        input_dict['DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS'] = delinquent_accts
        input_dict['NEW.ACCTS.IN.LAST.SIX.MONTHS'] = new_accts
        input_dict['age_at_disbursal'] = age_at_disbursal
        input_dict['loan_to_asset_ratio'] = loan_to_asset_ratio
        input_dict['pri_overdue_ratio'] = pri_overdue_ratio

        # Create dataframe
        input_df = pd.DataFrame([input_dict])

        # Predict
        probability = model.predict_proba(input_df)[0][1]
        risk_label, action, css_class = get_risk_category(probability)

        # ------------------------------------------------
        # SHOW RESULTS
        # ------------------------------------------------
        st.markdown("---")
        st.subheader("Risk Assessment Result")

        # Risk verdict
        st.markdown(
            f'<div class="{css_class}">'
            f'{risk_label} — {action}<br>'
            f'Default Probability: {probability:.1%}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown("---")

        # Key metrics
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Default Probability", f"{probability:.1%}")
        with col_b:
            st.metric("LTV Ratio", f"{ltv:.1f}%")
        with col_c:
            st.metric("Age at Disbursal", f"{age_at_disbursal} years")
        with col_d:
            st.metric("Overdue Ratio", f"{pri_overdue_ratio:.1%}")

        # ------------------------------------------------
        # SHAP EXPLANATION
        # ------------------------------------------------
        st.markdown("---")
        st.subheader("Why This Decision? — SHAP Explanation")

        shap_values_single = explainer.shap_values(input_df)

        fig, ax = plt.subplots(figsize=(12, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_single[0],
                base_values=explainer.expected_value,
                data=input_df.iloc[0],
                feature_names=feature_names
            ),
            show=False,
            max_display=10
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.info(
            "🔴 Red bars push toward DEFAULT risk. "
            "🔵 Blue bars push toward SAFE. "
            "Longer bar = stronger impact on decision."
        )

# ================================================
# TAB 2 — BATCH ASSESSMENT
# ================================================
with tab2:
    st.subheader("Batch Portfolio Risk Assessment")
    st.markdown(
        "Upload a CSV file with multiple applicants "
        "to assess portfolio risk at once."
    )

    # Column guard info
    st.info(
        "📋 Your CSV must contain these required columns: "
        "disbursed_amount, asset_cost, ltv, "
        "PERFORM_CNS.SCORE, Employment.Type, "
        "PRI.NO.OF.ACCTS, PRI.OVERDUE.ACCTS, "
        "NO.OF_INQUIRIES"
    )

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=['csv']
    )

    if uploaded_file is not None:
        try:
            # Load uploaded file
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded: {batch_df.shape[0]} applicants")
            st.dataframe(batch_df.head(3))

            # Column guard validation
            required_cols = [
                'disbursed_amount', 'asset_cost', 'ltv',
                'PERFORM_CNS.SCORE', 'Employment.Type',
                'PRI.NO.OF.ACCTS', 'PRI.OVERDUE.ACCTS',
                'NO.OF_INQUIRIES'
            ]

            missing_cols = [
                col for col in required_cols
                if col not in batch_df.columns
            ]

            if missing_cols:
                st.error(
                    f"Missing required columns: {missing_cols}. "
                    f"Please check your CSV file."
                )
            else:
                if st.button(
                    "🚀 Run Batch Assessment",
                    use_container_width=True
                ):
                    with st.spinner("Assessing risk for all applicants..."):

                        # Load feature names
                        with open('models/feature_names.json', 'r') as f:
                            feature_names = json.load(f)

                        # Prepare batch input
                        batch_input = pd.DataFrame(
                            0,
                            index=range(len(batch_df)),
                            columns=feature_names
                        )

                        # Fill available columns
                        for col in feature_names:
                            if col in batch_df.columns:
                                batch_input[col] = batch_df[col].values

                        # Predict probabilities
                        probabilities = model.predict_proba(
                            batch_input
                        )[:, 1]

                        # Add results to dataframe
                        batch_df['default_probability'] = probabilities.round(4)
                        batch_df['risk_category'] = pd.cut(
                            probabilities,
                            bins=[0, 0.30, 0.60, 1.0],
                            labels=['LOW RISK', 'MEDIUM RISK', 'HIGH RISK']
                        )
                        batch_df['recommendation'] = batch_df[
                            'risk_category'
                        ].map({
                            'LOW RISK': 'APPROVE',
                            'MEDIUM RISK': 'REVIEW',
                            'HIGH RISK': 'REJECT'
                        })

                    # Show summary
                    st.markdown("---")
                    st.subheader("Portfolio Risk Summary")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        total = len(batch_df)
                        st.metric("Total Applicants", total)
                    with col2:
                        high_risk = (
                            batch_df['risk_category'] == 'HIGH RISK'
                        ).sum()
                        st.metric(
                            "High Risk",
                            high_risk,
                            delta=f"{high_risk/total:.1%}"
                        )
                    with col3:
                        medium_risk = (
                            batch_df['risk_category'] == 'MEDIUM RISK'
                        ).sum()
                        st.metric(
                            "Medium Risk",
                            medium_risk,
                            delta=f"{medium_risk/total:.1%}"
                        )
                    with col4:
                        low_risk = (
                            batch_df['risk_category'] == 'LOW RISK'
                        ).sum()
                        st.metric(
                            "Low Risk",
                            low_risk,
                            delta=f"{low_risk/total:.1%}"
                        )

                    # Show results table
                    st.markdown("---")
                    st.subheader("High Risk Applicants")
                    high_risk_df = batch_df[
                        batch_df['risk_category'] == 'HIGH RISK'
                    ].sort_values(
                        'default_probability',
                        ascending=False
                    )
                    st.dataframe(high_risk_df)

                    # Download Excel report
                    st.markdown("---")
                    st.subheader("Download Risk Report")

                    output = io.BytesIO()
                    with pd.ExcelWriter(
                        output,
                        engine='openpyxl'
                    ) as writer:
                        # Sheet 1 — All results
                        batch_df.to_excel(
                            writer,
                            sheet_name='All Applicants',
                            index=False
                        )
                        # Sheet 2 — High risk only
                        high_risk_df.to_excel(
                            writer,
                            sheet_name='High Risk Applicants',
                            index=False
                        )
                        # Sheet 3 — Summary
                        summary = pd.DataFrame({
                            'Category': [
                                'Total', 'High Risk',
                                'Medium Risk', 'Low Risk'
                            ],
                            'Count': [
                                total, high_risk,
                                medium_risk, low_risk
                            ],
                            'Percentage': [
                                '100%',
                                f'{high_risk/total:.1%}',
                                f'{medium_risk/total:.1%}',
                                f'{low_risk/total:.1%}'
                            ]
                        })
                        summary.to_excel(
                            writer,
                            sheet_name='Summary',
                            index=False
                        )

                    st.download_button(
                        label="📥 Download Excel Risk Report",
                        data=output.getvalue(),
                        file_name="vehicle_loan_risk_report.xlsx",
                        mime="application/vnd.ms-excel",
                        use_container_width=True
                    )
                    st.success(
                        "Report ready! Click above to download."
                    )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# ------------------------------------------------
# FOOTER
# ------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    Vehicle Loan Default Risk Assessment System |
    Built by Koteshwar Katkuri |
    Powered by XGBoost + SHAP
    </div>
    """,
    unsafe_allow_html=True
)