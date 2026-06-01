import streamlit as st
import numpy as np
import pickle
import pandas as pd
# Load model
#model = pickle.load(open("artifacts/loan_rf_model.pkl", "rb"))
#model = pickle.load(open("artifacts/loan_rf_model.pkl", "rb"))
model = pickle.load(open("artifacts/loan_rf_model.pkl", "rb"))

with open("artifacts/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

print(type(model))

if hasattr(model, "n_features_in_"):
    print("Expected Features:", model.n_features_in_)

st.title("Loan Prediction App")

st.write("Enter customer details below:")

# ============================================================
# BASIC INFO
# ============================================================

age = st.number_input("Age", 18, 100, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married"])
education = st.selectbox("Education Level", ["High School", "Graduate", "Post Graduate"])
employment = st.selectbox("Employment Type", ["Salaried", "Self Employed", "Unemployed"])
city_tier = st.selectbox("City Tier", [1, 2, 3])

# ============================================================
# FINANCIAL INFO
# ============================================================

monthly_income = st.number_input("Monthly Income", 0)
monthly_expenses = st.number_input("Monthly Expenses", 0)
savings_balance = st.number_input("Savings Balance", 0)
existing_loan = st.number_input("Existing Loan Amount", 0)

credit_util = st.number_input("Credit Card Utilization", 0.0, 1.0)
debt_income = st.number_input("Debt to Income Ratio", 0.0, 1.0)

# ============================================================
# CREDIT INFO
# ============================================================

open_accounts = st.number_input("Number of Open Accounts", 0)
credit_cards = st.number_input("Number of Credit Cards", 0)
loan_amount = st.number_input("Loan Amount Requested", 0)
loan_tenure = st.number_input("Loan Tenure (Months)", 0)

interest_rate = st.number_input("Interest Rate", 0.0)
emi = st.number_input("EMI", 0)
emi_income_ratio = st.number_input("EMI to Income Ratio", 0.0)

credit_score = st.number_input("Credit Score", 300, 900, 650)
prev_defaults = st.number_input("Previous Defaults", 0)
delay_days = st.number_input("Payment Delay Days", 0)
credit_history = st.number_input("Credit History Years", 0)

inquiries = st.number_input("Recent Credit Inquiries", 0)
login_freq = st.number_input("App Login Frequency", 0)
doc_score = st.number_input("Document Verification Score", 0)

address_stability = st.number_input("Address Stability Years", 0)
job_stability = st.number_input("Job Stability Years", 0)

# ============================================================
# OTHER PRODUCTS (BOOLEAN FEATURES)
# ============================================================

credit_card = st.selectbox("Has Credit Card", [0, 1])
insurance = st.selectbox("Has Insurance", [0, 1])
mutual_fund = st.selectbox("Has Mutual Fund", [0, 1])
fixed_deposit = st.selectbox("Has Fixed Deposit", [0, 1])
gold_loan = st.selectbox("Has Gold Loan", [0, 1])
vehicle_loan = st.selectbox("Has Vehicle Loan", [0, 1])
home_loan = st.selectbox("Has Home Loan", [0, 1])
bnpl = st.selectbox("Has BNPL", [0, 1])
overdraft = st.selectbox("Has Overdraft", [0, 1])

# ============================================================
# ENCODING SIMPLE CATEGORICALS
# ============================================================

gender = 1 if gender == "Male" else 0
marital_status = 1 if marital_status == "Married" else 0

# ============================================================
# FINAL INPUT VECTOR
# ============================================================
user_data = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Marital_Status": marital_status,
    "Education": education,
    "Employment": employment,
    "City_Tier": city_tier,
    "Monthly_Income": monthly_income,
    "Monthly_Expenses": monthly_expenses,
    "Savings_Balance": savings_balance,
    "Existing_Loan": existing_loan,
    "Credit_Utilization": credit_util,
    "Debt_Income": debt_income,
    "Open_Accounts": open_accounts,
    "Credit_Cards": credit_cards,
    "Loan_Amount": loan_amount,
    "Loan_Tenure": loan_tenure,
    "Interest_Rate": interest_rate,
    "EMI": emi,
    "EMI_Income_Ratio": emi_income_ratio,
    "Credit_Score": credit_score,
    "Previous_Defaults": prev_defaults,
    "Delay_Days": delay_days,
    "Credit_History": credit_history,
    "Inquiries": inquiries,
    "Login_Frequency": login_freq,
    "Document_Score": doc_score,
    "Address_Stability": address_stability,
    "Job_Stability": job_stability,
    "Credit_Card": credit_card,
    "Insurance": insurance,
    "Mutual_Fund": mutual_fund,
    "Fixed_Deposit": fixed_deposit,
    "Gold_Loan": gold_loan,
    "Vehicle_Loan": vehicle_loan,
    "Home_Loan": home_loan,
    "BNPL": bnpl,
    "Overdraft": overdraft
}])

user_data = pd.get_dummies(user_data, drop_first=True)

user_data = user_data.reindex(
    columns=feature_columns,
    fill_value=0
)



# ============================================================
# PREDICTION
# ============================================================

if st.button("Predict Loan Eligibility"):
    prediction = model.predict(user_data)

    if prediction[0] == 1:
        st.error(" High Risk: Likely to take Personal Loan")
    else:
        st.success(" Low Risk: Unlikely to take Personal Loan")