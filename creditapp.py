import joblib
import pandas as pd
import streamlit as st

# 1. Load the model
# Using 'r' before the string to handle backslashes in Windows paths
model_path = r"C:\Users\hp\Desktop\ml_mini_project\softgrowth.py\Extratree_model.pkl"
model = joblib.load(model_path)

# 2. Load the encoders
# Ensure these names match your saved .pkl filenames exactly
columns_to_encode = ["Sex", "Saving accounts", "Housing", "Checking account"]
encoder = {}
for col in columns_to_encode:
    encoder[col] = joblib.load(f"label_encoder_{col}.pkl")

# --- UI Layout ---
st.title("Credit Risk Prediction App")
st.write("Enter the applicant information below to predict credit risk.")

# Inputs
age = st.number_input("Age", min_value=18, max_value=80, value=30)
sex = st.selectbox("Sex", ["male", "female"])
job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1)
housing = st.selectbox("Housing", ["own", "rent", "free"])
saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "rich"])
checking_accounts = st.selectbox("Checking Account", ["little", "moderate", "rich"])
credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
duration = st.number_input("Duration (months)", min_value=1, value=12)

# --- The Solution: Matching Feature Names and Order ---
# In your Colab, the order was: Age, Credit amount, Duration, Housing, Saving accounts, Sex, Job, Checking account
if st.button("Predict Risk"):
    input_dict = {
        "Age": age,
        "Credit amount": credit_amount,
        "Duration": duration,
        "Housing": encoder["Housing"].transform([housing])[0],
        "Saving accounts": encoder["Saving accounts"].transform([saving_accounts])[0],
        "Sex": encoder["Sex"].transform([sex])[0],
        "Job": job,
        "Checking account": encoder["Checking account"].transform([checking_accounts])[0]
    }
    
    # Create DataFrame from the dictionary
    input_data = pd.DataFrame([input_dict])

    # Prediction
    pred = model.predict(input_data)[0]

    if pred == "good" or pred == 1:
        st.success("The predicted credit risk is: **GOOD**")
    else:
        st.error("The predicted credit risk is: **BAD**")