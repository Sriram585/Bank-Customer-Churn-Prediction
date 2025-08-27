import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler


with open('xgb_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


st.set_page_config(page_title="Bank Customer Churn Prediction", layout="wide")
st.title("Bank Customer Churn Prediction 🏦")
st.write("This app predicts whether a customer is likely to churn based on their details.")


st.sidebar.header("Customer Details")

def user_input_features():
    
    geography = st.sidebar.selectbox('Geography', ('France', 'Germany', 'Spain'))
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    age = st.sidebar.slider('Age', 18, 92, 38)
    tenure = st.sidebar.slider('Tenure (Years)', 0, 10, 5)
    balance = st.sidebar.number_input('Balance', 0.0, 250000.0, 75000.0)
    credit_score = st.sidebar.slider('Credit Score', 350, 850, 650)
    num_of_products = st.sidebar.slider('Number of Products', 1, 4, 1)
    has_cr_card = st.sidebar.selectbox('Has Credit Card?', ('Yes', 'No'))
    is_active_member = st.sidebar.selectbox('Is Active Member?', ('Yes', 'No'))
    estimated_salary = st.sidebar.number_input('Estimated Salary', 0.0, 200000.0, 100000.0)

    
    data = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': 1 if has_cr_card == 'Yes' else 0,
        'IsActiveMember': 1 if is_active_member == 'Yes' else 0,
        'EstimatedSalary': estimated_salary,
        'Geography_Germany': 1 if geography == 'Germany' else 0,
        'Geography_Spain': 1 if geography == 'Spain' else 0,
        'Gender_Male': 1 if gender == 'Male' else 0
    }
    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input_features()

st.subheader("Customer Input Details:")
st.write(input_df)

if st.button("Predict Churn"):
    
    numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    st.subheader("Prediction Result:")
    
    if prediction[0] == 1:
        st.error(f"**This customer is LIKELY to churn.**")
        st.write(f"**Churn Probability:** {prediction_proba[0][1] * 100:.2f}%")
    else:
        st.success(f"**This customer is UNLIKELY to churn.**")
        st.write(f"**Churn Probability:** {prediction_proba[0][1] * 100:.2f}%")