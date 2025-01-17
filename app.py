import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Load pre-trained model and encoders
model = load_model('./churn_model.h5')

with open("label_encorder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("onehot_encorder_geo.pkl", "rb") as file:
    onehot_encorder_geo = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Streamlit app
st.title("Customer Churn Prediction")

# User inputs
geography = st.selectbox("Geography", onehot_encorder_geo.categories_[0])  # Example categories
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])  # 0 for No, 1 for Yes
is_active_member = st.selectbox("Is Active Member", [0, 1])  # 0 for No, 1 for Yes

# Button for prediction
if st.button("Predict"):
    # Prepare the input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary],
        'Geography': [geography],
    })

    # Encode 'Gender'
    gender_encoded = label_encoder_gender.transform(input_data["Gender"])

    # Encode 'Geography'
    geo_encoded = onehot_encorder_geo.transform(input_data[["Geography"]])
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encorder_geo.get_feature_names_out(['Geography'])
    )

    # Merge encoded data
    input_df = pd.DataFrame({
        'CreditScore': input_data['CreditScore'],
        'Gender': gender_encoded,
        'Age': input_data['Age'],
        'Tenure': input_data['Tenure'],
        'Balance': input_data['Balance'],
        'NumOfProducts': input_data['NumOfProducts'],
        'HasCrCard': input_data['HasCrCard'],
        'IsActiveMember': input_data['IsActiveMember'],
        'EstimatedSalary': input_data['EstimatedSalary'],
    })

    # Combine with geo_encoded_df
    input_df = pd.concat([input_df, geo_encoded_df], axis=1)

    # Scale the input data
    scaled_data = scaler.transform(input_df)

    # Make predictions
    prediction = model.predict(scaled_data)
    st.write(f"Churn Prediction Probability: {prediction[0][0]:.2f}")

    if prediction[0][0] > 0.5:
        st.write("Customer is likely to churn")
    else:
        st.write("Customer is likely not to churn")
