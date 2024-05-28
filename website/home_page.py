import joblib
import streamlit as st
import pandas as pd
import tensorflow as tf
import warnings
import pandas as pd
import joblib
import numpy as np
warnings.filterwarnings("ignore")

st.header('Customer Segmentation', divider='blue')

logistic = joblib.load("models/logistic.pkl")

linear_scaler = joblib.load("models/scaler_svm_linear.pkl")
svmLinear = joblib.load("models/svm_linear_model.pkl")

rbf_scaler = joblib.load("models/scaler_svm_rbf.pkl")
svmRBF = joblib.load("models/svm_rbf_model.pkl")

nn = tf.keras.models.load_model('models/nn.keras')

gender_mapping = {0: "Male", 1: "Female"}
gender = st.radio("Gender", options=list(gender_mapping.values()))

if gender == "Male":
    gender_code = 0
else:
    gender_code = 1

age = st.slider("Age", 1, 70, 1)
annual_income = st.slider("Annual Income (k$)", 1, 150, 1)

input = {
    'Gender': gender_code,
    'Age': age,
    'Annual Income (k$)': annual_income,
}


df = pd.DataFrame([input])

logicPred = logistic.predict(df.copy())

linear_data = linear_scaler.transform(df.copy())
linear_pred = svmLinear.predict(linear_data)

rbf_data = rbf_scaler.transform(df.copy())
rbf_pred = svmRBF.predict(rbf_data)

nnPred = nn.predict(df)

st.write(f"Logistic Predicted Spending Catagory: {logicPred[0]}")
st.write(f"Linear SVM Predicted Spending Catagory: {linear_pred[0]}")
st.write(f"Non-Linear SVM Predicted Spending Catagory: {rbf_pred[0]}")

categories = ['Low', 'Medium', 'High']
nn_category_index = np.argmax(nnPred[0])
nn_category = categories[nn_category_index]
st.write(f"Neural Network Predicted Spending Category: {nn_category}")
