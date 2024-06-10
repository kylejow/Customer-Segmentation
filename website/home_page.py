import joblib
import streamlit as st
import pandas as pd
import tensorflow as tf
import warnings
import numpy as np
warnings.filterwarnings("ignore")

# Title and introduction
st.title("What Are Your Predicted Spending Habits?")
st.write("""
Using supervised learning and regression modeling, this website is able to predict consumer spending habits using customer gender, age, and annual income.
Below is an example of your predicted spending habits for your specific demographic:
""")

# Load models
logistic = joblib.load("models/logistic.pkl")
linear_scaler = joblib.load("models/scaler_svm_linear.pkl")
svmLinear = joblib.load("models/svm_linear_model.pkl")
rbf_scaler = joblib.load("models/scaler_svm_rbf.pkl")
svmRBF = joblib.load("models/svm_rbf_model.pkl")
decision = joblib.load("models/decision.pkl")
nn = tf.keras.models.load_model('models/nn.keras')

# User input
gender_mapping = {0: "Male", 1: "Female"}
gender = st.radio("Gender", options=list(gender_mapping.values()))
gender_code = 0 if gender == "Male" else 1

age = st.slider("Age", 1, 70, 1)
annual_income = st.slider("Annual Income (k$)", 1, 150, 1)

input = {'Gender': gender_code, 'Age': age, 'Annual Income (k$)': annual_income}
df = pd.DataFrame([input])

# Model predictions
logicPred = logistic.predict(df.copy())
linear_data = linear_scaler.transform(df.copy())
linear_pred = svmLinear.predict(linear_data)
rbf_data = rbf_scaler.transform(df.copy())
rbf_pred = svmRBF.predict(rbf_data)
decision_pred = decision.predict(df.copy())
nnPred = nn.predict(df.copy())

# Categories for display
categories = ['Low', 'Medium', 'High']
nn_category_index = np.argmax(nnPred[0])
nn_category = categories[nn_category_index]

# Display predictions
st.write(f"Neural Network Predicted Spending Category: {nn_category}")
st.write(f"Logistic Predicted Spending Category: {logicPred[0]}")
st.write(f"Linear SVM Predicted Spending Category: {linear_pred[0]}")
st.write(f"Non-Linear SVM Predicted Spending Category: {rbf_pred[0]}")
st.write(f"Decision Tree Predicted Spending Category: {categories[decision_pred[0]-1]}")

# Additional information
st.write("""
## Project Description

The goal of this project is to develop a predictive model that can accurately forecast customer spending habits based on demographic features such as gender, age, and annual income. This model will help businesses better understand their customers, tailor marketing strategies, and optimize product offerings.

### Dataset Features
- **Gender:** Categorical variable indicating the customer's gender.
- **Age:** Numerical variable representing the customer's age.
- **Annual Income:** Numerical variable indicating the customer's yearly income.
- **Spending Score:** A numerical value that quantifies a customer's spending behavior.

### Models Implemented
We implemented five specific models: Logistic Regression, Support Vector Machines for Linear and Radial Basis Function, Decision Trees, and Neural Network models. A challenge we faced included poor results from our Logistic Regression model. To address this, we implemented a neural network, adjusting the parameters using hyperparameter tuning, resulting in our best accuracy of 0.80!

### Team Members
Our team consisted of Swati Iyer, Kyle Jow, Daphne Loustalet, BaoTran Tran, and Josh Winerman. 
- **Swati:** Built and trained the logistic regression model.
- **Kyle:** Worked on the neural network and hyperparameter tuning.
- **Daphne:** Conducted exploratory data analysis to find correlations and formatted the dataset for creating the models.
- **BaoTran:** Worked on the support vector machines with both the linear and RBF kernels.
- **Josh:** Built the decision tree.

All five members worked on the written report and project presentation together.
""")
