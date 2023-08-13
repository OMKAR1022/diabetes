import joblib
import streamlit as st
import pandas as pd

# Load the pre-trained RandomForestClassifier model
rf = joblib.load('model_joblib.pkl')

# Load the dataset
df = pd.read_csv('diabetes_dataset-2.csv')

# Streamlit UI
st.title('Diabetes Checkup')
st.sidebar.header('DATA OF PATIENT')

# Function to collect user report data
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('BloodPressure', 0, 122, 70)
    skinthickness = st.sidebar.slider('SkinThickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('DiabetesPedigreeFunction', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)

    user_report_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# Collect user data
user_data = user_report()
st.subheader('Data Of Patient')
st.write(user_data)

# Predict using the loaded model
user_result = rf.predict(user_data)

# Display prediction result
st.title('Visualised Patient Report')
st.bar_chart(df)
color = 'blue' if user_result[0] == 0 else 'red'
st.subheader('Your Report: ')
output = 'You are not Diabetic' if user_result[0] == 0 else 'You are Diabetic'
st.title(output)
