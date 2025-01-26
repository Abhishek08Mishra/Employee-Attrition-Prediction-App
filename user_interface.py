import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler


def display_intro():
    """
    Display the app's title and introductory information.
    """
    st.title("Employee Attrition Prediction 🎯")
    st.markdown("""
    ## Welcome to the Employee Attrition Prediction App 🌟
    This application predicts whether an employee is likely to **leave** 🚶‍♂️ or **stay** 💼 based on various factors such as:
    - Age 👶/👵
    - Department 🏢
    - Job Role 👩‍💼/👨‍💼
    - Monthly Income 💰
    - Gender 🚻 and more... 🎯

    ### About the Model:
    - **Algorithm:** Logistic Regression 🤖
    - **Trained on:** HR Analytics Dataset 📊
    - **Purpose:** To help HR professionals identify potential attrition risks ⚠️ and take proactive measures 🔍.
    """)

# Input form for employee data
def user_input_form():
    """
    Create an input form for users to enter employee data.
    """
    st.sidebar.header("Input Employee Details")

    # Input fields
    age = st.sidebar.number_input("Age 👶/👵", min_value=18, max_value=100)
    department = st.sidebar.selectbox("Department 🏢", ["Finance", "HR", "IT", "Marketing", "Sales"])
    job_role = st.sidebar.selectbox("Job Role 👩‍💼/👨‍💼", ["Analyst", "Clerk", "Executive", "Manager", "Specialist"])
    years_at_company = st.sidebar.number_input("Years at Company 🏢", min_value=0, max_value=40)
    monthly_income = st.sidebar.number_input("Monthly Income 💰", min_value=1000, max_value=500000)
    gender = st.sidebar.selectbox("Gender 🚻", ['Male', 'Female'])
    overtime = st.sidebar.selectbox("Overtime ⏰", ['Yes', 'No'])

    # Combine inputs into a dictionary
    user_data = {
        "Age": age,
        "Department": department,
        "JobRole": job_role,
        "YearsAtCompany": years_at_company,
        "MonthlyIncome": monthly_income,
        "Gender": gender,
        "Overtime": overtime
    }
    
    # Convert dictionary to DataFrame
    user_input = pd.DataFrame(user_data, index=[0])
    return user_input

# Preprocess user input
def preprocess_input(user_input):
    """
    Encode and scale user input data.
    """
    # Encoding mappings
    department_map = {"Finance" : 0,"HR" : 1, "IT" : 2, "Marketing" : 3, "Sales" : 4}
    job_role_map = {"Analyst": 0, "Clerk": 1, "Executive": 2, "Manager": 3, "Specialist": 4}
    gender_map = {'Male': 1, 'Female': 0}
    overtime_map = {'Yes': 1, 'No': 0}

    # Encode categorical variables
    user_input['Department_Encoded'] = user_input['Department'].map(department_map)
    user_input['JobRole_Encoded'] = user_input['JobRole'].map(job_role_map)
    user_input['Gender_Encoded'] = user_input['Gender'].map(gender_map)
    user_input['Overtime_Encoded'] = user_input['Overtime'].map(overtime_map)

    # Drop original categorical columns
    encoded_features = user_input.drop(columns=["Department", "JobRole", "Gender", "Overtime"])

    # Scale the numerical data
    scaler = StandardScaler()
    scaled_input = scaler.fit_transform(encoded_features)

    return scaled_input

# Make predictions
def make_prediction(model, user_input):
    """
    Use the trained model to predict employee attrition.
    """
    prediction = model.predict(user_input)
    if prediction == 1:
        st.write("### Prediction: The employee is likely to **leave** the company.")
    else:
        st.write("### Prediction: The employee is likely to **stay** with the company.")