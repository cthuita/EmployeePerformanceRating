import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the saved model
performance_rating_predictor = pickle.load(open("employeeperformancerating/notebook/performance_rating_prediction_model.pkl", 'rb'))

# Streamlit app title
st.title("Employee Performance Rating Predictor")

# Streamlit input form
age = st.number_input("Enter Age", min_value=18, max_value=70, value=30)
gender = st.selectbox("Enter Gender", options=["Female", "Male"])
educationbackground = st.selectbox("Enter Education Background", options=["Life Sciences", "Human Resources", "Medical", "Marketing", "Other", "Technical Degree"])
maritalstatus = st.selectbox("Enter Marital Status", options=["Single", "Married", "Divorced"])
empdepartment = st.selectbox("Enter Department", options=["Sales", "Research & Development", "Human Resources", "Data Science", "Development", "Finance"])
empjobrole = st.selectbox("Enter Job Role", options=["Sales Executive", "Research Scientist", "Sales Representative", "Healthcare Representative", "Research Director", "Laboratory Technician", "Manager", "Manager R&D", "Senior Manager R&D", "Human Resources", "Finance Manager", "Manufacturing Director"])
businesstravelfrequency = st.selectbox("Enter Business Travel Frequency", options=["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
distancefromhome = st.number_input("Enter Distance From Home", min_value=1, max_value=30)
empeducationlevel = st.number_input("Enter Education Level", min_value=1, max_value=5)
empenvironmentsatisfaction = st.number_input("Enter Environment Satisfaction (1-4)", min_value=1, max_value=4)
emphourlyrate = st.number_input("Enter Hourly Rate", min_value=1, max_value=100)
empjobinvolvement = st.number_input("Enter Job Involvement (1-4)", min_value=1, max_value=4)
empjoblevel = st.number_input("Enter Job Level (1-5)", min_value=1, max_value=5)
empjobsatisfaction = st.number_input("Enter Job Satisfaction (1-4)", min_value=1, max_value=4)
numcompaniesworked = st.number_input("Enter Number of Companies Worked", min_value=0, max_value=10)
overtime = st.selectbox("Enter OverTime", options=["No", "Yes"])
emplastsalaryhikepercent = st.number_input("Enter Last Salary Hike Percent", min_value=0, max_value=100)
emprelationshipsatisfaction = st.number_input("Enter Relationship Satisfaction (1-4)", min_value=1, max_value=4)
totalworkexperienceinyears = st.number_input("Enter Total Work Experience in Years", min_value=0, max_value=50)
trainingtimeslastyear = st.number_input("Enter Training Times Last Year", min_value=0, max_value=10)
empworklifebalance = st.number_input("Enter Work Life Balance (1-4)", min_value=1, max_value=4)
experienceyearsatthiscompany = st.number_input("Enter Experience Years at This Company", min_value=0, max_value=50)
experienceyearsincurrentrole = st.number_input("Enter Experience Years in Current Role", min_value=0, max_value=50)
yearssincelastpromotion = st.number_input("Enter Years Since Last Promotion", min_value=0, max_value=20)
yearswithcurrmanager = st.number_input("Enter Years With Current Manager", min_value=0, max_value=30)
attrition = st.selectbox("Enter Attrition", options=["No", "Yes"])

# Organize the input data into a dictionary
manual_test_input = {
    'age': [age],
    'gender': [gender],
    'educationbackground': [educationbackground],
    'maritalstatus': [maritalstatus],
    'empdepartment': [empdepartment],
    'empjobrole': [empjobrole],
    'businesstravelfrequency': [businesstravelfrequency],
    'distancefromhome': [distancefromhome],
    'empeducationlevel': [empeducationlevel],
    'empenvironmentsatisfaction': [empenvironmentsatisfaction],
    'emphourlyrate': [emphourlyrate],
    'empjobinvolvement': [empjobinvolvement],
    'empjoblevel': [empjoblevel],
    'empjobsatisfaction': [empjobsatisfaction],
    'numcompaniesworked': [numcompaniesworked],
    'overtime': [overtime],
    'emplastsalaryhikepercent': [emplastsalaryhikepercent],
    'emprelationshipsatisfaction': [emprelationshipsatisfaction],
    'totalworkexperienceinyears': [totalworkexperienceinyears],
    'trainingtimeslastyear': [trainingtimeslastyear],
    'empworklifebalance': [empworklifebalance],
    'experienceyearsatthiscompany': [experienceyearsatthiscompany],
    'experienceyearsincurrentrole': [experienceyearsincurrentrole],
    'yearssincelastpromotion': [yearssincelastpromotion],
    'yearswithcurrmanager': [yearswithcurrmanager],
    'attrition': [attrition]
}

# Convert the dictionary to a DataFrame
df_manual_test = pd.DataFrame(manual_test_input)

# Identify numerical and categorical columns
numerical_cols = df_manual_test.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df_manual_test.select_dtypes(include=['object']).columns

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the categorical values
for col in categorical_cols:
    df_manual_test[col] = label_encoder.fit_transform(df_manual_test[col])

# Button to predict the performance rating
if st.button("Predict Performance Rating"):
    # Make a prediction using the model
    predicted_performance_rating = performance_rating_predictor.predict(df_manual_test)

    # Output the prediction
    st.write(f"The predicted performance rating is: {predicted_performance_rating[0]}")
