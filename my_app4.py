import streamlit as st
import pandas as pd
import numpy as np

st.title("Welcome to our free employer behaviour prediction site")
st.text("Please ask your eployee following questions. Depending of their answer our model  \nwill predic whether they stay or leave. You can use left-sidebar to specify extra \nspecifications. Please enjoy using our model. ")

# To load machine learning model
import pickle

de_05_chur_transformer = pickle.load(open('column_trans_model', "rb"))
de_05_chur_scaler = pickle.load(open("scaler_model", "rb"))
de_05_chur_model = pickle.load(open("catboost_model", "rb"))


department= st.selectbox("Select your work department", ('sales','technical','support','IT','RandD','product_mng','marketing','accounting','hr','management'))
salary = st.selectbox("Select your salary level", ('low', 'medium', 'high'))
satisfaction_level = st.sidebar.selectbox("What is your satisfaction level:", (0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1))
last_evaluation = st.sidebar.selectbox("What is your last evaluation:", (0.36,0.4,0.5,0.6,0.7,0.8,0.9,1))
number_project = st.sidebar.selectbox("What is your number of project:", (2,3,4,5,6,7))
average_montly_hours = st.sidebar.slider("What is your average montly hours?", min_value=96, max_value=310, value=150,step=7)
time_spend_company = st.sidebar.slider("What is your time spend company:", min_value=2, max_value=10, value=5,step= 1)
work_accident = st.sidebar.selectbox("What is your work accident:",(0,1))
promotion_last_5years = st.sidebar.selectbox("What is your promotion last in the 5 years:",(0,1))


my_dict = {
    "satisfaction_level": satisfaction_level,
    "last_evaluation": last_evaluation,
    "number_project": number_project,
    "average_montly_hours": average_montly_hours,
    'time_spend_company':time_spend_company,
    "work_accident": work_accident, 
    'promotion_last_5years':promotion_last_5years,
    "department": department, 
    "salary": salary
    }

# my_dict2 = {
#     "Satisfaction level": satisfaction_level,
#     "Last evaluation": last_evaluation,
#     "Number of project": number_project,
#     "Average montly work hours": average_montly_hours,
#     'Time spend in company':time_spend_company,
#     "Work accident": work_accident, 
#     'Promotion in last 5 years':promotion_last_5years,
#     "Department": department, 
#     "Salary": salary
#     }

df2 = pd.DataFrame.from_dict([my_dict2])
st.write("The configuration of your selections is below") 
st.table(df2)


df = pd.DataFrame.from_dict([my_dict])

df_trans = de_05_chur_transformer.transform(df)
df_scal = de_05_chur_scaler.transform(df_trans)

st.subheader("Press predict if your configuration is okay")

# Prediction with user inputs
predict = st.button("Predict")
result = de_05_chur_model.predict(df_scal)
if predict:
    st.write("Based on our model your prediction is:")
    if int(result[0]) == 0:
        st.success(result[0], icon="✅")
        st.write("You can relax! Your staff is staying in your company.")
    else:
        st.error(result[0], icon="🚨")
        st.write("Bad news! Your staff is going.")
    
