import streamlit as st
import pandas as pd
import numpy as np

st.title("Welcome to our free car price prediction site")
st.text("Please choose car specifications. You can use left-sidebar to specify car's\nage, horse-power, km and number of gears. If you don't specify a feature the\nmodel uses its default values for the prediction. ")

# To load machine learning model
import pickle

de_05_chur_transformer = pickle.load(open('column_trans_model', "rb"))
de_05_chur_scaler = pickle.load(open("scaler_model", "rb"))
de_05_chur_model = pickle.load(open("catboost_model", "rb"))


department= st.selectbox("Select your work department", ('sales','technical','support','IT','RandD','product_mng','marketing','accounting','hr','management'))
salary = st.selectbox("Select your salary level", ('low', 'medium', 'high'))


satisfaction_level = st.sidebar.number_input("What is your satisfaction level:",  min_value=0, max_value=10)
last_evaluation = st.sidebar.slider("What is your last evaluation:", min_value=0, max_value=1, value=10, step=1)
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

df = pd.DataFrame.from_dict([my_dict])


st.header("The configuration of your selection is below")
st.table(df)

df_trans = de_05_chur_transformer.transform(df)
df_scal = de_05_chur_scaler.transform(df_trans)

st.subheader("Press predict if configuration is okay")

if st.button("Predict"):
    prediction = de_05_chur_model.predict(df_scal)
    st.success("The estimated decision based on our model is".format(prediction))
