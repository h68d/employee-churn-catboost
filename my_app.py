import streamlit as st
import pickle
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler

st.sidebar.title('Employer Prediction')
html_temp = """
<div style="background-color:blue;padding:10px">
<h2 style="color:white;text-align:center;">Streamlit ML Cloud App </h2>
</div>"""
st.markdown(html_temp, unsafe_allow_html=True)

satisfaction_level = st.sidebar.slider("What is your satisfaction level:", 0, 1, step=0.1)
last_evaluation = st.sidebar.slider("What is your last evaluation:", 0, 1, step=0.1)
number_project = st.sidebar.selectbox("What is your number of project:", (2,3,4,5,6,7))
average_montly_hours = st.sidebar.slider("What is your average montly hours?", 96, 310, step=7)
time_spend_company = st.sidebar.slider("What is your time spend company:", 2, 10, step= 1)
work_accident = st.sidebar.selectbox("What is your work accident:",(0,1))
promotion_last_5years = st.sidebar.selectbox("What is your promotion last in the 5 years:",(0,1))
department= st.sidebar.selectbox("Select your work department", ('sales','technical','support','IT','RandD','product_mng','marketing','accounting','hr','management'))
salary = st.sidebar.selectbox("Select your salary level", ('low', 'medium', 'high'))


de_05_chur_transformer = pickle.load(open('column_trans_model', 'rb'))
de_05_chur_scaler = pickle.load(open("scaler_model","rb"))
de_05_chur_model = pickle.load(open("catboost_model","rb"))


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
