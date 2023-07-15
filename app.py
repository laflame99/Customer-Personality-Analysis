from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
import streamlit as st
import pandas as pd
import pickle
import os

filename = 'ohe_xgb.sav'
loaded_model = pickle.load(open(filename, 'rb'))

st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True)
st.title("Prediction")

with st.form("my_form"):
    Income=st.number_input(label='Income',step=0.001,format="%.6f")
    Last_purchase=st.number_input(label='Last_purchase',step=0.001,format="%.6f")
    Wine=st.number_input(label='Wine',step=0.01,format="%.2f")
    Fruit=st.number_input(label='Fruit',step=0.01,format="%.2f")
    Meat=st.number_input(label='Meat',step=0.01,format="%.2f")
    Fish=st.number_input(label='Fish',step=0.01,format="%.6f")
    Sweets=st.number_input(label='Sweets',step=0.01,format="%.6f")
    Gold=st.number_input(label='Gold',step=0.1,format="%.6f")
    NumDealsPurchases=st.number_input(label='NumDealsPurchases',step=0.1,format="%.6f")
    NumWebPurchases=st.number_input(label='NumWebPurchases',step=0.1,format="%.6f")
    NumCatalogPurchases=st.number_input(label='NumCatalogPurchases',step=1)
    NumStorePurchases=st.number_input(label='NumStorePurchases',step=1)
    NumWebVisitsMonth=st.number_input(label='NumWebVisitsMonth',step=0.1,format="%.1f")
    Prev_camp_offer=st.number_input(label='Prev_camp_offer',step=0.01,format="%.6f")
    Complain=st.number_input(label='Complain',step=0.01,format="%.6f")
    Age=st.number_input(label='Age',step=0.01,format="%.6f")
    Total_Spent=st.number_input(label='Total_Spent',step=1)
    Children=st.number_input(label='Children',step=1)
    Married=st.number_input(label='Married',step=1)
    qualification = st.text_input(label='qualification')


    
    data=[[Income,Last_purchase, Wine, Fruit, Meat,	Fish, Sweets, Gold,	NumDealsPurchases, NumWebPurchases,	NumCatalogPurchases, NumStorePurchases,	NumWebVisitsMonth, Prev_camp_offer,	Complain, Age, Total_Spent,	Children, Married, qualification]]

    submitted = st.form_submit_button("Submit")

if submitted:
    clust=loaded_model.predict(data)[0]
    print('Data Belongs to Cluster',clust)

