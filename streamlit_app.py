import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('🥛 Milk Quality Prediction App')

st.info('This is app builds a machine learning model to predict milk quality using Random Forest Classification')

# Data Overview
with st.expander('Data Overview'):
    df = pd.read_csv('https://raw.githubusercontent.com/ggyymmnastiar/g-milkqualityprediction/refs/heads/master/milk_dataset.csv')
    
    st.write(f"**● Total Samples:** {len(df)}")
    
    st.write(f"**● Target Classes:** {df['Grade'].unique()}")
    
    st.write('**● Raw data:**')
    df
    
    st.write('**● X (Input):**')
    X_raw = df.drop('Grade', axis=1)
    X_raw
    
    st.write('**● y (Label):**')
    y_raw = df['Grade']
    y_raw

# Data Preprocessing
with st.expander('Preprocessing'):
    st.subheader("Data Cleaning")
    df = df.rename(columns={
    'Temprature': 'Temperature',
    'Fat ': 'Fat'})

    st.write("✅ Renamed columns: 'Temprature' → 'Temperature'")
    st.write("✅ Renamed columns: 'Fat ' → 'Temperature'")

