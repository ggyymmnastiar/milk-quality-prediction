import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('🥛 Milk Quality Prediction App')

st.info('This is app builds a machine learning model to predict milk quality using Random Forest Classification')

with st.expander('Data Overview'):
    st.write(f"**Total Samples:** {len(df)}")
    st.write(f"**Features:** {len(df.columns)}")
    st.write(f"**Target Classes:** {df['Grade'].nunique()}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df.head())
    with col2:
        st.write(df.info())
  
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/ggyymmnastiar/g-milkqualityprediction/refs/heads/master/milk_dataset.csv')
  df

  st.write('**X (Input)**')
  X_raw = df.drop('Grade', axis=1)
  X_raw

  st.write('**y (Label)**')
  y_raw = df['Grade']
  y_raw
