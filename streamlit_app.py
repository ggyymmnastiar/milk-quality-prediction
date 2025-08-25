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
    st.subheader("<u>Data Cleaning</u> ")

    df = df.rename(columns={
    'Temprature': 'Temperature',
    'Fat ': 'Fat'})

    st.write("● Renamed columns **'Temprature' → 'Temperature'**")
    st.write("● Renamed columns **'Fat ' → 'Fat'**")

    missing_counts = df.isnull().sum()
    st.write("**● Missing Values per Column:**")
    st.write(missing_counts)
    
    st.write('**● Final Data**')
    df

    st.subheader("<u>Feature Types</u> ")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Numeric Features:**")
        st.write("- pH (float64)")
        st.write("- Temperature (int64)")
        st.write("- Colour (int64)")
    with col2:
        st.write("**Binary Features:**")
        st.write("- Taste (0/1)")
        st.write("- Odor (0/1)")
        st.write("- Fat (0/1)")
        st.write("- Turbidity (0/1)")
