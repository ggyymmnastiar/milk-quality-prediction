import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, confusion_matrix, classification_report, ConfusionMatrixDisplay

st.title('🥛 Milk Quality Prediction App')

st.info('This is app builds a machine learning model to predict milk quality using Random Forest Classification.')

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

# Data Preparation & EDA
with st.expander('Data Preparation & EDA'):
    st.subheader("✨ Data Cleaning")

    df = df.rename(columns={
    'Temprature': 'Temperature',
    'Fat ': 'Fat'})

    st.write("● Renamed columns **'Temprature' → 'Temperature'**")
    st.write("● Renamed columns **'Fat ' → 'Fat'**")

    # Check missing value
    missing_counts = df.isnull().sum()
    st.write("**● Missing Values per Column:**")
    st.write(missing_counts)
    
    st.write('**● Final Data**')
    df

    # Features Types
    st.subheader("🔡🔢 Feature Types")
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

# Data Visualization
with st.expander("Data Visualization"):
    tab1, tab2, tab3, tab4 = st.tabs(["Target Distribution", "Feature Analysis", "Correlations", "Statistical Summary"])
    
    # Target Distribution
    with tab1:
        st.subheader("Milk Grade Distribution")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Milk Grade Distribution
        sns.countplot(data=df, x='Grade', palette='Set2', ax=ax1)
        ax1.set_title("Milk Grade Distribution", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Grade", fontsize=12)
        ax1.set_ylabel("Count", fontsize=12)
        
        # Grade Proportion
        grade_counts = df['Grade'].value_counts()
        ax2.pie(grade_counts.values, labels=grade_counts.index, autopct='%1.1f%%', 
                colors=sns.color_palette('Set2'), startangle=90)
        ax2.set_title("Grade Proportion", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Grade Statistics
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Grade Statistics:**")
            st.dataframe(grade_counts.reset_index().rename(columns={'index': 'Grade', 'Grade': 'Count'}))
        
        # Grade Percentages
        with col2:
            st.write("**Grade Percentages:**")
            percentages = (grade_counts / len(df) * 100).round(2)
            for grade, pct in percentages.items():
                st.write(f"Grade {grade}: {pct}%")
    
    # Feature Analysis
    with tab2:
        # Numeric Features Distribution
        st.subheader("Numeric Features Distribution")
        num_cols = ['pH', 'Temperature', 'Colour']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, col in enumerate(num_cols):
            sns.boxplot(data=df, x='Grade', y=col, ax=axes[i])
            axes[i].set_title(f'{col} Distribution')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Binary Features Distribution
        st.subheader("Binary Features Distribution")
        cat_cols = ['Taste', 'Odor', 'Fat', 'Turbidity']
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        for i, col in enumerate(cat_cols):
            ax = axes[i//2, i%2]
            sns.countplot(data=df, x=col, hue='Grade', ax=ax)
            ax.set_title(f'{col} vs Grade')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Feature Correlation Matrix
    with tab3:
        st.subheader("Feature Correlation Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
        plt.title("Correlation Matrix")
        st.pyplot(fig)
    
    with tab4:
        st.subheader("Statistical Summary")
        st.write(df.describe())

# # Change datatype for binary features (GANTI INI BRO YA JANGAN LUPA)
# binary_cols = ['Taste', 'Odor', 'Fat', 'Turbidity']
# label_map = {0: '0 (Not Satisfied)', 1: '1 (Satisfied)'}

# for c in binary_cols:
#     df[c] = df[c].map(label_map)

# Input Features
with st.sidebar:
    st.header('Input Features')

    pH = st.slider('pH', 0.0, 14.0, 6.6)
    Temperature = st.slider('Temperature', 30.0, 90.0, 44.22)
    
    Taste_label = st.selectbox('Taste', ('0 (Not Satisfied)', '1 (Satisfied)'))
    Odor_label = st.selectbox('Odor', ('0 (Not Satisfied)', '1 (Satisfied)'))
    Fat_label = st.selectbox('Fat', ('0 (Not Satisfied)', '1 (Satisfied)'))
    Turbidity_label = st.selectbox('Turbidity', ('0 (Not Satisfied)', '1 (Satisfied)'))

    Colour = st.slider('Colour', 240.0, 255.0, 251.84)

    label_to_num = {'0 (Not Satisfied)': 0, '1 (Satisfied)': 1}

    Taste = label_to_num[Taste_label]
    Odor = label_to_num[Odor_label]
    Fat = label_to_num[Fat_label]
    Turbidity = label_to_num[Turbidity_label]

    # Create a DataFrame for the input features
    data = {'pH': pH,
            'Temperature': Temperature,
            'Taste': Taste,
            'Odor': Odor,
            'Fat': Fat,
            'Turbidity': Turbidity,
            'Colour': Colour}
                
    input_df = pd.DataFrame(data=data, index=[0])
    combine_input_with_df = pd.concat([input_df, df], axis=0)



# Input Features
with st.expander("Input Features"):
    st.write("**Our Input**")
    input_df

    st.write("**Combined Input With Dataframe**")
    combine_input_with_df

# Define features
numeric = ['pH', 'Temperature', 'Colour']
binary = ['Taste', 'Odor', 'Fat', 'Turbidity']

# Preprocessor
preprocessor = ColumnTransformer([
                ('num', StandardScaler(), numeric), 
                ('bin', 'passthrough', binary)])

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor), 
    ('model', RandomForestClassifier(random_state=101))
])

# Define X and y
X = df.drop('Grade', axis=1)
y = df['Grade']
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                            test_size=0.2, random_state=101)

# Data Prepocessing
with st.expander("Data Preprocessing"):            
    tab1, tab2 = st.tabs(["Pipeline Setup", "Train | Test"])
    
    with tab1:
        st.subheader("Preprocessing Pipeline")
        st.write("**Feature Groups**")
        st.write(f"- Numeric: {numeric}")
        st.write(f"- Binary: {binary}")

        st.code("""
        numeric = ['pH', 'Temperature', 'Colour']
        binary = ['Taste', 'Odor', 'Fat', 'Turbidity']

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric),
            ('bin', 'passthrough', binary)
        ])
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestClassifier(random_state=101))
        ])
        """, language='python')

    with tab2:
        st.subheader("Train | Test Split")
        st.write(f"**Train Size:** {len(X_train)} samples")
        st.write(f"**Test Size:** {len(X_test)} samples")
        st.write(f"**Random State:** 101")
        st.write(f"**Test Split:** 20%")

        st.code("""
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                           test_size=0.2, random_state=101)
        """, language='python')

        st.subheader("Train Data")
        if st.button("Train the Data (.fit)"):
            # fit
            pipeline.fit(X_train, y_train)
            # predict
            y_pred = pipeline.predict(X_test)
            # confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=pipeline.named_steps['model'].classes_)
            
            # Buat classification report sebagai dictionary
            report_dict = classification_report(y_test, y_pred, output_dict=True, digits=4)

            # Convert ke DataFrame
            report_df = pd.DataFrame(report_dict).transpose()

            # Tampilkan sebagai tabel
            st.subheader("Classification Report")
            st.dataframe(report_df, use_container_width=True)



