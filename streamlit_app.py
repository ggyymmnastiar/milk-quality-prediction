import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

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

# Data Preprocessing
with st.expander('Preprocessing'):
    st.subheader("✨ Data Cleaning")

    df = df.rename(columns={
    'Temprature': 'Temperature',
    'Fat ': 'Fat'})

    # Rename columns
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

# EDA
with st.expander("Exploratory Data Analysis"):
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
    st.subheader("Feature Correlation Analysis")
    
    # Calculate correlation matrix
    corr_matrix = df.corr(numeric_only=True)
    
    # Create a more informative correlation visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Heatmap with better styling
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                square=True, ax=ax1, fmt='.2f', cbar_kws={'shrink': 0.8})
    ax1.set_title("Correlation Matrix", fontsize=14, fontweight='bold')
    
    # Top correlations
    # Get upper triangle of correlation matrix
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    corr_upper = corr_matrix.mask(mask)
    
    # Find top correlations
    top_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if not pd.isna(corr_upper.iloc[i, j]):
                top_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], 
                               corr_upper.iloc[i, j]))
    
    top_corr.sort(key=lambda x: abs(x[2]), reverse=True)
    
    # Plot top correlations
    top_features = [f"{pair[0]} vs {pair[1]}" for pair in top_corr[:10]]
    top_values = [pair[2] for pair in top_corr[:10]]
    colors = ['red' if x < 0 else 'blue' for x in top_values]
    
    ax2.barh(range(len(top_features)), top_values, color=colors, alpha=0.7)
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels(top_features)
    ax2.set_xlabel('Correlation Coefficient')
    ax2.set_title('Top 10 Feature Correlations', fontsize=14, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display top correlations as text
    st.write("**Top Feature Correlations:**")
    for i, (feat1, feat2, corr_val) in enumerate(top_corr[:10]):
        st.write(f"{i+1}. {feat1} ↔ {feat2}: {corr_val:.3f}")
    
    with tab4:
        st.subheader("Statistical Summary")
        st.write(df.describe())

# Input Features
with st.sidebar:
    st.header('Input features')
    
    island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
    bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
    bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
    flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
    body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
    gender = st.selectbox('Gender', ('male', 'female'))