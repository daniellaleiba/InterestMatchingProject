"""
data_preprocessing.py
---------------------
Module for preparing and cleaning the Interest Matching dataset.

Pipeline:
1. Load raw data
2. Clean missing values and duplicates
3. Select relevant columns
4. Encode categorical variables
5. Save the processed dataset

Author: Daniella Leiba
"""

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. DATA LOADING
def load_data(filepath):
   # Load the dataset from a CSV file.

    df = pd.read_csv(filepath)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df


# 2. DATA CLEANING
def clean_data(df):
    """
    Perform basic cleaning:
    - Remove duplicates
    - Handle missing values
    - Ensure correct data types
    """
    initial_shape = df.shape

    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values (for now, fill numeric with mean)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Fill categorical (like 'village-town') with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    print(f"Cleaned data. Removed {initial_shape[0] - df.shape[0]} duplicates.")
    return df

# 3. COLUMN SELECTION
def select_relevant_columns(df):
#  Keep only relevant columns for interest matching analysis.
#  Removes unrelated sections such as fears, personality traits, or metadata.
    
    print(f"Initial number of columns: {len(df.columns)}")

    cols_to_keep = [
        'History', 'Psychology', 'Politics', 'Mathematics', 'Physics', 'Internet',
        'PC', 'Economy Management', 'Biology', 'Chemistry', 'Reading', 'Geography',
        'Foreign languages', 'Medicine', 'Law', 'Cars', 'Art exhibitions', 'Religion',
        'Countryside, outdoors', 'Dancing', 'Musical instruments', 'Writing',
        'Passive sport', 'Active sport', 'Gardening', 'Celebrities', 'Shopping',
        'Science and technology', 'Theatre', 'Fun with friends', 'Adrenaline sports',
        'Pets', 'Village - town', 'Gender'
    ]
    
    # Keep only life interests and living environment columns
    lifeInterestsDf = df[cols_to_keep].copy()

    print(f"Remaining columns after filtering: {len(lifeInterestsDf.columns)}")
    print("Kept columns:")
    print(lifeInterestsDf.columns.tolist())

    pd.set_option('display.max_columns', None)
    #pd.set_option('display.width', None)

    print("\nSample rows after filtering:")
    print(lifeInterestsDf.head(3))

    return lifeInterestsDf

# 4. ENCODING CATEGORICAL VARIABLES
def encode_categorical(df):
  # Encode categorical columns (e.g., 'Village - town', 'Gender') using LabelEncoder.

    cat_cols = df.select_dtypes(include=['object']).columns
    encoder = LabelEncoder()

    for col in cat_cols:
        df[col] = encoder.fit_transform(df[col])
    
    print(f"Encoded categorical columns: {list(cat_cols)}")
    return df

# 5. INSPECTION UTILS
def inspect_categorical_distributions(df):
    # Print unique values in encoded categorical columns (for verification).
    cat_cols = df.select_dtypes(include=['int64']).columns
    for col in cat_cols:
        unique_vals = df[col].unique()
        print(f"{col}: {len(unique_vals)} unique values -> {unique_vals[:5]}")

# 6. SAVE PROCESSED DATA
def save_processed_data(df, output_path="data/processed/cleaned_data.csv"):
    # Save the processed dataframe to a CSV file

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")


# 7. FULL PIPELINE FUNCTION
def preprocess_data(filepath):
    """
    Run the full preprocessing pipeline:
    1. Load data
    2. Clean data
    3. Select relevant columns
    4. Encode categorical columns
    5. Save processed data
    """

    print("~"*50)
    print("Starting Data Preprocessing\n")
    
    df = load_data(filepath)
    df = clean_data(df)
    df = select_relevant_columns(df)
    df = encode_categorical(df)

    print("\nMissing values per column after cleaning:")
    print(df.isna().sum())

    inspect_categorical_distributions(df)
    save_processed_data(df)

    print("Data Preprocessing Completed Successfully\n")
    print("~"*50)

    return df