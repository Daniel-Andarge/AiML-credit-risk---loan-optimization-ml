import pandas as pd
import numpy as np

def identify_variable_types(df):
    # Identify categorical and numerical variables
    categorical_vars = df.select_dtypes(include=['object']).columns.tolist()
    numerical_vars = df.select_dtypes(include=['int', 'float']).columns.tolist()
    return categorical_vars, numerical_vars

def fill_missing_values(df, categorical_vars, numerical_vars):
    # Fill missing values for categorical variables with mode
    df[categorical_vars] = df[categorical_vars].fillna(df[categorical_vars].mode().iloc[0])
    
    # Fill missing values for numerical variables with mean
    df[numerical_vars] = df[numerical_vars].fillna(df[numerical_vars].mean())
    return df

def clean_data(df):
    try:
        # Identify variable types
        categorical_vars, numerical_vars = identify_variable_types(df)
        
        # Fill missing values
        df = fill_missing_values(df, categorical_vars, numerical_vars)
        
        # Drop rows with missing TransactionId
        df = df.dropna(subset=['TransactionId'])
        
        # Convert TransactionStartTime to datetime
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        
        # Remove outliers using z-score
        z_scores = np.abs((df['Amount'] - df['Amount'].mean()) / df['Amount'].std())
        df = df[(z_scores < 3)]
        
        return df
    except Exception as e:
        print("An error occurred during data cleaning:", str(e))
        return None
