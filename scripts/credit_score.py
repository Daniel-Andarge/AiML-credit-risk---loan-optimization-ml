import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def calculate_credit_score(df):
    """
    Calculates credit scores using a logistic regression model and maps them to FICO credit scores.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the RFMS data.
        
    Returns:
        df (pd.DataFrame): DataFrame with calculated credit scores and FICO credit scores.
    """
    # Handle missing values
    df.fillna(df.mean(), inplace=True)
    
    # Logistic Regression Model
    X = df[['RFMS_Score', 'RFMS_bin_woe']]
    y = df['Assessment_Binary']
    
    model = LogisticRegression()
    model.fit(X, y)
    
    # Calculate the Credit Score
    credit_score = model.coef_[0][0] * df['RFMS_Score'] + model.coef_[0][1] * df['RFMS_bin_woe']
    
    # Map to FICO Credit Score Range
    fico_credit_score = 300 + 550 * (1 / (1 + np.exp(-credit_score)))
    
    # Add credit scores to the dataframe
    df['credit_score'] = credit_score
    df['fico_credit_score'] = fico_credit_score
    
    return df
