import pandas as pd
import numpy as np


def assign_credit_score(df):

    # Apply the FICO score mapping function
    df['credit_score'] = 850 - (df['risk_probability'] * 550)

    
    return df
