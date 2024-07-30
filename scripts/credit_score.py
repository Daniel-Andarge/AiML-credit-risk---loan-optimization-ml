import pandas as pd


def assign_credit_score(df):
    # Apply the FICO score mapping
    df['credit_score'] = (850 - (df['risk_probability'] * 550)).round().astype(int)

    # Assign the rating based on the credit score range
    df['Rating'] = pd.cut(df['credit_score'], bins=[-1, 580, 669, 739, 799, float('inf')],
                         labels=['Poor', 'Fair', 'Good', 'Very Good', 'Exceptional'])

    return df
