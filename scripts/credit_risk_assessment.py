import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from category_encoders.woe import WOEEncoder

from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.linear_model import LogisticRegression

def calculate_woe_rfms_score(df):
    """
    Calculates the Recency, Frequency, Monetary, and Standard Deviation (RFMS) features
    for a given Pandas DataFrame 'df' and assigns a 'Good', 'Average', or 'Bad' label based on the RFMS_Score.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the necessary columns.
    
    Returns:
    pandas.DataFrame: The input DataFrame with the calculated RFMS features, the RFMS_Score, the 'RFMS_Segment' column,
                      the 'Assessment_Binary' column, and the additional features 'default_rate_per_bin', 'woe_per_bin',
                      and 'RFMS_bin_woe'.
    """

    df = df.copy()

    # Calculate Recency
    df['Recency'] = (df['TransactionYear'] * 365 + df['TransactionMonth'] * 30 + df['TransactionDay']) - \
                    (df.groupby('AccountId')[['TransactionYear', 'TransactionMonth', 'TransactionDay']]
                     .transform('max').sum(axis=1))
    
    # Calculate Frequency
    df['Frequency'] = df.groupby('AccountId')['TransactionId'].transform('count')
    
    # Calculate Monetary
    df['Monetary'] = df.groupby('AccountId')['Amount'].transform('sum')
    
    # Calculate Standard Deviation
    df['StdDev'] = df.groupby('AccountId')['Amount'].transform('std').fillna(0)
    
    # Standardize the RFMS features
    scaler = StandardScaler()
    df[['Recency', 'Frequency', 'Monetary', 'StdDev']] = scaler.fit_transform(df[['Recency', 'Frequency', 'Monetary', 'StdDev']])
    
    # Calculate RFMS score
    df['RFMS_Score'] = df['Recency'] + df['Frequency'] + df['Monetary'] + df['StdDev']
    
    # Bin the RFMS_Score values into 3 bins
    df['RFMS_bin'] = pd.qcut(df['RFMS_Score'], q=3, labels=False)

    # Calculate the central tendency and variability
    mean_rfms = df['RFMS_Score'].mean()
    median_rfms = df['RFMS_Score'].median()
    std_rfms = df['RFMS_Score'].std()

    print(f"Mean RFMS Score: {mean_rfms:.2f}")
    print(f"Median RFMS Score: {median_rfms:.2f}")
    print(f"Standard Deviation of RFMS Scores: {std_rfms:.2f}")

    # Determine the "Good" and "Bad" thresholds
    good_rfms_threshold = median_rfms
    bad_rfms_threshold = median_rfms - std_rfms

    # Create the "Assessment" column
    df['Assessment'] = np.where(df['RFMS_Score'] >= good_rfms_threshold, 'Good', 
                                np.where(df['RFMS_Score'] >= bad_rfms_threshold, 'Average', 'Bad'))
    
    # Create the "Assessment_Binary" column
    df['Assessment_Binary'] = np.where(df['Assessment'] == 'Good', 1, 0)

    # Calculate Weight of Evidence (WOE) and additional features
    woe_encoder = WOEEncoder(cols=['RFMS_bin'])
    df['RFMS_bin_woe'] = woe_encoder.fit_transform(df[['RFMS_bin']], df['Assessment_Binary'])
    
    # Calculate default rate per bin and WOE per bin
    df['default_rate_per_bin'] = df.groupby('RFMS_bin')['Assessment_Binary'].transform('mean')
    df['woe_per_bin'] = df.groupby('RFMS_bin')['RFMS_bin_woe'].transform('first')

    # Select the required features
    required_features = ['CustomerId', 'CurrencyCode', 'CountryCode', 'ProductId', 'ProductCategory', 'ChannelId',
                         'Amount', 'Value', 'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear',
                         'PricingStrategy', 'FraudResult', 'RFMS_Score', 'RFMS_bin', 'Assessment_Binary', 'RFMS_bin_woe',
                         'default_rate_per_bin', 'woe_per_bin']
    
    # Ensure the required features exist in the dataframe
    missing_features = set(required_features) - set(df.columns)
    for feature in missing_features:
        df[feature] = np.nan
    
    return df[required_features]



def credit_score_model(rfms_df):
    df = calculate_credit_score(rfms_df)
    # Split the data into features and target
    X = df[['RFMS_score', 'RFMS_bin_woe']]
    y = df['fico_credit_score']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")
    
    return model, df

def calculate_credit_score(df):
    # Step 1: Logistic Regression Model
    X = df[['RFMS_score', 'RFMS_bin_woe']]
    y = df['assessment_binary']
    
    model = LogisticRegression()
    model.fit(X, y)
    
    # Step 2: Calculate the Credit Score
    credit_score = model.coef_[0] * df['RFMS_score'] + model.coef_[1] * df['RFMS_bin_woe']
    
    # Step 3: Map to FICO Credit Score Range
    fico_credit_score = 300 + 550 * (1 / (1 + np.exp(-credit_score)))
    
    # Add the calculated credit scores to the dataframe
    df['credit_score'] = credit_score
    df['fico_credit_score'] = fico_credit_score
    
    return df


