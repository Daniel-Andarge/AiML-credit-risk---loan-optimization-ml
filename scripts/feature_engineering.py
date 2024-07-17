import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from datetime import datetime
import pytz



def create_rfms_features(df):
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], format='%Y-%m-%d %H:%M:%S%z')

    # Calculate Recency
    max_date = df['TransactionStartTime'].max()
    df['dif'] = max_date - df['TransactionStartTime']

    # Group by 'CustomerID' and calculate the minimum difference for recency
    df_recency = df.groupby('CustomerID')['dif'].min().reset_index()

    # Convert the 'dif' to days to get the recency value
    df_recency['Recency'] = df_recency['dif'].dt.days

    # Merge recency back to the main dataframe
    df = df.merge(df_recency[['CustomerID', 'Recency']], on='CustomerID')

    # Drop the 'dif' column
    df.drop(columns=['dif'], inplace=True)

    # Calculate Frequency
    df['Frequency'] = df.groupby('CustomerID')['TransactionId'].transform('count')

    # Calculate Monetary Value
    df['Monetary'] = df.groupby('CustomerID')['Amount'].transform('sum') / df['Frequency']

    # Calculate Standard Deviation of Amounts
    df['StdDev'] = df.groupby('CustomerID')['Amount'].transform(lambda x: np.std(x, ddof=0))

    # Dropping duplicates to get one row per customer with RFMS values
    rfms_df = df.drop_duplicates(subset='CustomerID', keep='first')

    # Selecting the relevant columns for the final RFMS dataframe
    rfms_df = rfms_df[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'StdDev']]

    return rfms_df


def create_rfms_indicator_features(rfms_df):
    # Calculate mean of each RFMS feature
    rfms_mean = rfms_df[['Recency', 'Frequency', 'Monetary', 'StdDev']].mean()

    # Create new indicator features based on the mean
    rfms_df.loc[:, '>Recency'] = (rfms_df['Recency'] > rfms_mean['Recency']).astype(int)
    rfms_df.loc[:, '>Frequency'] = (rfms_df['Frequency'] > rfms_mean['Frequency']).astype(int)
    rfms_df.loc[:, '>Monetary'] = (rfms_df['Monetary'] > rfms_mean['Monetary']).astype(int)
    rfms_df.loc[:, '>StdDev'] = (rfms_df['StdDev'] > rfms_mean['StdDev']).astype(int)

    return rfms_df


def create_aggregate_features(df):
    try:
        # Group transactions by customer
        grouped = df.groupby('CustomerId')

        # Calculate aggregate features
        aggregate_features = grouped['Amount'].agg(['sum', 'mean', 'count', 'std']).reset_index()
        aggregate_features.columns = ['CustomerId', 'TotalTransactionAmount', 'AverageTransactionAmount', 'TransactionCount', 'StdTransactionAmount']

        # Merge aggregate features with original dataframe
        df = pd.merge(df, aggregate_features, on='CustomerId', how='left')

        return df

    except Exception as e:
        print("An error occurred:", e)



def extract_time_features(df):
    try:
        # Convert TransactionStartTime to datetime
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

        # Extract time-related features
        df['TransactionHour'] = df['TransactionStartTime'].dt.hour
        df['TransactionDay'] = df['TransactionStartTime'].dt.day
        df['TransactionMonth'] = df['TransactionStartTime'].dt.month
        df['TransactionYear'] = df['TransactionStartTime'].dt.year

        return df

    except Exception as e:
        print("An error occurred:", e)



def remove_outliers(df):
    """
    Removes outliers from a DataFrame based on the IQR method.

    Parameters:
    df (pd.DataFrame): Input DataFrame from which to remove outliers.

    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    # Select numerical features
    numerical_features = df.select_dtypes(include=[np.number])
    
    # Initialize a boolean mask to keep track of outliers
    mask = pd.Series([True] * len(df))

    for column in numerical_features.columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Update the mask to filter out rows containing outliers
        mask = mask & ~((df[column] < lower_bound) | (df[column] > upper_bound))

    # Filter the DataFrame to remove outliers
    df_no_outliers = df[mask]
    return df_no_outliers


def encode_categorical_variables(df):
    """
    Encodes categorical variables in the input dataframe using Label Encoding.

    Parameters:
    df (pandas.DataFrame): The input dataframe containing the data.

    Returns:
    pandas.DataFrame: The dataframe with the categorical variables encoded and converted to numerical type.
    """
    try:
        # Copy the dataframe to avoid modifying the original
        data = df.copy()

        # Label Encoding
        label_encoder = LabelEncoder()
        for col in data.select_dtypes(include=['object']).columns:
            data[col] = label_encoder.fit_transform(data[col])

        return data
    except Exception as e:
        print(f"Error occurred during encoding categorical variables: {e}")
        return None


def handle_missing_values(df):
    """
    Handles missing values in the input dataframe using imputation or removal.

    Parameters:
    df (pandas.DataFrame): The input dataframe containing the data.

    Returns:
    pandas.DataFrame: The dataframe with missing values handled.
    """
    try:
        # Copy the dataframe to avoid modifying the original
        data = df.copy()

        # Impute missing values with mean
        imputer = SimpleImputer(strategy='mean')
        numerical_cols = data.select_dtypes(include=['int32', 'int64', 'float64']).columns
        data[numerical_cols] = imputer.fit_transform(data[numerical_cols])

        # Remove rows with missing values if they are few
        if data.isnull().sum().sum() / len(data) < 0.05:
            data = data.dropna()

        return data
    except Exception as e:
        print(f"Error occurred during handling missing values: {e}")
        return None

def normalize_and_standardize_numerical_features(df):
    """
    Normalizes and standardizes the numerical features in the input dataframe.

    Parameters:
    df (pandas.DataFrame): The input dataframe containing the data.

    Returns:
    pandas.DataFrame: The dataframe with the numerical features normalized and standardized.
    """
    try:
        # Copy the dataframe to avoid modifying the original
        data = df.copy()

        # Normalize numerical features
        numerical_cols = data.select_dtypes(include=['int32', 'int64', 'float64']).columns
        min_max_scaler = MinMaxScaler()
        data[numerical_cols] = min_max_scaler.fit_transform(data[numerical_cols])

        # Standardize numerical features
        standard_scaler = StandardScaler()
        data[numerical_cols] = standard_scaler.fit_transform(data[numerical_cols])

        return data
    except Exception as e:
        print(f"Error occurred during normalization and standardization of numerical features: {e}")
        return None
