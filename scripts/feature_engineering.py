import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pytz
from category_encoders.woe import WOEEncoder



# Set general aesthetics for the plots
sns.set_style("whitegrid")

def create_rfms_features(df):
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], format='%Y-%m-%d %H:%M:%S%z')

    # Calculate Recency
    max_date = df['TransactionStartTime'].max()
    df['dif'] = max_date - df['TransactionStartTime']

    # Group by 'AccountId' and calculate the minimum difference for recency
    df_recency = df.groupby('AccountId')['dif'].min().reset_index()

    # Convert the 'dif' to days to get the recency value
    df_recency['Recency'] = df_recency['dif'].dt.days

    # Merge recency back to the main dataframe
    df = df.merge(df_recency[['AccountId', 'Recency']], on='AccountId')

    # Drop the 'dif' column
    df.drop(columns=['dif'], inplace=True)

    # Calculate Frequency
    df['Frequency'] = df.groupby('AccountId')['TransactionId'].transform('count')

    # Calculate Monetary Value
    df['Monetary'] = df.groupby('AccountId')['Amount'].transform('sum') / df['Frequency']

    # Calculate Standard Deviation of Amounts
    df['StdDev'] = df.groupby('AccountId')['Amount'].transform(lambda x: np.std(x, ddof=0))

    # Dropping duplicates to get one row per customer with RFMS values
    rfms_df = df.drop_duplicates(subset='AccountId', keep='first')

    # Selecting the relevant columns for the final RFMS dataframe
    # rfms_df = rfms_df[['AccountId', 'Recency', 'Frequency', 'Monetary', 'StdDev']]

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
        grouped = df.groupby('AccountId')

        # Calculate aggregate features
        aggregate_features = grouped['Amount'].agg(['sum', 'mean', 'count', 'std']).reset_index()
        aggregate_features.columns = ['AccountId', 'TotalTransactionAmount', 'AverageTransactionAmount', 'TransactionCount', 'StdTransactionAmount']

        # Merge aggregate features with original dataframe
        df = pd.merge(df, aggregate_features, on='AccountId', how='left')

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

def normalize_and_standardize_features(df):
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

# Outliers
def detect_rfms_outliers(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=data, orient='v', ax=ax)
    ax.set_title('Box Plot of RFMS Features')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Range')

    # Get the outlier indices for each feature
    outlier_indices = {}
    for col in data.columns:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_indices[col] = data[(data[col] < lower_bound) | (data[col] > upper_bound)].index

    return outlier_indices

def scale_features(df):
    min_max_scaler = MinMaxScaler()
    scaled = min_max_scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled, columns=df.columns)
    return df_scaled

def assign_comparative_binary_score(df):

    # Calculate the average for all rfms
    recency_avg = df['Recency'].mean()
    frequency_avg = df['Frequency'].mean()
    monetary_avg = df['Monetary'].mean()
    std_avg = df['StdDev'].mean()

    # Create new feature columns
    df['<Recency_avg'] = (df['Recency'] < recency_avg).astype(int)
    df['>Frequency_avg'] = (df['Frequency'] > frequency_avg).astype(int)
    df['>Monetary_avg'] = (df['Monetary'] > monetary_avg).astype(int)
    df['>StdDev_avg'] = (df['StdDev'] > std_avg).astype(int)
    return df

# Define High-risk and Low-risk classification rules
def classify_customer(row):
    if row['<Recency_avg'] == 1 and row['>Frequency_avg'] == 1:
        return 'Low-risk'
    elif row['<Recency_avg'] == 1 and row['>Frequency_avg'] == 0 and row['>Monetary_avg'] == 1:
        return 'Low-risk'
    else:
        return 'High-risk'


def apply_classification(df):
    # Apply the classify_customer function to each row of the DataFrame
    df['Classification'] = df.apply(classify_customer, axis=1)

    # Create the Binary_Classification column based on the Classification column
    df['Binary_Classification'] = 0
    df.loc[df['Classification'] == 'High-risk', 'Binary_Classification'] = 1

    return df


def visualize_rfms(df):
    # Prepare the color mapping
    color_map = {'Low-risk': 'green', 'High-risk': 'red'}
    df['color'] = df['Classification'].map(color_map)

    # Create 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the scatter points
    ax.scatter(df['<Recency_avg'], df['>Frequency_avg'], df['>Monetary_avg'], c=df['color'], s=50, marker='o')

    # Set axis labels
    ax.set_xlabel('Recency')
    ax.set_ylabel('Frequency')
    ax.set_zlabel('Monetary')

    # Set title
    ax.set_title('RFMS 3D Visualization')

    # Adjust axis ranges
    x_min, x_max = min(df['<Recency_avg']) - 0.1 * abs(min(df['<Recency_avg'])), max(df['<Recency_avg']) + 0.1 * abs(
        max(df['<Recency_avg']))
    y_min, y_max = min(df['>Frequency_avg']) - 0.1 * abs(min(df['>Frequency_avg'])), max(
        df['>Frequency_avg']) + 0.1 * abs(max(df['>Frequency_avg']))
    z_min, z_max = min(df['>Monetary_avg']) - 0.1 * abs(min(df['>Monetary_avg'])), max(df['>Monetary_avg']) + 0.1 * abs(
        max(df['>Monetary_avg']))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    plt.show()





import pandas as pd
import numpy as np

def calculate_woe_iv(df, feature, target):
    eps = 1e-10  # to avoid division by zero
    df = df[[feature, target]].copy()
    df['bin'] = pd.qcut(df[feature], q=10, duplicates='drop')  # Adjust 'q' for number of bins
    grouped = df.groupby('bin')[target].agg(['count', 'sum'])
    grouped['non_event'] = grouped['count'] - grouped['sum']
    grouped['event_rate'] = grouped['sum'] / (grouped['sum'].sum() + eps)
    grouped['non_event_rate'] = grouped['non_event'] / (grouped['non_event'].sum() + eps)
    grouped['woe'] = np.log(grouped['event_rate'] / (grouped['non_event_rate'] + eps) + eps)
    grouped['iv'] = (grouped['event_rate'] - grouped['non_event_rate']) * grouped['woe']
    iv = grouped['iv'].sum()
    return grouped[['woe']], iv

def woe_binning(df, features, target):
    woe_dict = {}
    iv_dict = {}
    for feature in features:
        woe_values, iv = calculate_woe_iv(df, feature, target)
        woe_dict[feature] = woe_values
        iv_dict[feature] = iv
        # Map WoE values to the original DataFrame
        df = df.copy()
        df['bin'] = pd.qcut(df[feature], q=10, duplicates='drop')
        df = df.merge(woe_values, left_on='bin', right_index=True, how='left', suffixes=('', '_woe'))
        df[feature] = df['woe']
        df.drop(columns=['bin', 'woe'], inplace=True)
    return df, woe_dict, iv_dict
