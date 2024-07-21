import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set general aesthetics for the plots
sns.set_style("whitegrid")

def create_aggregate_features(df):
    try:
        # Convert TransactionStartTime to datetime
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], format='%Y-%m-%d %H:%M:%S%z')

        # Group transactions by customer
        grouped = df.groupby('CustomerId')

        # Calculate aggregate features
        aggregate_features = grouped['Amount'].agg(['sum', 'mean', 'count', 'std']).reset_index()
        aggregate_features.columns = ['CustomerId', 'Monetary', 'AverageTransactionAmount', 'Frequency', 'StdTransactionAmount']

        # Calculate Recency feature
        recency_feature = (df.groupby('CustomerId')['TransactionStartTime'].max().max() - df.groupby('CustomerId')['TransactionStartTime'].max()).dt.days.reset_index()
        recency_feature.columns = ['CustomerId', 'Recency']

        # Merge recency and aggregate features with the original dataframe
        df = df.merge(aggregate_features, on='CustomerId', how='left')
        df = df.merge(recency_feature, on='CustomerId', how='left')
        df = df.drop_duplicates(subset='CustomerId', keep='first')
        df = df.dropna()
        return df

    except KeyError as e:
        print(f"Column not found: {e}")
    except pd.errors.ParserError as e:
        print(f"Parsing error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")




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
    # Select only the numeric features
    numeric_cols = data.select_dtypes(include=['int64', 'int32','float64']).columns
    numeric_data = data[numeric_cols]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=numeric_data, orient='v', ax=ax)
    ax.set_title('Box Plot for Numeric Features')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Range')

    # Get the outlier indices for each numeric feature
    outlier_indices = {}
    for col in numeric_cols:
        q1 = numeric_data[col].quantile(0.25)
        q3 = numeric_data[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_indices[col] = numeric_data[(numeric_data[col] < lower_bound) | (numeric_data[col] > upper_bound)].index

    return outlier_indices

def scale_features(df):
    # Select only the numeric features
    numeric_cols = df.select_dtypes(include=['int64', 'int32', 'float64']).columns
    numeric_data = df[numeric_cols]

    # Scale the numeric features using MinMaxScaler
    min_max_scaler = MinMaxScaler()
    scaled_numeric = min_max_scaler.fit_transform(numeric_data)

    # Create a new dataframe with the scaled numeric features
    df_scaled = pd.DataFrame(scaled_numeric, columns=numeric_cols)

    # Combine the scaled numeric features with the original non-numeric features
    df_scaled = pd.concat([df_scaled, df.drop(numeric_cols, axis=1)], axis=1)

    return df_scaled

def assign_comparative_binary_score(df):

    # Calculate the average for all rfms
    recency_avg = df['Recency'].mean()
    frequency_avg = df['Frequency'].mean()
    monetary_avg = df['Monetary'].mean()
    segmt_avg = df['StdDev'].mean()
    onTime_avg = df['OnTimePayments'].mean()

    # Create new feature columns
    df['<Recency_avg'] = (df['Recency'] < recency_avg).astype(int)
    df['>Frequency_avg'] = (df['Frequency'] > frequency_avg).astype(int)
    df['>Monetary_avg'] = (df['Monetary'] > monetary_avg).astype(int)
    df['>StdDev_avg'] = (df['StdDev'] > segmt_avg).astype(int)
    df['>OnTimePayment_avg'] = (df['OnTimePayments'] > onTime_avg).astype(int)

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
    # Normalize the RFMS scores for clustering
    scaler = StandardScaler()
    rfms_scaled = scaler.fit_transform(df[['Recency_Score', 'Frequency_Score', 'Monetary_Score', 'Std_Score']])

    # Perform K-means clustering to find natural groupings
    kmeans = KMeans(n_clusters=2, random_state=42)  # Assuming we want to split into two groups: high and low
    df['Cluster'] = kmeans.fit_predict(rfms_scaled)

    # Visualize clusters
    sns.pairplot(df, vars=['Recency_Score', 'Frequency_Score', 'Monetary_Score', 'Std_Score'], hue='Cluster',
                 palette='viridis')
    plt.show()

    # Find cluster centers
    centers = kmeans.cluster_centers_
    centers_unscaled = scaler.inverse_transform(centers)

    return centers_unscaled


def apply_segment_based_on_clusters(df, centers):

    high_risk_cluster = np.argmin(centers[:, :3].mean(axis=1))
    df['Segment'] = np.where(df['Cluster'] == high_risk_cluster, 'High-risk', 'Low-risk')

    return df.drop(columns=['Cluster'])




