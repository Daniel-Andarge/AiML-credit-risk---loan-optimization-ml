import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
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

        df = df.drop('TransactionStartTime' , axis=1)
        return df

    except Exception as e:
        print("An error occurred:", e)



def standardize_features(df):
    """
    Standardizes the numerical features in the input dataframe.

    Parameters:
    df (pd.DataFrame): The input dataframe containing the data.

    Returns:
    pd.DataFrame: The dataframe with the numerical features standardized.
    """
    try:
        data = df.copy()

        # Identify numerical columns
        numerical_cols = data.select_dtypes(include=['int32', 'int64', 'float64']).columns

        # Standardize numerical features
        standard_scaler = StandardScaler()
        data[numerical_cols] = standard_scaler.fit_transform(data[numerical_cols])

        return data

    except Exception as e:
        print(f"Error occurred during standardization of numerical features: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error


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


def rfms_segmentation(df):
    """
    Performs RFM segmentation on the input DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing Recency, Frequency, Monetary, and StdDev features.

    Returns:
        pandas.DataFrame: Updated DataFrame with RFM segmentation features.
    """
    # Define Scale
    scale = 3

    recency_q = df['Recency'].quantile([0.25, 0.50])
    frequency_q = df['Frequency'].quantile([0.25, 0.50])
    monetary_q = df['Monetary'].quantile([0.25, 0.50])

    # Step 3: Assign Scores
    def assign_recency_score(x):
        if x <= recency_q[0.25]:
            return scale
        elif x <= recency_q[0.50]:
            return scale - 1
        else:
            return 1

    def assign_frequency_score(x):
        if x <= frequency_q[0.25]:
            return 1
        elif x <= frequency_q[0.50]:
            return scale - 1
        else:
            return scale

    def assign_monetary_score(x):
        if x <= monetary_q[0.25]:
            return 1
        elif x <= monetary_q[0.50]:
            return scale - 1
        else:
            return scale

    df['Recency_Score'] = df['Recency'].apply(assign_recency_score)
    df['Frequency_Score'] = df['Frequency'].apply(assign_frequency_score)
    df['Monetary_Score'] = df['Monetary'].apply(assign_monetary_score)
    df['Std_Score'] = df['StdTransactionAmount'].apply(assign_recency_score)

    df['RFM_Score'] = df.Recency_Score.map(str) \
                                    + df.Frequency_Score.map(str) \
                                    + df.Monetary_Score.map(str) \
                                    + df.Std_Score.map(str)

    return df




def visualize_rfms(df):
    """
    Performs K-means clustering on the RFMS scores and visualizes the results.

    Parameters:
    df (pd.DataFrame): The input dataframe containing RFMS scores.

    Returns:
    np.ndarray: The cluster centers.
    """
    try:
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=2, random_state=42)
        df['Cluster'] = kmeans.fit_predict(df)



        # Visualize clusters using all features
        # Visualize clusters
        sns.pairplot(df, vars=['Amount', 'Value', 'PricingStrategy',
        'FraudResult', 'Monetary', 'AverageTransactionAmount', 'Frequency',
        'StdTransactionAmount', 'Recency', 'TransactionHour', 'TransactionDay',
        'TransactionMonth', 'TransactionYear', 'Recency_Score',
        'Frequency_Score', 'Monetary_Score', 'Std_Score'], hue='Cluster',
                     palette='viridis')
        plt.show()

        # Find cluster centers
        centers = kmeans.cluster_centers_

        return centers

    except Exception as e:
        print(f"Error occurred during visualization and clustering of RFMS scores: {e}")
        return None




def apply_segment_based_on_clusters(df, centers):

    high_risk_cluster = np.argmin(centers[:, :3].mean(axis=1))
    df['Segment'] = np.where(df['Cluster'] == high_risk_cluster, 'High-risk', 'Low-risk')

    return df.drop(columns=['Cluster'])


import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


def calculate_woe_iv(data, feature, target):
    """
    Calculate Weight of Evidence (WoE) and Information Value (IV) for a given feature.

    Parameters:
    - data: pandas DataFrame
    - feature: the feature column name
    - target: the target column name

    Returns:
    - iv: Information Value (IV) for the feature
    """
    eps = 1e-10  # To avoid division by zero
    # Calculate the total number of 'High-risk' and 'Low-risk'
    total_good = np.sum(data[target] == 'Low-risk')
    total_bad = np.sum(data[target] == 'High-risk')

    # Group by feature bins and calculate WoE and IV
    grouped = data.groupby(feature)[target].value_counts(normalize=False).unstack().fillna(0)
    grouped['good'] = grouped['Low-risk']
    grouped['bad'] = grouped['High-risk']

    grouped['good_dist'] = grouped['good'] / total_good
    grouped['bad_dist'] = grouped['bad'] / total_bad

    grouped['woe'] = np.log((grouped['good_dist'] + eps) / (grouped['bad_dist'] + eps))
    grouped['iv'] = (grouped['good_dist'] - grouped['bad_dist']) * grouped['woe']

    iv = grouped['iv'].sum()

    return iv


def woe_binning(data, target='Segment', n_bins=10):
    """
    Perform WoE binning on all features of the DataFrame.

    Parameters:
    - data: pandas DataFrame
    - target: the target column name
    - n_bins: number of bins to divide continuous features into

    Returns:
    - binned_data: DataFrame with binned features
    - woe_iv_dict: Dictionary with IV values for each feature
    """
    binned_data = data.copy()
    woe_iv_dict = {}

    for feature in data.columns:
        if feature == target:
            continue

        # Apply binning to continuous features
        if np.issubdtype(data[feature].dtype, np.number):
            est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
            binned_data[feature] = est.fit_transform(data[[feature]]).astype(int)

        # Calculate IV for the feature
        iv = calculate_woe_iv(binned_data, feature, target)
        woe_iv_dict[feature] = iv

    return binned_data, woe_iv_dict




