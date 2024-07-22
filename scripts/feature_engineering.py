import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer
from datetime import datetime
from sklearn.decomposition import PCA
# Set general aesthetics for the plots
sns.set_style("whitegrid")


def create_aggregate_features(data):
    """
    Process customer transaction data to calculate RFMS scores and classify users into good and bad based on clustering.

    Args:
        file_path (str): Path to the CSV file containing transaction data.

    Returns:
        pd.DataFrame: DataFrame with RFMS scores and user classification labels.
    """
    # Load the transaction data


    # Ensure the TransactionStartTime is in datetime format
    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'], format='%Y-%m-%d %H:%M:%S%z')

    # Extract temporal features
    data['TransactionHour'] = data['TransactionStartTime'].dt.hour
    data['TransactionDay'] = data['TransactionStartTime'].dt.day
    data['TransactionMonth'] = data['TransactionStartTime'].dt.month
    data['TransactionYear'] = data['TransactionStartTime'].dt.year

    # Ensure current time is timezone-aware
    now = datetime.now(data['TransactionStartTime'].dt.tz)

    # Aggregate transaction data by CustomerId
    customer_data = data.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (now - x.max()).days,
        'TransactionId': 'count',
        'Amount': ['sum', 'mean', 'std'],
        'TransactionHour': 'mean',
        'TransactionDay': 'mean',
        'TransactionMonth': 'mean',
        'TransactionYear': 'mean'
    }).reset_index()


    customer_data.columns = ['CustomerId', 'Recency', 'TransactionCount', 'TotalTransactionAmount', 'AverageTransactionAmount', 'StdTransactionAmount', 'AvgTransactionHour', 'AvgTransactionDay', 'AvgTransactionMonth', 'AvgTransactionYear']

    # Normalize the scores (simple min-max normalization)
    customer_data['R_norm'] = 1 - (customer_data['Recency'] - customer_data['Recency'].min()) / (customer_data['Recency'].max() - customer_data['Recency'].min())
    customer_data['F_norm'] = (customer_data['TransactionCount'] - customer_data['TransactionCount'].min()) / (customer_data['TransactionCount'].max() - customer_data['TransactionCount'].min())
    customer_data['M_norm'] = (customer_data['TotalTransactionAmount'] - customer_data['TotalTransactionAmount'].min()) / (customer_data['TotalTransactionAmount'].max() - customer_data['TotalTransactionAmount'].min())
    customer_data['S_norm'] = (customer_data['StdTransactionAmount'] - customer_data['StdTransactionAmount'].min()) / (customer_data['StdTransactionAmount'].max() - customer_data['StdTransactionAmount'].min())
    customer_data['Hour_norm'] = (customer_data['AvgTransactionHour'] - customer_data['AvgTransactionHour'].min()) / (customer_data['AvgTransactionHour'].max() - customer_data['AvgTransactionHour'].min())
    customer_data['Day_norm'] = (customer_data['AvgTransactionDay'] - customer_data['AvgTransactionDay'].min()) / (customer_data['AvgTransactionDay'].max() - customer_data['AvgTransactionDay'].min())
    customer_data['Month_norm'] = (customer_data['AvgTransactionMonth'] - customer_data['AvgTransactionMonth'].min()) / (customer_data['AvgTransactionMonth'].max() - customer_data['AvgTransactionMonth'].min())
    customer_data['Year_norm'] = (customer_data['AvgTransactionYear'] - customer_data['AvgTransactionYear'].min()) / (customer_data['AvgTransactionYear'].max() - customer_data['AvgTransactionYear'].min())

    # Calculate the RFMS score
    weights = {'R': 0.25, 'F': 0.25, 'M': 0.25, 'S': 0.25}
    customer_data['RFMS_Score'] = (
            weights['R'] * customer_data['R_norm'] +
            weights['F'] * customer_data['F_norm'] +
            weights['M'] * customer_data['M_norm'] +
            weights['S'] * customer_data['S_norm']
    )

    df = customer_data.dropna()
    return df


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



# Outliers
def detect_rfms_outliers(data):
    # Select only the numeric features
    numeric_cols = data.select_dtypes(include=['int64', 'int32', 'float64']).columns
    numeric_data = data[numeric_cols]

    fig, axes = plt.subplots(nrows=len(numeric_cols) // 3 + 1, ncols=3, figsize=(18, 6 * len(numeric_cols) // 3 + 6))

    for i, col in enumerate(numeric_cols):
        row = i // 3
        col_idx = i % 3

        sns.boxplot(x=col, data=numeric_data, ax=axes[row, col_idx])
        axes[row, col_idx].set_title(f'Box Plot for {col}')
        axes[row, col_idx].set_xlabel(col)
        axes[row, col_idx].set_ylabel('Range')

    plt.tight_layout()

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


def visualize_rfms(df):
    # Scatter plot of RFMS scores
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='R_norm', y='F_norm', hue='RFMS_Score', palette='viridis')
    plt.title('RFMS Score Distribution')
    plt.xlabel('Recency (Normalized)')
    plt.ylabel('Frequency (Normalized)')
    plt.show()

    # Fit K-Means clustering
    kmeans = KMeans(n_clusters=2)
    df['Cluster'] = kmeans.fit_predict(df[['R_norm', 'F_norm', 'M_norm', 'S_norm']])

    # Assign labels based on clusters
    cluster_centers = kmeans.cluster_centers_
    return cluster_centers


def apply_segment_based_on_clusters(df, cluster_centers):
    """
    Assign labels to clusters based on the highest RFMS score cluster.

    Parameters:
    df (pd.DataFrame): DataFrame containing cluster assignments.
    cluster_centers (np.ndarray): Array of cluster center coordinates.

    Returns:
    pd.DataFrame: DataFrame with an additional 'Classification' and 'Binary_class' columns.
    """
    if 'Cluster' not in df.columns:
        raise ValueError("The dataframe must contain a 'Cluster' column with cluster assignments.")

    if cluster_centers.shape[1] == 0:
        raise ValueError("Cluster centers array must have at least one column.")

    # Determine the index of the cluster with the highest value in the last column (RFMS score)
    high_cluster = np.argmax(cluster_centers[:, -1])

    # Assign labels based on the cluster assignment
    df['Classification'] = df['Cluster'].apply(lambda x: 'Low-risk' if x == high_cluster else 'High-risk')

    # Create the Binary_class feature
    df['Binary_class'] = df['Classification'].apply(lambda x: 1 if x == 'High-risk' else 0)

    return df


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
    total_good = np.sum(data[target] == 0)
    total_bad = np.sum(data[target] == 1)

    # Group by feature bins and calculate WoE and IV
    grouped = data.groupby(feature)[target].value_counts(normalize=False).unstack().fillna(0)
    grouped['good'] = grouped[0]
    grouped['bad'] = grouped[1]

    grouped['good_dist'] = grouped['good'] / total_good
    grouped['bad_dist'] = grouped['bad'] / total_bad

    grouped['woe'] = np.log((grouped['good_dist'] + eps) / (grouped['bad_dist'] + eps))
    grouped['iv'] = (grouped['good_dist'] - grouped['bad_dist']) * grouped['woe']

    iv = grouped['iv'].sum()

    return iv


def woe_binning(data, target='Binary_class', n_bins=5):
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


def Visualize_clusters(customer_data):

    features = customer_data[
        ['R_norm', 'F_norm', 'M_norm', 'S_norm', 'Hour_norm', 'Day_norm', 'Month_norm', 'Year_norm']]

    # Scale features for clustering
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(features_scaled)
    customer_data['PC1'] = principal_components[:, 0]
    customer_data['PC2'] = principal_components[:, 1]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=customer_data, x='PC1', y='PC2', hue='Classification', palette='viridis')
    plt.title('RFMS Score Clusters with Temporal Features')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Classification')
    plt.show()
