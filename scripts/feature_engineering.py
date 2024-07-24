import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from datetime import datetime
from sklearn.decomposition import PCA

# Set general aesthetics for the plots
sns.set_style("whitegrid")



def create_aggregate_features(df):
    """
    Create aggregate features from a transaction dfset.

    Parameters:
    df (pandas.dfFrame): The input transaction dfset.

    Returns:
    pandas.dfFrame: A dfFrame with aggregated features for each customer.
    """
    # datetime format
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], format='%Y-%m-%d %H:%M:%S%z')

    # Extract temporal features
    df['TransactionHour'] = df['TransactionStartTime'].dt.hour
    df['TransactionDay'] = df['TransactionStartTime'].dt.day
    df['TransactionMonth'] = df['TransactionStartTime'].dt.month
    df['TransactionYear'] = df['TransactionStartTime'].dt.year

    # Aggregate transaction df by CustomerId
    customer_df = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (df['TransactionStartTime'].max() - x.max()).days,
        'TransactionId': 'count',
        'Amount': ['sum', 'mean', 'std'],
        'TransactionHour': 'mean',
        'TransactionDay': 'mean',
        'TransactionMonth': 'mean',
        'TransactionYear': 'mean'
    }).reset_index()

    # Rename the columns
    customer_df.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary', 'MeanAmount',
                             'StdAmount', 'AvgTransactionHour', 'AvgTransactionDay', 'AvgTransactionMonth',
                             'AvgTransactionYear']

    df = customer_df.dropna()

    return df



def calculate_rfms_score(df):
    # Define the weights for each feature
    recency_weight = 0.4
    frequency_weight = 0.3
    monetary_weight = 0.2
    stdamount_weight = 0.05
    meanamount_weight = 0.05
    avgtransactionhour_weight = 0.1
    avgtransactionday_weight = 0.1
    avgtransactionmonth_weight = 0.1
    avgtransactionyear_weight = 0.1

    # Calculate the normalized RFMS score
    df['RFMS_Score'] = (
        df['Recency'].rank(method='dense', ascending=True) / len(df) * recency_weight + +
        df['Frequency'].rank(method='dense', ascending=True) / len(df) * frequency_weight +
        df['Monetary'].rank(method='dense', ascending=True) / len(df) * monetary_weight +
        df['StdAmount'].rank(method='dense', ascending=True) / len(df) * stdamount_weight +
        df['MeanAmount'].rank(method='dense', ascending=True) / len(df) * meanamount_weight +
        df['AvgTransactionHour'].rank(method='dense', ascending=True) / len(df) * avgtransactionhour_weight +
        df['AvgTransactionDay'].rank(method='dense', ascending=True) / len(df) * avgtransactionday_weight +
        df['AvgTransactionMonth'].rank(method='dense', ascending=True) / len(df) * avgtransactionmonth_weight +
        df['AvgTransactionYear'].rank(method='dense', ascending=True) / len(df) * avgtransactionyear_weight
    )

    return df


def visualize_rfms_space(df):
    # Extract the RFMS_Score
    rfms_score = df['RFMS_Score']

    # Create the RFMS scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(range(len(rfms_score)), rfms_score, c=rfms_score, cmap='viridis', alpha=0.5)
    ax.set_xlabel('User Index')
    ax.set_ylabel('RFMS Score')
    ax.set_title('RFMS Space Visualization')
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('RFMS Score')

    # Defining  the boundary between high and low RFMS scores
    rfms_threshold = np.percentile(rfms_score, 60)
    ax.axhline(y=rfms_threshold, color='r', linestyle='--', label='RFMS Threshold')
    ax.legend()

    plt.show()

    return  rfms_threshold


def classify_users_by_rfms(df, rfms_threshold):
    df['Classification'] = 'High-risk'
    df.loc[df['RFMS_Score'] >= rfms_threshold, 'Classification'] = 'Low-risk'
    df['Binary_class'] = np.where(df['Classification'] == 'Low-risk', 0, 1)
    return df


import numpy as np
import pandas as pd


def calculate_woe_and_bin_features(data, features_to_bin, target, num_bins=5):
    """
    Create binned features and calculate Weight of Evidence (WoE) for specified features in the input DataFrame.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing the features and target column.
    features_to_bin (list): A list of feature names to be binned.
    target (str): The name of the target column (binary class).
    num_bins (int): The number of bins to create for the features (default is 5).

    Returns:
    pd.DataFrame: The input DataFrame with new columns for binned features and their corresponding WoE values.
    """

    def woe_binning(df, feature, target):
        """
        Calculate the Weight of Evidence (WoE) for a given feature.

        Parameters:
        df (pd.DataFrame): The input dataframe containing the feature and target columns.
        feature (str): The name of the feature column for which WoE is to be calculated.
        target (str): The name of the target column (binary class).

        Returns:
        dict: A dictionary with bins as keys and their corresponding WoE values.
        """
        woe_dict = {}
        total_good = df[target].sum()
        total_bad = df[target].count() - total_good

        for bin_id in df[feature].unique():
            bin_data = df[df[feature] == bin_id]
            good = bin_data[target].sum()
            bad = bin_data[target].count() - good

            if good == 0 or bad == 0:
                woe = 0  # Handling cases where good or bad count is zero to avoid division by zero
            else:
                woe = np.log((good / total_good) / (bad / total_bad))

            woe_dict[bin_id] = woe

        return woe_dict

    def create_binned_features(df, features, num_bins):
        """
        Create binned features for the specified features in the input DataFrame using the quantile method.

        Parameters:
        df (pd.DataFrame): The input DataFrame.
        features (list): A list of feature names to be binned.
        num_bins (int): The number of bins to create for the features.

        Returns:
        pd.DataFrame: The input DataFrame with the new binned features added.
        """
        for feature in features:
            df[f"{feature}_binned"] = pd.qcut(df[feature], q=num_bins, labels=False, duplicates='drop')
        return df

    # Ensure the target feature is included in the returned DataFrame
    data[target] = data[target]

    # Create binned features
    data = create_binned_features(data, features_to_bin, num_bins)

    # Calculate WoE for binned features
    for feature in features_to_bin:
        binned_feature = f"{feature}_binned"
        woe_dict = woe_binning(data, binned_feature, target)
        data[f'{binned_feature}_WoE'] = data[binned_feature].map(woe_dict)

    # Drop binned features
    binned_columns = [f"{feature}_binned" for feature in features_to_bin]
    data.drop(columns=binned_columns, inplace=True)

    return data



