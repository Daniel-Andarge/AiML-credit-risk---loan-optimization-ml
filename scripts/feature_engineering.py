import pandas as pd
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# Set general aesthetics for the plots
sns.set_style("whitegrid")
import pandas as pd


def create_aggregate_features(df):
    """
    Create aggregate features from a transaction dataframe.

    Parameters:
    df (pandas.DataFrame): The input transaction dataframe.

    Returns:
    pandas.DataFrame: A dataframe with aggregated features for each customer.
    """
    # Ensure datetime format
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], format='%Y-%m-%d %H:%M:%S%z')

    # Extract temporal features
    df['TransactionHour'] = df['TransactionStartTime'].dt.hour
    df['TransactionDay'] = df['TransactionStartTime'].dt.day
    df['TransactionMonth'] = df['TransactionStartTime'].dt.month
    df['TransactionYear'] = df['TransactionStartTime'].dt.year

    # Aggregate transaction data by CustomerId
    df_agg = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (x.max() - x.min()).days,
        'TransactionId': 'count',
        'Amount': ['sum', 'mean', 'std'],
        'TransactionHour': 'mean',
        'TransactionDay': 'mean',
        'TransactionMonth': 'mean',
        'TransactionYear': 'mean'
    }).reset_index()

    # Rename the columns
    df_agg.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary', 'MeanAmount',
                      'StdAmount', 'AvgTransactionHour', 'AvgTransactionDay', 'AvgTransactionMonth',
                      'AvgTransactionYear']

    # Calculate additional features
    total_debits = df[df['Amount'] > 0].groupby('CustomerId')['Amount'].sum()
    total_credits = df[df['Amount'] < 0].groupby('CustomerId')['Amount'].sum()
    debit_count = df[df['Amount'] > 0].groupby('CustomerId')['TransactionId'].count()
    credit_count = df[df['Amount'] < 0].groupby('CustomerId')['TransactionId'].count()
    transaction_volatility = df.groupby('CustomerId')['Amount'].std()

    # Merge additional features
    df_agg = df_agg.merge(total_debits.rename('TotalDebits'), on='CustomerId', how='left')
    df_agg = df_agg.merge(total_credits.rename('TotalCredits'), on='CustomerId', how='left')
    df_agg = df_agg.merge(debit_count.rename('DebitCount'), on='CustomerId', how='left')
    df_agg = df_agg.merge(credit_count.rename('CreditCount'), on='CustomerId', how='left')
    df_agg = df_agg.merge(transaction_volatility.rename('TransactionVolatility'), on='CustomerId', how='left')

    # Calculate derived features
    df_agg['MonetaryAmount'] = df_agg['TotalDebits'] + abs(df_agg['TotalCredits'])
    df_agg['NetCashFlow'] = df_agg['TotalDebits'] - abs(df_agg['TotalCredits'])
    df_agg['DebitCreditRatio'] = df_agg['TotalDebits'] / abs(df_agg['TotalCredits'])

    return df_agg.dropna()


def visualize_rfms_space(df):
    # Extract the RFMS scores
    r_score = df['Recency']
    f_score = df['Frequency']
    m_score = df['Monetary']
    debit_credit_ratio = df['DebitCreditRatio']
    transaction_volatility = df['TransactionVolatility']

    # Visualize the RFMS space
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(r_score, f_score, m_score, c=debit_credit_ratio, cmap='viridis')
    ax.set_xlabel('Recency')
    ax.set_ylabel('Frequency')
    ax.set_zlabel('Monetary Value')
    plt.title('RFMS Space')

    # Defining the boundary between high and low RFMS scores
    r_threshold = np.percentile(r_score, 60)
    f_threshold = np.percentile(f_score, 50)
    m_threshold = np.percentile(m_score, 50)
    dc_threshold = np.percentile(debit_credit_ratio, 60)
    tv_threshold = np.percentile(transaction_volatility, 50)

    # Plot the thresholds
    ax.plot([r_threshold, r_threshold], [0, max(f_score)], [0, max(m_score)], color='r', linestyle='--', label='Recency Threshold')
    ax.plot([0, max(r_score)], [f_threshold, f_threshold], [0, max(m_score)], color='g', linestyle='--', label='Frequency Threshold')
    ax.plot([0, max(r_score)], [0, max(f_score)], [m_threshold, m_threshold], color='b', linestyle='--', label='Monetary Threshold')
    ax.plot([0, max(r_score)], [0, max(f_score)], [0, max(m_score)], color='y', linestyle='--', label='Debit-Credit Ratio Threshold')
    ax.plot([0, max(r_score)], [0, max(f_score)], [0, max(m_score)], color='m', linestyle='--', label='Transaction Volatility Threshold')
    ax.legend()

    plt.show()

    return r_threshold, f_threshold, m_threshold, dc_threshold, tv_threshold

def classify_users_by_rfms(df, r_threshold, f_threshold, m_threshold, dc_threshold, tv_threshold):
    df['Classification'] = 'High-risk'

    # Identify Low-risk users based on RFMS thresholds
    df.loc[(df['Recency'] <= r_threshold) & (df['Frequency'] >= f_threshold) & (
        df['Monetary'] >= m_threshold), 'Classification'] = 'Low-risk'

    # Reclassify users with low debit-credit ratio and low transaction volatility as Low-risk
    df.loc[(df['Classification'] == 'High-risk') & (
        df['DebitCreditRatio'] <= dc_threshold) & (
        df['TransactionVolatility'] <= tv_threshold), 'Classification'] = 'Low-risk'

    df['is_high_risk'] = (df['Classification'] == 'High-risk').astype(int)

    return df


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
                woe = 0 
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