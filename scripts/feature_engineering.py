import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns


# Set general aesthetics for the plots
sns.set_style("whitegrid")


def process_ontime_payments(df):
    # Convert the 'TransactionStartTime' column to datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

    # Define a hypothetical payment due period (e.g., 30 days)
    payment_due_period = pd.Timedelta(days=30)

    # Split the data into debits and credits
    debits = df[df['Amount'] > 0].copy()
    credits = df[df['Amount'] < 0].copy()

    # Merge debits with credits on CustomerId to find corresponding credits within the due period
    credits['TransactionEndTime'] = credits['TransactionStartTime']
    merged = pd.merge(debits, credits, on='CustomerId', suffixes=('_debit', '_credit'))

    # Filter out credits that fall outside the due period
    merged = merged[merged['TransactionStartTime_credit'] <= (merged['TransactionStartTime_debit'] + payment_due_period)]

    # Determine on-time payments
    merged['OnTimePayment'] = merged['TransactionStartTime_credit'] <= (merged['TransactionStartTime_debit'] + payment_due_period)

    # Aggregate the payment history to get the number of on-time payments per customer
    df_final = merged.groupby('CustomerId').agg(
        OnTimePayments=('OnTimePayment', 'sum')
    ).reset_index()

    return df_final


import pandas as pd


def calculate_credit_utilization_ratio(df):
    """
    Calculates the customer's credit utilization ratio for BNPL transactions.

    Args:
        df (pandas.DataFrame): The BNPL transaction data, must include 'CustomerId', 'Amount'.

    Returns:
        pandas.DataFrame: The original DataFrame with added 'credit_utilization_ratio' column for each customer.
    """
    # Ensure 'Amount' is numeric
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

    # Calculate the total purchases and repayments for each customer
    df['PurchaseAmount'] = df['Amount'].apply(lambda x: x if x > 0 else 0)
    df['RepaymentAmount'] = df['Amount'].apply(lambda x: -x if x < 0 else 0)

    # Aggregate purchases and repayments per customer
    aggregated = df.groupby('CustomerId').agg(
        TotalPurchases=('PurchaseAmount', 'sum'),
        TotalRepayments=('RepaymentAmount', 'sum')
    ).reset_index()

    # Calculate the current balance and credit limit
    aggregated['CurrentBalance'] = aggregated['TotalPurchases'] - aggregated['TotalRepayments']
    aggregated['CreditLimit'] = aggregated['TotalPurchases']  # Assuming credit limit equals total purchases

    # Calculate the credit utilization ratio
    aggregated['CreditUtilizationRatio'] = aggregated['CurrentBalance'] / aggregated['CreditLimit']

    # Replace infinite and NaN values with 0
    aggregated['CreditUtilizationRatio'].replace([float('inf'), -float('inf')], 0, inplace=True)
    aggregated['CreditUtilizationRatio'].fillna(0, inplace=True)

    # Merge the result back to the original DataFrame
    df = pd.merge(df, aggregated[['CustomerId', 'CreditUtilizationRatio']], on='CustomerId', how='left')

    return df



def create_rfms_features(df, r_weight=0.4, f_weight=0.3, m_weight=0.2, s_weight=0.1):
    """
    Creates RFMS features (Recency, Frequency, Monetary, and Standard Deviation) for customer transactions.
    Includes the RFMS combined feature, which is the weighted sum of R, F, M, and S.

    Args:
        df (pandas.DataFrame): The transaction data, must include 'AccountId', 'TransactionStartTime',
                               'TransactionId', 'ChannelId', 'ProductId', 'Amount', 'credit_utilization_ratio',
                               and 'OnTimePayments'.
        r_weight (float): Weight for the Recency feature, default is 0.4.
        f_weight (float): Weight for the Frequency feature, default is 0.3.
        m_weight (float): Weight for the Monetary feature, default is 0.2.
        s_weight (float): Weight for the Standard Deviation feature, default is 0.1.

    Returns:
        pandas.DataFrame: DataFrame with RFMS features, the RFMS combined feature, and the 'OnTimePayments' feature.
    """
    # Convert 'TransactionStartTime' to datetime format
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

    # Calculate the RFMS combined feature with custom weights
    #df['RFMS_score'] = df['Recency'] * r_weight + df['Frequency'] * f_weight + df['Monetary'] * m_weight + df['StdDev'] * s_weight

    # Optionally calculate credit utilization ratio and multiply it with Monetary value
    if 'credit_utilization_ratio' in df.columns:
        df['Monetary'] *= df['credit_utilization_ratio']

    # Dropping duplicates to get one row per customer with RFMS values
    rfms_df = df.drop_duplicates(subset='AccountId', keep='first')

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
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
    # Prepare the color mapping
    color_map = {'Low-risk': 'green', 'High-risk': 'red'}
    df['color'] = df['Classification'].map(color_map)

    # Create 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the scatter points
    ax.scatter(df['<Recency_avg'], df['>Monetary_avg'], df['>Frequency_avg'], c=df['color'], s=50, marker='o')

    # Set axis labels
    ax.set_xlabel('Recency')
    ax.set_zlabel('Frequency')
    ax.set_ylabel('Monetary')

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

def calculate_woe_iv(df, feature, target, bins=10):
    # Binning the continuous variable if necessary
    if df[feature].dtype.kind in 'bifc':
        df[feature + '_bin'], bin_edges = pd.qcut(df[feature], q=bins, duplicates='drop', retbins=True)
    else:
        df[feature + '_bin'] = df[feature]

    # Calculate the total number of events (High-risk) and non-events (Low-risk)
    total_good = df[target].value_counts()[0]
    total_bad = df[target].value_counts()[1]

    # Create a dataframe to store the WoE and IV
    woe_df = pd.DataFrame()

    # Group by the binned feature and calculate counts
    grouped = df.groupby(feature + '_bin')[target].value_counts().unstack(fill_value=0)
    grouped.columns = ['good', 'bad']

    # Add a small value to prevent division by zero
    epsilon = 0.5
    grouped['good'] = grouped['good'] + epsilon
    grouped['bad'] = grouped['bad'] + epsilon

    # Recalculate the total number of events (High-risk) and non-events (Low-risk)
    total_good += epsilon * grouped.shape[0]
    total_bad += epsilon * grouped.shape[0]

    # Calculate the distribution of good and bad
    grouped['good_dist'] = grouped['good'] / total_good
    grouped['bad_dist'] = grouped['bad'] / total_bad

    # Calculate WoE and IV
    grouped['WoE'] = np.log(grouped['bad_dist'] / grouped['good_dist'])
    grouped['IV'] = (grouped['bad_dist'] - grouped['good_dist']) * grouped['WoE']

    # Append WoE and IV to the dataframe
    woe_df = pd.concat([woe_df, grouped])

    # Clean up temporary bin column
    df.drop(columns=[feature + '_bin'], inplace=True)

    return woe_df['WoE'], woe_df['IV'].sum()

def woe_binning(df, features, target='Classification', bins=10):
    woe_info = {}
    iv_info = {}

    for feature in features:
        woe, iv = calculate_woe_iv(df, feature, target, bins)
        df[f'{feature}_WoE'] = df[feature].map(woe)
        woe_info[feature] = woe
        iv_info[feature] = iv

    return df, woe_info, iv_info

