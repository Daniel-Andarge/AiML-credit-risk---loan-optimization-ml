import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns


# Set general aesthetics for the plots
sns.set_style("whitegrid")

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


def calculate_credit_utilization_ratio(df):
    """
    Calculates the customer's credit utilization ratio for BNPL transactions.

    Args:
        df (pandas.DataFrame): The BNPL transaction data, must include 'CustomerId', 'Amount'.

    Returns:
        pandas.DataFrame: The original DataFrame with added 'credit_utilization_ratio' column for each customer.
    """

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
    aggregated['CreditLimit'] = aggregated['TotalPurchases']

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
        df (pandas.DataFrame): The transaction data, must include 'CustomerId', 'TransactionStartTime',
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

    # Group by CustomerId and calculate the minimum difference for recency
    df_recency = df.groupby('CustomerId')['dif'].min().reset_index()

    # Convert the dif to days to get the recency value
    df_recency['Recency'] = df_recency['dif'].dt.days

    # Merge recency back to the main dataframe
    df = df.merge(df_recency[['CustomerId', 'Recency']], on='CustomerId')

    # Drop the 'dif' column
    df.drop(columns=['dif'], inplace=True)

    # Calculate Frequency
    df['Frequency'] = df.groupby('CustomerId')['TransactionId'].transform('count')

    # Calculate Monetary Value
    df['Monetary'] = df.groupby('CustomerId')['Amount'].transform('sum') / df['Frequency']

    # Calculate Standard Deviation of Amounts
    df['StdDev'] = df.groupby('CustomerId')['Amount'].transform(lambda x: np.std(x, ddof=0))

    # Calculate the RFMS combined feature with custom weights
    #df['RFMS_score'] = df['Recency'] * r_weight + df['Frequency'] * f_weight + df['Monetary'] * m_weight + df['StdDev'] * s_weight

    # credit utilization ratio multiplied  with Monetary value
    if 'credit_utilization_ratio' in df.columns:
        df['Monetary'] *= df['credit_utilization_ratio']

    # Dropping duplicates to get one row per customer with RFMS values
    rfms_df = df.drop_duplicates(subset='CustomerId', keep='first')

    return rfms_df


import pandas as pd


def rfms_segmentation(df):
    """
    Performs RFM segmentation on the input DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing Recency, Frequency, Monetary, and StdDev features.

    Returns:
        pandas.DataFrame: Updated DataFrame with RFM segmentation features.
    """
    # Step 1: Choose the Suitable Scale
    scale = 3

    # Step 2: Define Intervals for Each Point
    # Calculate quartiles for Recency, Frequency, and Monetary features
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

    df['RFM_Score'] = df.Recency_Score.map(str) \
                                    + df.Frequency_Score.map(str) \
                                    + df.Monetary_Score.map(str)


    # Segment Customers High-risk and Low-risk
    def segment_customers(r, f, m, sd):
        if r == 1 and sd > 2:
            return 'High-risk'
        elif r >= 2 and f >= 2 and m >= 2:
            return 'Low-risk'
        else:
            return 'Low-risk'

    df['Segment'] = df.apply(
        lambda x: segment_customers(x['Recency_Score'], x['Frequency_Score'], x['Monetary_Score'], x['StdDev']), axis=1)

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


