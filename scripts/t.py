import pandas as pd

def process_OnTime_payments(df):
    # Convert the 'TransactionStartTime' column to datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

    # Define a hypothetical payment due period (e.g., 30 days)
    payment_due_period = pd.Timedelta(days=30)

    # Split the data into debits and credits
    debits = df[df['Amount'] > 0].copy()
    credits = df[df['Amount'] < 0].copy()

    # Create an empty DataFrame to store the payment history
    payment_history = df.copy()
    payment_history['OnTimePayment'] = False

    # Iterate over each debit and find corresponding credits
    for _, debit in debits.iterrows():
        customer_id = debit['CustomerId']
        debit_time = debit['TransactionStartTime']
        due_date = debit_time + payment_due_period

        # Find matching credits within the due period
        matching_credits = credits[(credits['CustomerId'] == customer_id) & (credits['TransactionStartTime'] <= due_date)]

        for _, credit in matching_credits.iterrows():
            on_time = credit['TransactionStartTime'] <= due_date
            payment_history.loc[(payment_history['CustomerId'] == customer_id) & (payment_history['TransactionStartTime'] == credit['TransactionStartTime']), 'OnTimePayment'] = on_time

    # Aggregate the payment history to get the number of on-time payments and total payments per customer
    df_final = payment_history.groupby('CustomerId').agg(
        OnTimePayments=('OnTimePayment', 'sum')
    ).reset_index()

    return df_final

def calculate_credit_utilization_ratio(df):
    """
    Calculates the customer's credit utilization ratio for BNPL transactions.

    Args:
        df (pandas.DataFrame): The BNPL transaction data.

    Returns:
        pandas.DataFrame: The original DataFrame with an added 'credit_utilization_ratio' column.
    """
    # Identify BNPL purchases and repayments
    purchases = df[df['Amount'] > 0]['Amount'].sum()
    repayments = abs(df[df['Amount'] < 0]['Amount'].sum())

    # Calculate the customer's current BNPL balance
    current_balance = purchases - repayments

    # Calculate the customer's total BNPL credit limit
    total_credit_limit = purchases

    # Calculate the credit utilization ratio
    df['credit_utilization_ratio'] = current_balance / total_credit_limit

    return df


def create_rfms_features(df):
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], format='%Y-%m-%d %H:%M:%S%z')

        # Calculate Recency
    max_date = df['TransactionStartTime'].max()
    df['dif'] = max_date - df['TransactionStartTime']

        # Group by 'AccountId' and calculate the minimum difference for recency
    df_recency = df.groupby('AccountId')['dif'].min().reset_index()

        # Convert the 'dif' to days to get the recency value
    df_recency['Recency'] = df_recency['dif'].dt.days

        # Add the OnTimePayments value to the Recency
    df_recency['Recency'] = df_recency['Recency'] + df['OnTimePayments']

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
    rfms_df.loc[:, '>OnTimePayment_avg'] = (rfms_df['StdDev'] > rfms_mean['StdDev']).astype(int)

    return rfms_df