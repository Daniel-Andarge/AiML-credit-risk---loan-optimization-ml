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