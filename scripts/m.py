
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
    dc_threshold = np.percentile(debit_credit_ratio, 40)
    tv_threshold = np.percentile(transaction_volatility, 40)

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
        df['Monetary'] >= m_threshold) & (df['DebitCreditRatio'] <= dc_threshold) & (
        df['TransactionVolatility'] <= tv_threshold), 'Classification'] = 'Low-risk'

    df['is_high_risk'] = (df['Classification'] == 'High-risk').astype(int)

    return df
