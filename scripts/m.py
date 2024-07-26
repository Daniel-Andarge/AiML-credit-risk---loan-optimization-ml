
def visualize_rfms_space(df):
    # Extract the RFMS scores
    r_score = df['Recency']
    f_score = df['Frequency']
    m_score = df['Monetary']

    #   Visualize the RFMS space.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['Recency'], df['Frequency'], df['Monetary'])
    ax.set_xlabel('Recency')
    ax.set_ylabel('Frequency')
    ax.set_zlabel('Monetary Value')
    plt.title('RFMS Space')
    plt.show()

    # Defining  the boundary between high and low RFMS scores
    r_threshold = np.percentile(r_score, 60)
    f_threshold = np.percentile(f_score, 60)
    m_threshold = np.percentile(m_score, 60)
    ax.axhline(y=rfms_threshold, color='r', linestyle='--', label='RFMS Threshold')
    ax.legend()

    plt.show()

    return  r_threshold, f_threshold, m_threshold