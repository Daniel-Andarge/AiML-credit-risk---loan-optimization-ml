
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import math
import logging
import math

logging.basicConfig(filename='eda_analysis.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Set general aesthetics for the plots
sns.set_style("whitegrid")
def eda_overview(df):
    """
    Performs Exploratory Data Analysis (EDA) to provide an overview of the dataset.

    Args:
    - df (DataFrame): Input DataFrame.

    Returns:
    - None
    """
    try:
        logging.info("Started EDA overview")

        if df.empty:
            logging.error("Empty DataFrame provided.")
            print("DataFrame is empty.")
            return

        # Display the number of rows, columns, and data types
        logging.info(f"Number of rows: {df.shape[0]}")
        logging.info(f"Number of columns: {df.shape[1]}")
        logging.info("Data Types:")
        logging.info(df.dtypes.to_string())

        print("Dataset Overview:")
        print("Number of rows:", df.shape[0])
        print("Number of columns:", df.shape[1])
        print("\nData Types:")
        print(df.dtypes)
    
        logging.info("Completed EDA overview")

    except Exception as e:
        logging.error(f"An error occurred during EDA: {str(e)}")
        print("An error occurred during EDA:", str(e))


def descriptive_stat(df):
    """
    Analyze credit scoring using the provided dataset.

    Args:
    - df (DataFrame): Input DataFrame containing credit scoring data.

    Returns:
    - None
    """
    try:
        logging.info("Started analyzing credit scoring")
        
        if df.empty:
            logging.error("Empty DataFrame provided for analysis.")
            return
        
        # Summary statistics
        logging.info("Computing summary statistics")
        credit_stats = df.describe()
        print("Summary Statistics:")
        print(credit_stats)
   
        logging.info("Completed analysis")
    
    except Exception as e:
        logging.error(f"An error occurred during credit scoring analysis: {str(e)}")



def visualize_numerical_distribution(df):
    """
    Visualize the distribution of numerical features in the provided dataset.

    Args:
    - df (DataFrame): Input DataFrame containing numerical features.

    Returns:
    - None
    """
    logging.info("Started visualizing numerical feature distribution")
    
    # Extract numerical features
    numerical_features = df.select_dtypes(include=['float64', 'int64'])
    
    # Visualize distribution of numerical features
    num_cols = 2
    num_rows = math.ceil(len(numerical_features.columns) / num_cols)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 6 * num_rows))
    axes = axes.ravel()
    
    for i, column in enumerate(numerical_features.columns):
        sns.histplot(df[column], kde=True, bins=20, color='skyblue', ax=axes[i])
        axes[i].set_title(f'Distribution of {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    logging.info("Completed visualization")
    
    print("Observations:")
    for column in numerical_features.columns:
        skewness = df[column].skew()
        if skewness > 1:
            print(f"{column} is right-skewed with skewness {skewness}. This indicates a longer tail on the right side.")
        elif skewness < -1:
            print(f"{column} is left-skewed with skewness {skewness}. This indicates a longer tail on the left side.")
        else:
            print(f"{column} is approximately symmetric with skewness {skewness}.")



def analyze_categorical_distribution(df):
    """
    Analyze the distribution of categorical features in the provided dataset.

    Args:
    - df (DataFrame): Input DataFrame containing categorical features.

    Returns:
    - None
    """
    try:
        logging.info("Started analyzing categorical feature distribution")

        # Limit the DataFrame to the first 2000 rows
        df = df.head(2000)

        # Extract categorical features
        categorical_features = df.select_dtypes(include=['object'])

        if categorical_features.empty:
            logging.error("No categorical features found in the provided DataFrame.")
            return

        # Aggregate data before plotting
        aggregated_data = {col: df[col].value_counts() for col in categorical_features.columns}

        # Calculate number of rows and columns for subplots
        num_cols = 2
        num_rows = math.ceil(len(categorical_features.columns) / num_cols)

        # Create the figure and subplots
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 4 * num_rows))

        for i, (column, value_counts) in enumerate(aggregated_data.items()):
            row = i // num_cols
            col = i % num_cols
            axes[row, col].bar(value_counts.index, value_counts.values)
            axes[row, col].set_title(f'Distribution of {column}')
            axes[row, col].set_xlabel(column)
            axes[row, col].set_ylabel('Count')

        plt.tight_layout()
        plt.show()

        logging.info("Completed analysis")

        print("Observations:")
        for column, value_counts in aggregated_data.items():
            unique_categories = len(value_counts)
            print(f"{column} has {unique_categories} unique categories.")

    except Exception as e:
        logging.error(f"An error occurred during categorical feature analysis: {str(e)}")
        print(f"An error occurred during categorical feature analysis: {str(e)}")



def correlation_analysis(df):
    """
    Perform correlation analysis on numerical features in the provided dataset.

    Args:
    - df (DataFrame): Input DataFrame containing numerical features.

    Returns:
    - None
    """
    try:
        logging.info("Started correlation analysis")

        # Extract numerical features
        numerical_features = df.select_dtypes(include=['float64', 'int64'])

        if numerical_features.empty:
            logging.error("No numerical features found in the provided DataFrame.")
            return

        # Compute correlation matrix
        corr_matrix = numerical_features.corr()

        # Visualize correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix of Numerical Features')
        plt.show()

        logging.info("Completed correlation analysis")

        print("Observations:")
        for col1 in numerical_features.columns:
            for col2 in numerical_features.columns:
                if col1 != col2:
                    correlation = df[col1].corr(df[col2])
                    if abs(correlation) >= 0.7:
                        print(f"There is a strong correlation between '{col1}' and '{col2}' (correlation = {correlation:.2f}).")
                    elif abs(correlation) >= 0.5:
                        print(f"There is a moderate correlation between '{col1}' and '{col2}' (correlation = {correlation:.2f}).")
                    elif abs(correlation) >= 0.3:
                        print(f"There is a weak correlation between '{col1}' and '{col2}' (correlation = {correlation:.2f}).")
                    else:
                        print(f"There is a very weak or no correlation between '{col1}' and '{col2}' (correlation = {correlation:.2f}).")

    except Exception as e:
        logging.error(f"An error occurred during correlation analysis: {str(e)}")



def identify_missing_values(df):
    """
    Identifies missing values in the provided dataset.

    Args:
    - df (DataFrame): Input DataFrame.

    Returns:
    - None
    """
    try:
        logging.info("Started identifying missing values")
        
        if df.empty:
            logging.error("Empty DataFrame provided.")
            return
        
        # Check for missing values
        missing_values = df.isnull().sum()
        
        if missing_values.sum() == 0:
            logging.info("No missing values found in the dataset.")
            print("No missing values found.")
            return
        
        # Display missing values information
        logging.info("Identified missing values")
        print("Missing Values:")
        print(missing_values)
        
        # Calculate percentage of missing values
        total_cells = df.size
        total_missing = missing_values.sum()
        percentage_missing = (total_missing / total_cells) * 100
        print(f"Percentage of missing values: {percentage_missing:.2f}%")
        
        logging.info("Completed identifying missing values")
        
        # Observations
        print("\nObservations:")
        # Your observations based on missing values analysis
        
    except Exception as e:
        logging.error(f"An error occurred during missing values identification: {str(e)}")




def detect_outliers(df):
    """
    Identifies outliers in the provided dataset using box plots.

    Args:
    - df (DataFrame): Input DataFrame.

    Returns:
    - None
    """
    try:
        logging.info("Started outlier detection")
        
        if df.empty:
            logging.error("Empty DataFrame provided.")
            return
        
        # Select numerical features for outlier detection
        numerical_features = df.select_dtypes(include=['float64', 'int64'])
        
        if numerical_features.empty:
            logging.error("No numerical features found in the provided DataFrame.")
            return
        
        # Visualize outliers using box plots
        num_cols = 2
        num_rows = math.ceil(len(numerical_features.columns) / num_cols)
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 6 * num_rows))
        axes = axes.ravel()
        
        for i, column in enumerate(numerical_features.columns):
            sns.boxplot(x=df[column], ax=axes[i])
            axes[i].set_title(f'Boxplot of {column}')
            axes[i].set_xlabel(column)
        
        plt.tight_layout()
        plt.show()
        
        logging.info("Completed outlier detection")
        
        # Observations
        print("\nObservations:")
        for column in numerical_features.columns:
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
            if outliers.empty:
                print(f"No outliers detected in '{column}'.")
            else:
                print(f"Detected outliers in '{column}':")
                print(outliers)
        
    except Exception as e:
        logging.error(f"An error occurred during outlier detection: {str(e)}")


def remove_outliers(df):
    """
    Removes outliers from a dfFrame based on the IQR method.

    Parameters:
    df (pd.dfFrame): Input dfFrame from which to remove outliers.

    Returns:
    pd.dfFrame: dfFrame with outliers removed.
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
    Encodes categorical variables in the input dfframe using Label Encoding.

    Parameters:
    df (pandas.dfFrame): The input dfframe containing the df.

    Returns:
    pandas.dfFrame: The dfframe with the categorical variables encoded and converted to numerical type.
    """
    try:
        # Copy the dfframe to avoid modifying the original
        df = df.copy()

        # Label Encoding
        label_encoder = LabelEncoder()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = label_encoder.fit_transform(df[col])

        return df
    except Exception as e:
        print(f"Error occurred during encoding categorical variables: {e}")
        return None