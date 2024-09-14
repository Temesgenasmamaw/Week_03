import pandas as pd
import numpy as np
import scipy.stats 
from scipy.stats import zscore

def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # dtype of missing values
    mis_val_dtype = df.dtypes

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent, mis_val_dtype], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns={0: 'Missing Values', 1: '% of Total Values', 2: 'Dtype'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


# def format_float(value):
#     return f'{value:,.2f}'


# def find_agg(df: pd.DataFrame, agg_column: str, agg_metric: str, col_name: str, top: int, order=False) -> pd.DataFrame:
#     new_df = df.groupby(agg_column)[agg_column].agg(agg_metric).reset_index(name=col_name). \
#         sort_values(by=col_name, ascending=order)[:top]
#     return new_df


def convert_bytes_to_megabytes(df, bytes_data):
    megabyte = 1 * 10e+5
    df[bytes_data] = df[bytes_data] / megabyte
    return df[bytes_data]


def fix_outlier(df, column):
    df[column] = np.where(df[column] > df[column].quantile(0.95), df[column].median(), df[column])
    return df[column]

def remove_outliers(df, column_to_process, z_threshold=3):
    # Apply outlier removal to the specified column
    z_scores = zscore(df[column_to_process])
    outlier_column = column_to_process + '_Outlier'
    df[outlier_column] = (np.abs(z_scores) > z_threshold).astype(int)
    df = df[df[outlier_column] == 0]  # Keep rows without outliers

    # Drop the outlier column as it's no longer needed
    df = df.drop(columns=[outlier_column], errors='ignore')

    return df
import numpy as np

import numpy as np
from scipy import stats

def remove_all_columns_outliers(df, method="iqr"):
    """
    Detect and remove outliers from all numerical columns in the DataFrame using either the IQR or Z-score method.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    method (str): The method to use for outlier detection, either 'iqr' or 'zscore'. Defaults to 'iqr'.

    Returns:
    pd.DataFrame: A DataFrame with outliers removed.
    """
    df_clean = df.copy()  # Copy the original DataFrame to avoid modifying it directly
    # df_clean.fillna(df_clean.mean(), inplace=True)
    
    if method == "iqr":
        # Loop through each numeric column in the DataFrame
        for column in df_clean.select_dtypes(include=[np.number]).columns:
            Q1 = df_clean[column].quantile(0.25)  # First quartile (25th percentile)
            Q3 = df_clean[column].quantile(0.75)  # Third quartile (75th percentile)
            IQR = Q3 - Q1  # Interquartile range
            
            # Calculate lower and upper bounds for detecting outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
             # Debugging output
            print(f"Column: {column}")
            print(f"  Q1: {Q1}, Q3: {Q3}, IQR: {IQR}, Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")           
            # Remove rows with outliers for the current column
            df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
    
    elif method == "zscore":
        # Apply Z-score method for detecting outliers
        z_scores = np.abs(stats.zscore(df_clean.select_dtypes(include=[np.number])))
        df_clean = df_clean[(z_scores < 3).all(axis=1)]
    
    else:
        raise ValueError("Method must be either 'iqr' or 'zscore'")
    
    return df_clean


