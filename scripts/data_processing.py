# data_processing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler,OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

def load_and_clean_data(data):
    # Removing duplicates
    data = data.drop_duplicates(keep="first")
    return data
def drop_columns(df,columns_to_drop):
    data = df.drop(columns=columns_to_drop)
    return data
def column_detail(df,categorical_columns):
    summary_df = pd.DataFrame({
    'Column': categorical_columns,
    'DataType': [df[col].dtype for col in categorical_columns],
    'NumUniqueValues': [df[col].nunique() for col in categorical_columns]
     })
    return summary_df
def distribution(df):
    print(f'Gender Distribution:\n{df.Gender.value_counts()}')
    print(f'Title Distribution:\n {df.Title.value_counts()}')
    print(f'Marital Status Distribution:\n {df.MaritalStatus.value_counts()}')

def title_to_gender_map(df,title_to_gender_map):
    
    # Fill missing Gender based on Title
    df['Gender'] = df.apply(lambda row: title_to_gender_map.get(row['Title'], row['Gender']), axis=1)

    # Display the updated Gender counts
    return df
    print(df['Gender'].value_counts())
   
def encoder(method, dataframe, columns_label, columns_onehot):
    
    if method == 'labelEncoder':      
        df_lbl = dataframe.copy()
        for col in columns_label:
            label = LabelEncoder()
            label.fit(list(dataframe[col].values))
            df_lbl[col] = label.transform(df_lbl[col].values)
        return df_lbl
    
    elif method == 'oneHotEncoder':
        df_oh = dataframe.copy()
        df_oh= pd.get_dummies(data=df_oh, prefix='ohe', prefix_sep='_',
                       columns=columns_onehot, drop_first=True, dtype='int8')
        return df_oh


def column_encoder(df,label_encode_cols,one_hot_encode_cols,frequency_encode_cols):
    print()

    # 1. Label Encoding for low cardinality columns
    label_encoders = {}
    for feature in label_encode_cols:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
        label_encoders[feature] = le

    # One-Hot Encoding for moderate cardinality features
    # 2. One-Hot Encoding for moderate cardinality columns
    df = pd.get_dummies(df, columns=one_hot_encode_cols, drop_first=True)
    # 3. Frequency Encoding for high cardinality columns
    for col in frequency_encode_cols:
        freq = df[col].value_counts()
        df[col + '_freq'] = df[col].map(freq)
        df = df.drop(columns=[col])
    return df

def scaler(method, data, columns_scaler):    
    if method == 'standardScaler':        
        Standard = StandardScaler()
        df_standard = data.copy()
        df_standard[columns_scaler] = Standard.fit_transform(df_standard[columns_scaler])        
        return df_standard
        
    elif method == 'minMaxScaler':        
        MinMax = MinMaxScaler()
        df_minmax = data.copy()
        df_minmax[columns_scaler] = MinMax.fit_transform(df_minmax[columns_scaler])        
        return df_minmax
    
    elif method == 'npLog':        
        df_nplog = data.copy()
        df_nplog[columns_scaler] = np.log(df_nplog[columns_scaler])        
        return df_nplog
    
    return data
