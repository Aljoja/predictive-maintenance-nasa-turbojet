# src/data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(train_path, test_path):
    """Load the training and test data from CSV files."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def clean_data(df):
    """Clean the data by dropping unnecessary columns and handling missing values."""
    # Drop columns that are not useful (e.g., engine_id and time_in_cycles)
    df = df.drop(columns=['engine_id', 'time_in_cycles'])
    
    # Handle missing values by filling with the median of each column
    df.fillna(df.median(), inplace=True)
    
    return df

def normalize_data(train_df, test_df):
    """Normalize the sensor data using StandardScaler."""
    sensor_columns = [f'sensor_{i}' for i in range(1, 22)]
    scaler = StandardScaler()
    
    # Fit on training data and transform both train and test data
    train_df[sensor_columns] = scaler.fit_transform(train_df[sensor_columns])
    test_df[sensor_columns] = scaler.transform(test_df[sensor_columns])
    
    return train_df, test_df