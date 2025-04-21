# src/feature_engineering.py

import pandas as pd

def create_rolling_features(df, window_size=5):
    """Create rolling mean and standard deviation features for each sensor."""
    sensor_columns = [f'sensor_{i}' for i in range(1, 22)]
    
    for sensor in sensor_columns:
        df[f'{sensor}_rolling_mean'] = df[sensor].rolling(window=window_size).mean()
        df[f'{sensor}_rolling_std'] = df[sensor].rolling(window=window_size).std()
    
    df.dropna(inplace=True)  # Drop NaN values resulting from rolling
    return df

def create_delta_features(df):
    """Create delta (difference) features between consecutive cycles for each sensor."""
    sensor_columns = [f'sensor_{i}' for i in range(1, 22)]
    
    for sensor in sensor_columns:
        df[f'{sensor}_delta'] = df[sensor] - df.groupby('engine_id')[sensor].shift(1)
    
    df.dropna(inplace=True)  # Drop NaN values resulting from shift
    return df

def create_lag_features(df, lag=1):
    """Create lag features (previous cycle values) for each sensor."""
    sensor_columns = [f'sensor_{i}' for i in range(1, 22)]
    
    for sensor in sensor_columns:
        df[f'{sensor}_lag_{lag}'] = df.groupby('engine_id')[sensor].shift(lag)
    
    df.dropna(inplace=True)  # Drop NaN values resulting from lag operation
    return df
