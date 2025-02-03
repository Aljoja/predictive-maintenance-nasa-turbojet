import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(train_df, test_df):
    # Assuming train_df and test_df are the raw dataframes
    # Normalize sensor columns using StandardScaler
    scaler = StandardScaler()
    sensor_columns = [f'sensor_{i}' for i in range(1, 22)]
    
    train_df[sensor_columns] = scaler.fit_transform(train_df[sensor_columns])
    test_df[sensor_columns] = scaler.transform(test_df[sensor_columns])

    return train_df, test_df