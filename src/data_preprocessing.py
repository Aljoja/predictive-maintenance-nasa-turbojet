# src/data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(series: int = 1) -> tuple:
    """Load the training and test data from CSV files."""
    # Check if number of series is valid
    if series not in [1, 2, 3, 4]:
        raise ValueError("Series must be one of [1, 2, 3, 4]")

    # Creat paths
    train_data_path = f'data/CMaps/train_FD00{series}.txt'
    test_data_path = f'data/CMaps/test_FD00{series}.txt'
    rul_data_path = f'data/CMaps/RUL_FD00{series}.txt'

    # Creat column names # TODO: the columns are wrong, also ignoring first column?
    col_names_train = [
        'engine_id', 'time_in_cycles', 
        'operational_setting_1', 'operational_setting_2', 'operational_setting_3'
        ] + [f'sensor_{i}' for i in range(1, 27)] # This creates sensor_1 to sensor_20

    col_names_test = [
        'engine_id', 'time_in_cycles', 
        'operational_setting_1', 'operational_setting_2', 'operational_setting_3'
        ] + [f'sensor_{i}' for i in range(1, 27)] # This creates sensor_1 to sensor_21

    
    # Read the data into pandas dataframes
    train_df = pd.read_csv(train_data_path, sep=' ', header=None, names=col_names_train)
    test_df = pd.read_csv(test_data_path, sep=' ', header=None, names=col_names_test)
    rul_df = pd.read_csv(rul_data_path, header=None, names=['RUL']) # this is the rul of the test data

    return train_df, test_df, rul_df

def preprocess_data(train_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the train data by adding RUL values."""
    # Calculate RUL values for the training data
    EOL = []
    for id in train_df['engine_id']:
        EOL.append(((train_df[train_df['engine_id'] == id]['time_in_cycles']).values)[-1])

    # Add RUL values to the test data
    train_df['RUL'] = EOL

    return train_df

def clean_data(df):
    """Clean the data by dropping unnecessary columns and handling missing values."""
    # Drop columns that are not useful (e.g., engine_id and time_in_cycles)
    df = df.drop(columns=['engine_id', 'time_in_cycles'])
    
    # Handle missing values by filling with the median of each column
    # df.fillna(df.median(), inplace=True)

    # Drop columns that are completely empty
    df.dropna(axis=1, how='all', inplace=True)
    
    return df

def normalize_data(train_df, test_df):
    """Normalize the sensor data using StandardScaler."""
    sensor_columns = [f'sensor_{i}' for i in range(1, 22)]
    scaler = StandardScaler()
    
    # Fit on training data and transform both train and test data - makes ML models learn better and faster
    train_df[sensor_columns] = scaler.fit_transform(train_df[sensor_columns])
    test_df[sensor_columns] = scaler.transform(test_df[sensor_columns])
    
    return train_df, test_df