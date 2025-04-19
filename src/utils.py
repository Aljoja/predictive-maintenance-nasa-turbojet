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

def load_dataset():
    """Function to load the test and train datasets into Pandas dataframes"""

    # Datasets paths
    train_data_path = '../data/CMaps/train_FD001.txt'
    test_data_path = '../data/CMaps/test_FD001.txt'

    # Define the column names based on the dataset's structure
    col_names = [
        'engine_id', 'time_in_cycles', 
        'operational_setting_1', 'operational_setting_2', 'operational_setting_3'
    ] + [f'sensor_{i}' for i in range(1, 27)] 

    # Load the train and test datasets
    train_df = pd.read_csv(train_data_path, sep=' ', header=None, names=col_names)
    test_df = pd.read_csv(test_data_path, sep=' ', header=None, names=col_names)

    return train_df, test_df