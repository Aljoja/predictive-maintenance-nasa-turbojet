import pandas as pd
from sklearn.preprocessing import StandardScaler
# src/utils.py

import os
import logging
import joblib
import xgboost as xgb
from tensorflow.keras.models import save_model


def create_logger(log_file='logs/project.log'):
    """Set up a logger to log messages to a file."""
    if not os.path.exists('logs'):
        os.makedirs('logs')  # Create the logs directory if it doesn't exist
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def save_model_to_file(model, model_name):
    """Save a trained model to a file."""
    model_path = f'models/{model_name}.pkl'

    if isinstance(model, xgb.Booster):
        model.save_model(model_path)
    elif isinstance(model, RandomForestRegressor):
        joblib.dump(model, model_path)
    else:
        save_model(model, f'models/{model_name}.h5')

    logging.info(f'Model saved as {model_name} at {model_path}')

def load_model_from_file(model_name):
    """Load a trained model from a file."""
    model_path = f'models/{model_name}'

    if model_name.endswith('.json'):
        model = xgb.XGBRegressor()
        model.load_model(model_path)
    elif model_name.endswith('.pkl'):
        model = joblib.load(model_path)
    else:
        model = load_model(model_path)

    logging.info(f'Model {model_name} loaded from {model_path}')
    return model

def check_directory_exists(directory):
    """Check if a directory exists, if not create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Directory {directory} created.")
    else:
        logging.info(f"Directory {directory} already exists.")

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