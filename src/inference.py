# src/inference.py

import joblib
import xgboost as xgb
from tensorflow.keras.models import load_model
import pandas as pd
from src.feature_engineering import create_rolling_features, create_delta_features, create_lag_features
from src.data_preprocessing import load_data, clean_data, normalize_data

def load_model_from_file(model_name):
    """Load a trained model from the models directory."""
    if model_name.endswith('.json'):
        model = xgb.XGBRegressor()
        model.load_model(f'models/{model_name}')
    elif model_name.endswith('.pkl'):
        model = joblib.load(f'models/{model_name}')
    else:
        model = load_model(f'models/{model_name}')
    
    return model

def make_inference(model, X):
    """Make predictions using the trained model."""
    predictions = model.predict(X)
    return predictions

def main():
    # Load and preprocess new data
    train_df, test_df = load_data('../data/raw/train_FD001.txt', '../data/raw/test_FD001.txt')
    test_df = clean_data(test_df)
    test_df = create_rolling_features(test_df)
    test_df = create_delta_features(test_df)
    test_df = create_lag_features(test_df)

    # Define features for prediction
    X_test = test_df.drop(columns=['RUL'])

    # Load the trained model
    model_name = 'random_forest'  # Replace with the model you want to use
    model = load_model_from_file(f'{model_name}_model')

    # Make predictions
    predictions = make_inference(model, X_test)
    
    print(f"Predictions: {predictions[:10]}")  # Print the first 10 predictions

if __name__ == "__main__":
    main()
