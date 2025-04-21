# src/model_definition.py

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_random_forest():
    """Create a Random Forest Regressor model."""
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    return rf_model

def create_xgboost():
    """Create an XGBoost model."""
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42)
    return xgb_model

def create_lstm(input_shape):
    """Create an LSTM model for time-series data."""
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output layer for regression task
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model