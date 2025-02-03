# src/model.py
from sklearn.ensemble import RandomForestRegressor
import joblib
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_random_forest(X_train, y_train):
    # Initialize and train Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def train_xgboost(X_train, y_train):
    # Prepare data for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse'}
    model = xgb.train(params, dtrain, num_boost_round=100)
    return model

def train_lstm(X_train, y_train):
    # Reshape data for LSTM input
    X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=False, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output layer for regression task

    # Compile and train the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train_lstm, y_train, epochs=10, batch_size=32)
    
    return model

def evaluate_model(model, X_val, y_val):
    # Make predictions on the validation set
    if isinstance(model, RandomForestRegressor) or isinstance(model, xgb.Booster):
        val_predictions = model.predict(X_val)
    elif isinstance(model, Sequential):
        X_val_lstm = X_val.values.reshape((X_val.shape[0], 1, X_val.shape[1]))
        val_predictions = model.predict(X_val_lstm)

    # Evaluate the model
    mae = mean_absolute_error(y_val, val_predictions)
    rmse = mean_squared_error(y_val, val_predictions, squared=False)
    
    return mae, rmse

def save_model(model, model_name):
    # Save the trained model
    joblib.dump(model, f'models/{model_name}.pkl')