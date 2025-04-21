# src/model.pySkipping analyzing "joblib": module is installed, but missing library stubs or py.typed markerMypyimport-u
from sklearn.ensemble import RandomForestRegressor
import joblib
import xgboost as xgb
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from src.model_definition import create_random_forest, create_xgboost, create_lstm
from src.feature_engineering import create_rolling_features, create_delta_features, create_lag_features
from src.data_preprocessing import load_data, clean_data, normalize_data
import logging
from src.utils import create_logger, save_model_to_file

def train_random_forest(X_train, y_train)-> RandomForestRegressor:
    # Initialize and train Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def train_xgboost(X_train, X_val, y_train, y_val):
    # Prepare data for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse'}
    model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dval, 'eval')])
    return model, dval 

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
    if isinstance(model, RandomForestRegressor):
        val_predictions = model.predict(X_val)
    elif isinstance(model, xgb.Booster):
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

def main():

    logger = create_logger()
    
    # Load and preprocess data
    logger.info('Loading and preprocessing data.')
    train_df, test_df = load_data(
        train_path = 'data/CMaps/train_FD001.txt',
        test_path = 'data/CMaps/test_FD001.txt'
        )
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)
    train_df, test_df = normalize_data(train_df, test_df)

    # Feature engineering
    train_df = create_rolling_features(train_df)
    test_df = create_rolling_features(test_df)
    train_df = create_delta_features(train_df)
    test_df = create_delta_features(test_df)
    train_df = create_lag_features(train_df)
    test_df = create_lag_features(test_df)

    # Define features and target
    X = train_df.drop(columns=['RUL'])
    y = train_df['RUL']

    # Train models
    logger.info('Training Random Forest model started.')
    rf_model = train_random_forest(X, y)
    logger.info(f'Random Forest training completed.')


    xgb_model = train_xgboost(X, y)

    # Prepare data for LSTM
    X_train_lstm = X.values.reshape((X.shape[0], 1, X.shape[1]))
    lstm_model = train_lstm(X_train_lstm, y)

    # Evaluate models
    rf_mae, rf_rmse = evaluate_model(rf_model, X, y)
    xgb_mae, xgb_rmse = evaluate_model(xgb_model, X, y)
    lstm_mae, lstm_rmse = evaluate_model(lstm_model, X, y)

    logger.info(f'Random Forest - MAE: {rf_mae}, RMSE: {rf_rmse}')
    # print(f"Random Forest - MAE: {rf_mae}, RMSE: {rf_rmse}")
    print(f"XGBoost - MAE: {xgb_mae}, RMSE: {xgb_rmse}")
    print(f"LSTM - MAE: {lstm_mae}, RMSE: {lstm_rmse}")

    # Save the models
    save_model(rf_model, 'random_forest')
    save_model(xgb_model, 'xgboost')
    save_model(lstm_model, 'lstm')

if __name__ == "__main__":
    main()