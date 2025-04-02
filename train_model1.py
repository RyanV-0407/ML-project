import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import time
from feature_engineering import add_time_features
from hyperparameter_tuning import tune_hyperparameters

# Load dataset
df = pd.read_csv("realistic_energy_forecast_dataset.csv", parse_dates=["Timestamp"])

# Feature engineering
df = add_time_features(df)

# Define features and target
features = [
    "Temperature_C", "Humidity_%", "Wind_Speed_mps", "Solar_Radiation_Wm2",
    "Industrial_Usage_kWh", "Residential_Usage_kWh", "Commercial_Usage_kWh",
    "Grid_Frequency_Hz", "Voltage_Level_V", "Renewable_Energy_Contribution_%",
    "hour", "dayofweek", "month", "Holiday_Indicator", "Weekday"
]
target = "Energy_Consumption_kWh"

X = df[features]
y = df[target]

# Train-test split
train_size = int(len(df) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Hyperparameter tuning
best_params = tune_hyperparameters(X_train, y_train, X_test, y_test, n_trials=30)

# Fallback parameters
if not best_params:
    best_params = {
        'learning_rate': 0.05,
        'num_leaves': 31,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'min_data_in_leaf': 20
    }

# Prepare datasets
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Model parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': best_params.get('learning_rate', 0.05),
    'num_leaves': best_params.get('num_leaves', 31),
    'verbose': -1,
    'bagging_freq': 1,
    'subsample': best_params.get('subsample', 1.0),
    'colsample_bytree': best_params.get('colsample_bytree', 1.0),
    'min_data_in_leaf': best_params.get('min_data_in_leaf', 20)
}

# Train model
start_time = time.time()
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[test_data],
    callbacks=[
        lgb.log_evaluation(50),
        lgb.early_stopping(50)
    ]
)
train_time = time.time() - start_time

# Predictions and metrics
predictions = model.predict(X_test, num_iteration=model.best_iteration)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
metrics = {
    'r2': r2_score(y_test, predictions),
    'rmse': rmse,
    'mae': mean_absolute_error(y_test, predictions),
    'train_time': train_time,
    'last_demand': y_test.iloc[-1],
    'data_points': len(X_train)   # Save last actual demand
}

# Save artifacts
joblib.dump(model, "energy_forecast_model.pkl")
joblib.dump(features, "model_features.pkl")
joblib.dump(metrics, "model_metrics.pkl")
joblib.dump(y_test.iloc[-1], "last_demand.pkl")  # Save last demand separately

print("Model and artifacts saved successfully!")