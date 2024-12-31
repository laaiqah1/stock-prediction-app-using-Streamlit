import xgboost as xgb
import pandas as pd

# Train the XGBoost model
def train_xgboost_model(train_data, target):
    model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1, max_depth=5, alpha=10, n_estimators=10)
    model.fit(train_data, target)
    return model
