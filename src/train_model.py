from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent

TRAINING_DATA_PATH = BASE_DIR / "data" / "zara_training.csv"
MODEL_PARAMS_PATH = BASE_DIR / "model" / "model_params.json"
MODEL_PATH = BASE_DIR / "model" / "xgb_model.joblib"
SCALER_PATH = BASE_DIR / "model" / "scaler.joblib"

# Define the raw input features (before one-hot encoding) and numerical features
ORIGINAL_CATEGORICAL_FEATURES = ['Product Position', 'Promotion', 'Product Category', 'Seasonal', 'terms', 'section', 'season', 'material', 'origin']
NUMERICAL_FEATURES = ['price']

# Define the target variable
TARGET = 'Sales Volume'

def train_and_save_model():
  if not TRAINING_DATA_PATH.exists():
    raise FileNotFoundError(f"Training data not found: {TRAINING_DATA_PATH}")

  df = pd.read_csv(TRAINING_DATA_PATH)

  required_columns = ORIGINAL_CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [TARGET]
  missing_columns = set(required_columns) - set(df.columns)

  if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

  df = df.dropna(subset = required_columns)

  # Encoding
  df_encoded = df.copy()

  columns_to_drop = ['Product ID', 'url', 'name', 'description']

  if df_encoded['brand'].nunique() == 1:
    print(f"Dropping 'brand' column as it has only one unique value: {df_encoded['brand'].iloc[0]}")
    columns_to_drop.append('brand')

  if df_encoded['currency'].nunique() == 1:
    print(f"Dropping 'currency' column as it has only one unique value: {df_encoded['currency'].iloc[0]}")
    columns_to_drop.append('currency')

  df_encoded = df_encoded.drop(columns=columns_to_drop, errors='ignore')

  categorical_cols_found = df_encoded.select_dtypes(include='object').columns.tolist()
  print(f"\nCategorical columns to one-hot encode: {categorical_cols_found}")

  # Changed drop_first to False to ensure all desired dummy variables are created
  df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols_found, drop_first=False, dtype=int)

  print(f"Shape of the processed DataFrame after one-hot encoding: {df_encoded.shape}")

  # Define the specific features the model will be trained on after one-hot encoding and selection
  # This list must be updated if the original categorical features change or new features are added.
  # For now, we'll take all columns from df_encoded that are not the target as model features.
  # If specific features are desired, this list should be manually curated.
  MODEL_FEATURES = [col for col in df_encoded.columns if col != TARGET]

  # Scaling
  df_scaled = df_encoded.copy()

  # Ensure numerical features that need scaling are present
  numerical_cols_to_scale_actual = [col for col in NUMERICAL_FEATURES if col in df_scaled.columns and col != TARGET]

  print(f"Numerical columns to scale: {numerical_cols_to_scale_actual}")

  scaler = StandardScaler()

  if numerical_cols_to_scale_actual:
    df_scaled[numerical_cols_to_scale_actual] = scaler.fit_transform(df_scaled[numerical_cols_to_scale_actual])

  print(f"Shape of the processed DataFrame after scaling: {df_scaled.shape}")

  X_xgb = df_scaled[MODEL_FEATURES]
  y_xgb = df_scaled[TARGET]

  X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X_xgb, y_xgb, test_size=0.2, random_state=42)

  xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
  xgb_model.fit(X_train_xgb, y_train_xgb)

  y_pred_xgb = xgb_model.predict(X_test_xgb)

  mae_xgb = mean_absolute_error(y_test_xgb, y_pred_xgb)
  mse_xgb = mean_squared_error(y_test_xgb, y_pred_xgb)
  rmse_xgb = np.sqrt(mse_xgb)
  r2_xgb = r2_score(y_test_xgb, y_pred_xgb)

  model_params = {
      "model_features": MODEL_FEATURES,
      "original_categorical_features": ORIGINAL_CATEGORICAL_FEATURES,
      "numerical_features": NUMERICAL_FEATURES,
      "validation_mae": float(mae_xgb),
      "validation_rmse": float(rmse_xgb),
      "validation_r2": float(r2_xgb)
  }

  MODEL_PARAMS_PATH.parent.mkdir(parents = True, exist_ok = True)
  MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
  SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)

  with open(MODEL_PARAMS_PATH, "w") as f:
    json.dump(model_params, f, indent = 4)

  joblib.dump(xgb_model, MODEL_PATH)
  joblib.dump(scaler, SCALER_PATH)

  print("Model training completed")
  print(f"Validation R-Square: {r2_xgb:.2f}")
  print(f"Validation MAE: {mae_xgb:.2f}")
  print(f"Validation RMSE: {rmse_xgb:.2f}")
  print(f"Model parameters saved to: {MODEL_PARAMS_PATH}")
  print(f"XGBoost model saved to: {MODEL_PATH}")
  print(f"StandardScaler saved to: {SCALER_PATH}")

  return model_params

if __name__ == "__main__":
  train_and_save_model()
