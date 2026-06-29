from pathlib import Path
import json
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PARAMS_PATH = BASE_DIR / "model" / "model_params.json"
MODEL_PATH = BASE_DIR / "model" / "xgb_model.joblib"
SCALER_PATH = BASE_DIR / "model" / "scaler.joblib"

def load_model_params():
  if not MODEL_PARAMS_PATH.exists():
    raise FileNotFoundError(f"Model parameter file not found: {MODEL_PARAMS_PATH}")
  if not MODEL_PATH.exists():
    raise FileNotFoundError(f"XGBoost model file not found: {MODEL_PATH}")
  if not SCALER_PATH.exists():
    raise FileNotFoundError(f"StandardScaler file not found: {SCALER_PATH}")

  with open(MODEL_PARAMS_PATH, "r") as f:
    model_params = json.load(f)

  xgb_model = joblib.load(MODEL_PATH)
  scaler = joblib.load(SCALER_PATH)

  return xgb_model, scaler, model_params

def preprocess_input(input_data, model_params, scaler):
  # Convert input_data dictionary to a DataFrame
  input_df = pd.DataFrame([input_data])

  # Ensure all original categorical features are present, fill with a placeholder if not
  for col in model_params["original_categorical_features"]:
    if col not in input_df.columns:
      # Use the first category found during training or a sensible default if 'Unknown' isn't handled
      # For simplicity, we'll assume new categories might appear and let get_dummies handle them
      # Or, a more robust solution would be to get all unique categories from training data
      # and map/handle unseen ones. For now, we'll proceed assuming input aligns well.
      input_df[col] = '' # Placeholder for new categorical column

  # One-hot encode categorical features
  input_df_encoded = pd.get_dummies(input_df, columns=model_params["original_categorical_features"], drop_first=False, dtype=int)

  # Align columns to model_features (add missing, drop extra)
  # Create a DataFrame with all expected model features, filled with zeros
  aligned_df = pd.DataFrame(0, index=input_df_encoded.index, columns=model_params["model_features"])

  # Copy values from input_df_encoded to aligned_df for matching columns
  for col in input_df_encoded.columns:
    if col in aligned_df.columns:
      aligned_df[col] = input_df_encoded[col]

  # Scale numerical features
  numerical_features_to_scale_in_input = [col for col in model_params["numerical_features"] if col in aligned_df.columns]
  if numerical_features_to_scale_in_input:
    aligned_df[numerical_features_to_scale_in_input] = scaler.transform(aligned_df[numerical_features_to_scale_in_input])

  return aligned_df

def predict_sales_volume(input_data):
  xgb_model, scaler, model_params = load_model_params()

  # Preprocess the raw input data
  preprocessed_input = preprocess_input(input_data, model_params, scaler)

  # Make prediction using the XGBoost model
  prediction = xgb_model.predict(preprocessed_input)

  return float(prediction[0]) # XGBoost predict returns an array, take the first element
