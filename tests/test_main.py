import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predict_utils import load_model_params, predict_sales_volume
from log_utils import log_prediction

def test_model_params_exists():
  xgb_model, scaler, model_params = load_model_params()

  assert "model_features" in model_params
  assert "original_categorical_features" in model_params
  assert "numerical_features" in model_params

def test_prediction_output_is_numeric():
  # Updated input_data to use Zara product features
  input_data = {
      "Product Position": "Aisle",
      "Promotion": "No",
      "Product Category": "clothing",
      "Seasonal": "No",
      "terms": "jackets",
      "section": "WOMAN",
      "season": "Autumn",
      "material": "Cotton",
      "origin": "Spain",
      "price": 50.0
  }

  prediction = predict_sales_volume(input_data)

  assert isinstance(prediction, float)

def test_log_prediction_creates_correct_error():
  # Updated log_prediction call to use Zara product features
  row = log_prediction(
      product_position = "Aisle",
      promotion = "No",
      product_category = "clothing",
      seasonal = "No",
      terms = "jackets",
      section = "WOMAN",
      season = "Autumn",
      material = "Cotton",
      origin = "Spain",
      price = 50.0,
      prediction = 100.0,
      actual_sales_volume = 85.0,
      latency_ms = 12.5,
      feedback_score = 4,
      feedback_text = "Test log"
  )

  assert row["absolute_error"] == 15.0
  assert row["feedback_score"] == 4
