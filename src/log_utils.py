from datetime import datetime
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "data" / "monitoring_logs.csv"

def log_prediction(
    product_position,
    promotion,
    product_category,
    seasonal,
    terms,
    section,
    season,
    material,
    origin,
    price,
    prediction,
    actual_sales_volume,
    latency_ms,
    feedback_score,
    feedback_text
):
  error = float(actual_sales_volume) - float(prediction)
  absolute_error = abs(error)

  row = {
      "timestamp": datetime.utcnow().isoformat(),
      "Product Position": product_position,
      "Promotion": promotion,
      "Product Category": product_category,
      "Seasonal": seasonal,
      "terms": terms,
      "section": section,
      "season": season,
      "material": material,
      "origin": origin,
      "price": price,
      "prediction": float(prediction),
      "actual_sales_volume": float(actual_sales_volume),
      "error": error,
      "absolute_error": absolute_error,
      "latency_ms": float(latency_ms),
      "feedback_score": int(feedback_score),
      "feedback_text": feedback_text or ""
  }

  LOG_PATH.parent.mkdir(parents = True, exist_ok = True)

  df_new = pd.DataFrame([row])

  if LOG_PATH.exists():
    df_new.to_csv(LOG_PATH, mode = "a", header = False, index = False)
  else:
    df_new.to_csv(LOG_PATH, index = False)

  return row

def load_monitoring_logs():
  if not LOG_PATH.exists():
    return pd.DataFrame()

  return pd.read_csv(LOG_PATH, parse_dates = ["timestamp"])
