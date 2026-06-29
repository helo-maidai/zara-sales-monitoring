import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

from predict_utils import load_model_params, predict_sales_volume
from log_utils import log_prediction, load_monitoring_logs

TRAINING_DATA_PATH = BASE_DIR / "data" / "zara_training.csv"
PRODUCTION_DATA_PATH = BASE_DIR / "data" / "zara_production.csv"

st.set_page_config(
    page_title = "Sales Volume Monitoring Dashboard",
    layout = "wide"
)

st.title("Sales Volume Prediction and Monitoring Dashboard")

st.write("This dashboard demonstrates how a deployed prediction model can be monitored using actual outcomes, latency, manual business feedback and input drift indicators.")

@st.cache_data
def load_training_data():
  return pd.read_csv(TRAINING_DATA_PATH)

@st.cache_data
def load_production_data():
  return pd.read_csv(PRODUCTION_DATA_PATH)

@st.cache_data
def load_metrics_params():
  # load_model_params returns (xgb_model, scaler, model_params_dict)
  # We only need the model_params_dict here for displaying metrics.
  _, _, metrics_params = load_model_params()
  return metrics_params

training_data = load_training_data()
production_data = load_production_data()
model_metrics = load_metrics_params()

tab_viz, tab1, tab2, tab3 = st.tabs([
 "Feature Distributions",
 "Prediction and Manual Feedback",
 "Monitoring Dashboard",
 "Agile Retrospective"
])

with tab_viz:
  st.header("Feature Distributions in Training Data")
  st.write("These visualizations show the distribution of key features from the training dataset, providing context for your input selections.")

  # Add filters for key categorical features
  st.subheader("Filter Training Data for Visualizations")
  col_filters1, col_filters2, col_filters3 = st.columns(3)

  with col_filters1:
    selected_promotion = st.selectbox(
      "Promotion",
      options=training_data['Promotion'].unique()
    )
    selected_seasonal = st.selectbox(
      "Seasonal",
      options=training_data['Seasonal'].unique()
    )

  with col_filters2:
    selected_section = st.multiselect(
      "Section",
      options=training_data['section'].unique(),
      default=training_data['section'].unique()
    )
    selected_season = st.multiselect(
      "Season",
      options=training_data['season'].unique(),
      default=training_data['season'].unique()
    )
    selected_terms = st.multiselect(
      "Terms",
      options=training_data['terms'].unique(),
      default=training_data['terms'].unique()
    )

  with col_filters3:
    selected_material = st.multiselect(
      "Material",
      options=training_data['material'].unique(),
      default=training_data['material'].unique()
    )
    selected_origin = st.multiselect(
      "Origin",
      options=training_data['origin'].unique(),
      default=training_data['origin'].unique()
    )

  # Apply filters to training data
  filtered_training_data = training_data[
      (training_data['Promotion'] == selected_promotion) &
      (training_data['Seasonal'] == selected_seasonal) &
      (training_data['section'].isin(selected_section)) &
      (training_data['season'].isin(selected_season)) &
      (training_data['terms'].isin(selected_terms)) &
      (training_data['material'].isin(selected_material)) &
      (training_data['origin'].isin(selected_origin))
  ]

  if filtered_training_data.empty:
    st.warning("No data available for the selected filters.")
  else:
    st.markdown("---  ")
    st.subheader("Visualizations for Filtered Training Data")
    
    # Visualization 1: Price Distribution
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.histplot(filtered_training_data['price'], bins=30, kde=True, ax=ax1, color='skyblue')
    ax1.set_title('Distribution of Price in Filtered Training Data', fontsize=14)
    ax1.set_xlabel('Price', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1) # Close the figure to prevent it from being displayed twice

    # Visualization 2: Product Position Distribution
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.countplot(y=filtered_training_data['Product Position'], order=filtered_training_data['Product Position'].value_counts().index, ax=ax2, palette='viridis')
    ax2.set_title('Distribution of Product Position in Filtered Training Data', fontsize=14)
    ax2.set_xlabel('Count', fontsize=12)
    ax2.set_ylabel('Product Position', fontsize=12)
    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # Visualization 3: Material Distribution
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.countplot(y=filtered_training_data['material'], order=filtered_training_data['material'].value_counts().index, ax=ax3, palette='magma')
    ax3.set_title('Distribution of Material in Filtered Training Data', fontsize=14)
    ax3.set_xlabel('Count', fontsize=12)
    ax3.set_ylabel('Material', fontsize=12)
    ax3.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

with tab1:
  st.header("Prediction and Manual Feedback")

  st.write("Use this section to simulate a business user generating a prediction and manually providing feedback after comparing the prediction with actual sales volume.")

  st.sidebar.header("Product Features")


  product_position = st.sidebar.multiselect("Product Position", ('Aisle', 'End-cap', 'Front of Store'), default=['Aisle'])

  product_position_for_prediction = product_position[0] if product_position else 'Aisle'

  promotion = st.sidebar.checkbox("Promotion", False)
  product_category = st.sidebar.selectbox("Product Category", ('clothing',))
  seasonal = st.sidebar.checkbox("Seasonal", False)

  terms = st.sidebar.selectbox("Terms", ('jackets', 'jeans', 'shoes', 'sweaters', 't-shirts'))
  section = st.sidebar.selectbox("Section", ('MAN', 'WOMAN'))
  season = st.sidebar.selectbox("Season", ('Autumn', 'Spring', 'Summer', 'Winter'))
  material = st.sidebar.selectbox("Material", ('Acrylic', 'Cotton', 'Denim', 'Linen', 'Linen Blend', 'Polyester', 'Satin', 'Silk', 'Viscose', 'Wool', 'Wool Blend'))
  origin = st.sidebar.selectbox("Origin", ('Argentina', 'Bangladesh', 'Brazil', 'Cambodia', 'China', 'India', 'Morocco', 'Pakistan', 'Portugal', 'Spain', 'Turkey', 'Vietnam'))
  price = st.sidebar.number_input("Price", min_value = 0.0, value = 50.0, step = 1.0)

  input_data = {
    "Product Position": product_position_for_prediction,
    "Promotion": 'Yes' if promotion else 'No',
    "Product Category": product_category,
    "Seasonal": 'Yes' if seasonal else 'No',
    "terms": terms,
    "section": section,
    "season": season,
    "material": material,
    "origin": origin,
    "price": price
  }

  input_df = pd.DataFrame([input_data])

  st.subheader("Input Data")
  st.dataframe(input_df)

  if "prediction" not in st.session_state:
    st.session_state["prediction"] = None

  if "latency_ms" not in st.session_state:
    st.session_state["latency_ms"] = None

  if st.button("Run Prediction"):
    start_time = time.time()

    # Use the predict_sales_volume function
    prediction = predict_sales_volume(input_data)

    latency_ms = (time.time() - start_time) * 1000

    st.session_state["prediction"] = float(prediction)
    st.session_state["latency_ms"] = float(latency_ms)

  if st.session_state["prediction"] is not None:
    st.subheader("Prediction Result")
    st.metric(
        "Predicted Sales Volume", f"Units {st.session_state['prediction']:,.2f}"
    )
    st.metric(
        "Prediction Latency",
        f"{st.session_state['latency_ms']:.2f} ms"
    )

    st.subheader("Manual Actual Outcome and Business Feedback")

    actual_sales_volume = st.number_input("Enter actual sales volume", min_value = 0.0, value = 0.0, step = 1.0)

    feedback_score = st.slider("Business feedback score (1 = Poor, 5 = Excellent)", 1, 5, 4)

    feedback_text = st.text_area("Business feedback comment", placeholder = "Example: Prediction is useful for inventory planning.")

    if st.button("Submit Monitoring Log"):
      log_prediction(
          product_position = product_position_for_prediction,
          promotion = 'Yes' if promotion else 'No',
          product_category = product_category,
          seasonal = 'Yes' if seasonal else 'No',
          terms = terms,
          section = section,
          season = season,
          material = material,
          origin = origin,
          price = price,
          prediction = st.session_state["prediction"],
          actual_sales_volume = actual_sales_volume,
          latency_ms = st.session_state["latency_ms"],
          feedback_score = feedback_score,
          feedback_text = feedback_text
      )

      st.success("Monitoring log saved successfully. Open the Monitoring Dashboard tab to view results")
    else:
      st.info("Click Run Prediction to generate a prediction.")

with tab2:
  st.header("Model Monitoring Dashboard")

  logs = load_monitoring_logs()

  if logs.empty:
    st.warning("No monitoring logs found yet. Please use the Prediction tab and submit at least one monitoring log.")
  else:
    st.subheader("Key Monitoring Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Prediction Logged", len(logs))
    col2.metric("Average Feedback Score", f"{logs['feedback_score'].mean():.2f}")
    col3.metric("Average Latency", f"{logs['latency_ms'].mean():.2f} ms")
    col4.metric("Mean Absolute Error", f"{logs['absolute_error'].mean():.2f}")

    st.markdown("---")

    monitor_tab1, monitor_tab2, monitor_tab3, monitor_tab4 = st.tabs([
        "Model Performance",
        "Operational Health",
        "Business Feedback",
        "Input Drift"
    ])

    with monitor_tab1:
      st.subheader("Prediction Error Monitoring")

      mae = logs["absolute_error"].mean()
      rmse = np.sqrt(np.mean(logs["error"] ** 2))

      col_a, col_b = st.columns(2)
      col_a.metric("MAE", f"{mae:.2f}")
      col_b.metric("RMSE", f"{rmse:.2f}")

      error_chart = logs[["timestamp", "absolute_error"]].set_index("timestamp")
      st.line_chart(error_chart)

      st.write("A rising error trend may indicate model performance degradation.")

    with monitor_tab2:
      st.subheader("Latency Monitoring")

      latency_chart = logs[["timestamp", "latency_ms"]].set_index("timestamp")
      st.line_chart(latency_chart)

      st.write("Latency moinitoring helps determine whether the prediction service remains responsive.")

    with monitor_tab3:
      st.subheader("Manual Business Feedback Monitoring")

      feedback_chart = logs[["timestamp", "feedback_score"]].set_index("timestamp")
      st.line_chart(feedback_chart)

      st.subheader("Recent Business Feedback Comments")

      comments = logs[logs["feedback_text"].astype(str).str.strip() != ""]
      comments = comments.sort_values("timestamp", ascending = False).head(10)

      if comments.empty:
        st.info("No feedback comments available.")
      else:
        for _, row in comments.iterrows():
          st.write(f"**{row['timestamp']} | Score: {row['feedback_score']}**")
          st.write(row["feedback_text"])
          st.markdown("---")

    with monitor_tab4:
      st.subheader("Simple Input Drift Check")

      # Updated drift features for Zara product data
      drift_features = [
          "price"
      ]

      drift_rows = []

      for feature in drift_features:
        train_mean = training_data[feature].mean()
        prod_mean = production_data[feature].mean()
        difference = prod_mean - train_mean
        percent_change = (difference / train_mean) * 100 if train_mean != 0 else 0

        drift_rows.append({
            "Feature": feature,
            "Training Mean": train_mean,
            "Production Mean": prod_mean,
            "Difference": difference,
            "Percent Change": percent_change
        })

        drift_df = pd.DataFrame(drift_rows)

        st.dataframe(drift_df.style.format({
            "Training Mean": "{:.2f}",
            "Production Mean": "{:.2f}",
            "Difference": "{:.2f}",
            "Percent Change": "{:.2f}%"
        }))

        major_drift = drift_df[drift_df["Percent Change"].abs() > 20]

        if not major_drift.empty:
          st.error("Potential input drift detected in one or more features.")
        else:
          st.success("No major input drift detected using this simple mean comparison.")

        st.markdown("---")
        st.subheader("Raw Monitoring Logs")
        st.dataframe(logs)

with tab3:
  st.header("Agile Retrospective")

  st.subheader("Model Validation Result from Training Data")

  col1, col2 = st.columns(2)

  col1.metric("Validation MAE", f"{model_metrics['validation_mae']:.2f}")
  col2.metric("Validation RMSE", f"{model_metrics['validation_rmse']:.2f}")

  st.markdown("""
    Use the monitoring results to support the next sprint dicussion.

    ### What went well?
    - Was the model able to generate sales volume predictions?
    - Was the prediction latency acceptable?
    - Did business users provide useful feedback?

    ### What did not go well?
    - Were prediction errors high?
    - Was there evidence of input drift?
    - Did business users report low confidence in predictions?

    ### What should be improve next?
    - Should the model be retrained using newer sales data?
    - Should additional product features be added?
    - Should data validation be improved？
    - Should the feedback collection process be improved?

    ### Example Backlog Item
    As an inventory manager, I want the sales volume prediction model retrained using recent sales data so that sales forecasts remain reliable as product trends change.
  """)
