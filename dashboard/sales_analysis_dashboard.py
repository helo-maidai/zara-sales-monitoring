import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

from predict_utils import load_model_params, predict_sales_volume

PRODUCTION_DATA_PATH = BASE_DIR / "data" / "zara_production.csv"

st.set_page_config(
    page_title = "Sales Analysis and Prediction Dashboard",
    layout = "wide"
)

st.title("Zara Sales Analysis and Prediction Dashboard")

st.write("Explore sales data, filter by various product attributes, and get a predicted sales volume.")

@st.cache_data
def load_production_data():
  return pd.read_csv(PRODUCTION_DATA_PATH)

@st.cache_data
def load_model_and_params():
    xgb_model, scaler, model_params = load_model_params()
    return xgb_model, scaler, model_params

production_data = load_production_data()
model, scaler, model_params = load_model_and_params()

st.sidebar.header("Filters & Prediction Input")

# Interactive Feature 1: Product Category filter
selected_categories = st.sidebar.multiselect(
    "Select Product Categories",
    options=production_data['Product Category'].unique(),
    default=production_data['Product Category'].unique()
)

# Interactive Feature 2: Price range slider
min_price, max_price = st.sidebar.slider(
    "Price Range",
    min_value=float(production_data['price'].min()),
    max_value=float(production_data['price'].max()),
    value=(float(production_data['price'].min()), float(production_data['price'].max()))
)

# Filter data based on selections
filtered_data = production_data[
    (production_data['Product Category'].isin(selected_categories)) &
    (production_data['price'] >= min_price) &
    (production_data['price'] <= max_price)
]

# Dynamic Prediction Input
st.sidebar.subheader("Prediction Input (for a single item)")
prediction_product_position = st.sidebar.selectbox("Position", ('Aisle', 'End-cap', 'Front of Store'))
prediction_promotion = st.sidebar.selectbox("Promotion", ('No', 'Yes'))
prediction_product_category = st.sidebar.selectbox("Category", production_data['Product Category'].unique())
prediction_seasonal = st.sidebar.selectbox("Seasonal", ('No', 'Yes'))
prediction_terms = st.sidebar.selectbox("Terms", production_data['terms'].unique())
prediction_section = st.sidebar.selectbox("Section", production_data['section'].unique())
prediction_season = st.sidebar.selectbox("Season", production_data['season'].unique())
prediction_material = st.sidebar.selectbox("Material", production_data['material'].unique())
prediction_origin = st.sidebar.selectbox("Origin", production_data['origin'].unique())
prediction_price = st.sidebar.number_input("Input Price", min_value = 0.0, value = 50.0, step = 1.0)

prediction_input_data = {
    
