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
    "Product Position": prediction_product_position,
    "Promotion": prediction_promotion,
    "Product Category": prediction_product_category,
    "Seasonal": prediction_seasonal,
    "terms": prediction_terms,
    "section": prediction_section,
    "season": prediction_season,
    "material": prediction_material,
    "origin": prediction_origin,
    "price": prediction_price
}


st.header("Data Overview")
if filtered_data.empty:
    st.warning("No data matches the selected filters.")
else:
    st.write(f"Displaying {len(filtered_data)} records out of {len(production_data)} total records.")
    st.dataframe(filtered_data.head())

    # Visualization 1: Sales Volume Distribution
    st.subheader("Sales Volume Distribution")
    fig_hist = px.histogram(filtered_data, x='Sales Volume', nbins=50,
                            title='Distribution of Sales Volume for Filtered Data')
    st.plotly_chart(fig_hist)

    # Visualization 2: Sales by Product Category
    st.subheader("Sales Volume by Product Category")
    sales_by_category = filtered_data.groupby('Product Category')['Sales Volume'].sum().reset_index()
    fig_cat = px.bar(sales_by_category, x='Product Category', y='Sales Volume',
                     title='Total Sales Volume by Product Category')
    st.plotly_chart(fig_cat)

    # Visualization 3: Sales by Material
    st.subheader("Sales Volume by Material")
    sales_by_material = filtered_data.groupby('material')['Sales Volume'].sum().reset_index().sort_values(by='Sales Volume', ascending=False)
    fig_material = px.bar(sales_by_material.head(10), x='material', y='Sales Volume',
                          title='Top 10 Materials by Sales Volume')
    st.plotly_chart(fig_material)

    # Analytical Output 1: Average Sales Volume for Filtered Data
    st.subheader("Analytical Output")
    avg_sales_volume = filtered_data['Sales Volume'].mean()
    st.metric("Average Sales Volume (Filtered Data)", f"{avg_sales_volume:,.2f} Units")

    # Analytical Output 2: Prediction for a single item
    st.subheader("Predicted Sales Volume for Selected Item")
    if st.sidebar.button("Get Single Item Prediction"):
        try:
            predicted_volume = predict_sales_volume(prediction_input_data)
            st.metric("Predicted Sales Volume", f"{predicted_volume:,.2f} Units")
        except Exception as e:
            st.error(f"Error making prediction: {e}. Please ensure all required features are selected and model parameters are loaded correctly.")
