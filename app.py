import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
import os
import h3

# --- Layout and Configuration ---
st.set_page_config(page_title="Early Outbreak Detection Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- Styling ---
st.markdown("""
    <style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .metric-value {
        font-size: 36px;
        color: #FF4B4B;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data():
    file_path = os.path.join("notebooks", "ml_pipeline_test_results.csv")
    if not os.path.exists(file_path):
        st.error(f"Data file not found at {file_path}")
        return pd.DataFrame()
    df = pd.read_csv(file_path)
    df['date_bucket'] = pd.to_datetime(df['date_bucket']).dt.date
    return df

with st.spinner("Loading outbreak predictions..."):
    df = load_data()

if df.empty:
    st.stop()

# --- Sidebar Controls ---
st.sidebar.title("Timeline Controls")
st.sidebar.markdown("Drag the slider to visualize the outbreak evolution over time.")

dates = sorted(df['date_bucket'].unique())
if not dates:
    st.error("No dates found in data.")
    st.stop()

selected_date = st.sidebar.select_slider("Select Date", options=dates, value=dates[-1])

# --- Filtered Data ---
filtered_df = df[df['date_bucket'] == selected_date].copy()

# --- Main Dashboard ---
st.title("🦠 Spatio-Temporal Outbreak Detection System")
st.markdown(f"**Current Time Slice:** `{selected_date}`")

# --- Metrics ---
col1, col2, col3, col4 = st.columns(4)

total_hexes = len(filtered_df)
active_anomalies = filtered_df['is_anomaly'].sum()
total_expected_cases = int(filtered_df['xgb_predicted_count'].sum())
avg_anomaly_score = filtered_df['anomaly_score'].mean()

col1.metric("Active Spatial Anomalies", f"{active_anomalies:,}")
col2.metric("Predicted Next-Day Cases", f"{total_expected_cases:,}")
col3.metric("Monitored Regions", f"{total_hexes:,}")
col4.metric("Avg Anomaly Score", f"{avg_anomaly_score:.3f}")

st.markdown("---")

# --- Map ---
st.subheader("Geospatial Anomaly View")

# Prepare data for pydeck
def get_color(is_anomaly):
    if is_anomaly:
        return [255, 75, 75, 200]  # Glowing Red/Yellow for anomaly
    return [40, 40, 200, 80]       # Cool Blue for safe zones

filtered_df['color'] = filtered_df['is_anomaly'].apply(get_color)

# Pydeck H3HexagonLayer
layer = pdk.Layer(
    "H3HexagonLayer",
    filtered_df.drop(columns=['date_bucket']),
    pickable=True,
    stroked=True,
    filled=True,
    extruded=False,
    get_hexagon="h3_index",
    get_fill_color="color",
    get_line_color=[255, 255, 255, 50],
    line_width_min_pixels=1,
)

# Set map center to show the whole map
lat, lon = 39.8283, -98.5795 # Center of continental US

view_state = pdk.ViewState(
    latitude=lat,
    longitude=lon,
    zoom=3.0,
    pitch=0
)

# Render DeckGL map
r = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={
        "html": "<b>Hex ID:</b> {h3_index}<br/>"
                "<b>Current Cases:</b> {event_count}<br/>"
                "<b>Predicted Cases:</b> {xgb_predicted_count}<br/>"
                "<b>Anomaly Score:</b> {anomaly_score}<br/>"
                "<b>Is Anomaly:</b> {is_anomaly}",
        "style": {"color": "white"}
    }
)

st.pydeck_chart(r)

st.markdown("---")

# --- Line Chart ---
st.subheader("Temporal Trends (XGBoost Predictions vs Reality)")

# Group by date to get daily totals
daily_stats = df.groupby('date_bucket').agg(
    actual_cases=('event_count', 'sum'),
    predicted_cases=('xgb_predicted_count', 'sum')
).reset_index()

fig = px.line(daily_stats, x='date_bucket', y=['actual_cases', 'predicted_cases'],
              labels={'value': 'Total Cases', 'date_bucket': 'Date', 'variable': 'Metric'},
              color_discrete_sequence=['#4B4BFF', '#FF4B4B'])

fig.update_layout(legend_title_text='Event Type')

# Add a vertical line for the selected date on the slider
fig.add_vline(x=selected_date, line_width=2, line_dash="dash", line_color="white")

st.plotly_chart(fig, use_container_width=True)
