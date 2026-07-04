import streamlit as st
import pandas as pd
import joblib
import datetime

st.set_page_config(page_title="Taxi Demand Hotspots", layout="centered")

FEATURES = ['pickup_location_id', 'pickup_hour', 'day_of_week', 'lag_1', 'lag_24', 'rolling_mean_3']

@st.cache_resource
def load_model():
    return joblib.load('demand_model.pkl')

@st.cache_data
def load_data():
    df = pd.read_parquet('hourly_features.parquet')
    df['pickup_date'] = pd.to_datetime(df['pickup_date'])
    return df

@st.cache_data
def load_zones():
    return pd.read_csv('zone_lookup.csv')

model = load_model()
data = load_data()
zones = load_zones()

st.title("Where should I drive next?")
st.caption("A demand-hotspot recommender for NYC taxi drivers, built on a GPU-accelerated pipeline.")

min_date = data['pickup_date'].min().date()
max_date = data['pickup_date'].max().date()

today = datetime.date.today()
mapped_day = min(today.day, max_date.day)
default_date = datetime.date(min_date.year, min_date.month, mapped_day)
default_date = min(max(default_date, min_date), max_date)

default_hour = datetime.datetime.now().hour

st.subheader("Pick a snapshot")
st.caption(f"Live data only covers {min_date.strftime('%B %Y')} — defaulting to today's date and hour, mapped onto that month.")

col1, col2 = st.columns([1, 1.4])
with col1:
    chosen_date = st.date_input("Date", value=default_date, min_value=min_date, max_value=max_date)
with col2:
    chosen_hour = st.select_slider(
        "Hour",
        options=list(range(24)),
        value=default_hour,
        format_func=lambda h: datetime.time(h).strftime("%I %p").lstrip("0"),
    )

snapshot_df = data[
    (data['pickup_date'] == pd.Timestamp(chosen_date)) &
    (data['pickup_hour'] == chosen_hour)
].copy()

if snapshot_df.empty:
    st.warning("No recorded trips for that exact date and hour — try a nearby hour.")
    st.stop()

snapshot_df['predicted_demand'] = model.predict(snapshot_df[FEATURES])

ranked = (
    snapshot_df[['pickup_location_id', 'predicted_demand']]
    .sort_values('predicted_demand', ascending=False)
    .head(10)
)

ranked = ranked.merge(zones, left_on='pickup_location_id', right_on='zone_id', how='left')
ranked['display_name'] = ranked['zone_name'].fillna('Zone ' + ranked['pickup_location_id'].astype(str))

st.subheader("Top 10 zones to head to")
st.bar_chart(ranked.set_index('display_name')['predicted_demand'])
st.dataframe(ranked[['display_name', 'borough', 'predicted_demand']], use_container_width=True)

st.subheader("Why acceleration matters")
st.caption("Same aggregation, CPU vs GPU — 6.7M trip records")
benchmark = pd.DataFrame({
    'Backend': ['CPU (pandas)', 'GPU (cuDF, warm)'],
    'Seconds': [0.767, 0.129]
}).set_index('Backend')
st.bar_chart(benchmark)
st.metric("Speedup", "5.9x faster")