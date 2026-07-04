import streamlit as st
import pandas as pd
import joblib

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

model = load_model()
data = load_data()

@st.cache_data
def load_zones():
    return pd.read_csv('zone_lookup.csv')

zones = load_zones()

st.title("Where should I drive next?")
st.caption("A demand-hotspot recommender for NYC taxi drivers, built on a GPU-accelerated pipeline.")

snapshots = (
    data[['pickup_date', 'pickup_hour']]
    .drop_duplicates()
    .sort_values(['pickup_date', 'pickup_hour'], ascending=False)
)
snapshots['label'] = snapshots['pickup_date'].dt.strftime('%b %d') + ' — ' + snapshots['pickup_hour'].astype(str) + ':00'

choice = st.selectbox("Choose a snapshot hour:", snapshots['label'])
chosen = snapshots[snapshots['label'] == choice].iloc[0]

snapshot_df = data[
    (data['pickup_date'] == chosen['pickup_date']) &
    (data['pickup_hour'] == chosen['pickup_hour'])
].copy()


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