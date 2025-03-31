import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime

# Load the dataset
@st.cache_data  # Cache data to improve performance
def load_data(file_path):
    df = pd.read_csv(file_path, na_filter=False)
    return df

# Preprocessing function (handle missing values and categorical features)
@st.cache_data
def preprocess_data(df):
    # Replace 'NA' strings with np.nan
    df = df.replace('NA', np.nan)

    # Convert Date to datetime and extract year, month, day
    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day

    # Handle missing values using median for numerical columns
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())

    # Convert RainToday and RainTomorrow to numerical (0 and 1)
    df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0}).fillna(0)
    df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0}).fillna(0)

    # Encode categorical features
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df

# Train the ANN model
@st.cache_data
def train_model(df):
    # Check if the DataFrame is empty
    if df.empty:
        st.warning("DataFrame is empty. Cannot train the model.")
        return None, None, None, None

    X = df.drop(['RainTomorrow', 'Date'], axis=1)
    y = df['RainTomorrow']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = MLPClassifier(random_state=42, max_iter=300)
    model.fit(X_train, y_train)
    return model, scaler, X_test, y_test

# location_coords dictionary (MAKE SURE IT'S COMPLETE)
location_coords = {
    'Albury': (-36.0737, 146.9135),
    'BadgerysCreek': (-33.8800, 150.7440),
    'Cobar': (-31.4941, 145.8353),
    'CoffsHarbour': (-30.2964, 153.1142),
    'Moree': (-29.4635, 149.8449),
    'NorfolkIsland': (-29.0341, 167.9547),
    'Penrith': (-33.7557, 150.6723),
    'Richmond': (-33.6007, 150.7497),
    'Sydney': (-33.8688, 151.2093),
    'SydneyAirport': (-33.9461, 151.1772),
    'SydneyOlympicPark': (-33.8447, 151.0694),
    'Williamtown': (-32.8150, 151.8428),
    'Wollongong': (-34.4278, 150.8931),
    'Canberra': (-35.2809, 149.1300),
    'Tuggeranong': (-35.4333, 149.0667),
    'MountGinini': (-35.5217, 148.7758),
    'Ballarat': (-37.5622, 143.8503),
    'Bendigo': (-36.7578, 144.2809),
    'Sale': (-38.1067, 147.0656),
    'MelbourneAirport': (-37.6733, 144.8433),
    'Melbourne': (-37.8136, 144.9631),
    'MelbourneCBD': (-37.8178, 144.9659),
    'Mildura': (-34.1872, 142.1578),
    'Nhil': (-36.6487, 141.6511),
    'Portland': (-38.3433, 141.6033),
    'Watsonia': (-37.7167, 145.0833),
    'Dartmoor': (-37.8333, 141.2333),
    'Brisbane': (-27.4698, 153.0251),
    'Cairns': (-16.9186, 145.7781),
    'GoldCoast': (-28.0167, 153.4000),
    'Townsville': (-19.2589, 146.8169),
    'Adelaide': (-34.9285, 138.6007),
    'MountGambier': (-37.8274, 140.7817),
    'Nuriootpa': (-34.4703, 138.9919),
    'Woomera': (-31.1667, 136.8167),
    'Albany': (-35.0275, 117.8836),
    'Witchcliffe': (-34.1167, 115.0500),
    'PearceRAAF': (-31.6667, 116.0333),
    'PerthAirport': (-31.9403, 115.9669),
    'Perth': (-31.9505, 115.8605),
    'SalmonGums': (-32.9833, 121.7833),
    'Walpole': (-34.9778, 116.7333),
    'Hobart': (-42.8821, 147.3272),
    'Launceston': (-41.4419, 147.1450),
    'AliceSprings': (-23.6980, 133.8807),
    'Darwin': (-12.4634, 130.8456),
    'Katherine': (-14.4667, 132.2667),
    'Uluru': (-25.3444, 131.0369)
}

# Correct way to create map_data
map_data = pd.DataFrame(list(location_coords.items()), columns=['Location', 'Coordinates'])
map_data[['Latitude', 'Longitude']] = pd.DataFrame(map_data['Coordinates'].tolist(), index=map_data.index)
map_data.drop('Coordinates', axis=1, inplace=True)

# Load data
file_path = "weatherAUS.csv"  # Use the local file path
df = load_data(file_path)

# Preprocess data (this will create 'year', 'month', 'day')
df = preprocess_data(df.copy())

# Streamlit app
st.title("Rainfall Prediction Dashboard")

# Sidebar filters
st.sidebar.header("Filters")

# Location filter
locations = df['Location'].unique()
selected_locations = st.sidebar.multiselect("Select Locations", locations, default=locations)
filtered_df = df[df['Location'].isin(selected_locations)]

# Date range filter
min_date = filtered_df['Date'].min().date()
max_date = filtered_df['Date'].max().date()
start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", max_date)

# Convert selected dates to datetime objects
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Filter by date
filtered_df = filtered_df[(filtered_df['Date'] >= start_date) & (filtered_df['Date'] <= end_date)]

# Numerical feature filters
st.sidebar.header("Numerical Filters")
min_temp_threshold = st.sidebar.slider("Minimum Temperature Threshold", float(df['MinTemp'].min()), float(df['MinTemp'].max()), float(df['MinTemp'].min()))
max_temp_threshold = st.sidebar.slider("Maximum Temperature Threshold", float(df['MaxTemp'].min()), float(df['MaxTemp'].max()), float(df['MaxTemp'].max()))
filtered_df = filtered_df[(filtered_df['MinTemp'] >= min_temp_threshold) & (filtered_df['MaxTemp'] <= max_temp_threshold)]

# Display filtered data
st.subheader("Filtered Data")
st.dataframe(filtered_df)

# Visualizations
st.subheader("Visualizations")

# Rainfall distribution
st.subheader("Rainfall Distribution")
fig_rainfall = px.histogram(filtered_df, x="Rainfall", title="Rainfall Distribution")
st.plotly_chart(fig_rainfall)

# Average rainfall by location
avg_rainfall = filtered_df.groupby('Location')['Rainfall'].mean().reset_index()
fig_avg_rainfall = px.bar(avg_rainfall, x='Location', y='Rainfall', title='Average Rainfall by Location')
st.plotly_chart(fig_avg_rainfall)

# Australia Map
st.subheader("Rainfall Map of Australia")

# Clean column names by stripping whitespace
map_data.columns = map_data.columns.str.strip()
avg_rainfall.columns = avg_rainfall.columns.str.strip()

# Ensure 'Location' columns are of the same type (e.g., string)
map_data['Location'] = map_data['Location'].astype(str)
avg_rainfall['Location'] = avg_rainfall['Location'].astype(str)

# Check for missing values in 'Location' columns
if map_data['Location'].isnull().sum() > 0:
    map_data = map_data.dropna(subset=['Location'])
if avg_rainfall['Location'].isnull().sum() > 0:
    avg_rainfall = avg_rainfall.dropna(subset=['Location'])

# Perform the merge after the above checks
map_data = pd.merge(map_data, avg_rainfall, on='Location', how='left')

# Merge with average rainfall data
map_data = pd.merge(map_data, avg_rainfall, on='Location', how='left')
map_data['Rainfall'].fillna(0, inplace=True)  # Fill NaN rainfall values with 0

# Create the map
fig_map = go.Figure(go.Scattergeo(
    lon=map_data['Longitude'],
    lat=map_data['Latitude'],
    text=map_data['Location'] + ': ' + map_data['Rainfall'].astype(str) + ' mm',
    mode='markers',
    marker=dict(
        size=map_data['Rainfall'] * 5,  # Adjust size multiplier as needed
        color=map_data['Rainfall'],
        colorscale='Viridis',
        colorbar_title="Average Rainfall (mm)"
    )
))

fig_map.update_layout(
    title_text='Average Rainfall by Location',
    geo=dict(
        scope='australia',
        landcolor='lightgreen',
        showocean=True,
        oceancolor="lightblue",
        projection_type='natural earth'
    )
)

st.plotly_chart(fig_map)

# Train model
model, scaler, X_test, y_test = train_model(filtered_df)

# Prediction section
st.subheader("Rainfall Prediction")

# Prepare data for prediction
X = filtered_df.drop(['RainTomorrow', 'Date'], axis=1)
X_scaled = scaler.transform(X)

# Make predictions
y_pred = model.predict(X_scaled)

# Display predictions
st.write("Predictions:")
st.write(y_pred)

# Model Evaluation
st.subheader("Model Evaluation")
y_pred_test = model.predict(scaler.transform(X_test))
accuracy = accuracy_score(y_test, y_pred_test)
st.write(f"Accuracy on Test Set: {accuracy:.2f}")

# Debugging step: Print available columns before filling NaN values
print("Available columns in map_data:", map_data.columns)

# Check if 'Rainfall' column exists before calling fillna()
if 'Rainfall' in map_data.columns:
    map_data['Rainfall'].fillna(0, inplace=True)
else:
    print("'Rainfall' column not found in map_data.")
    # You can log this or handle it as needed

