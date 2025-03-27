import streamlit as st
st.set_page_config(layout="wide")  
import requests
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px
import numpy as np
from streamlit_folium import folium_static
from collections import Counter
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

start_date = int(pd.to_datetime('2024-01-01').timestamp())

end_date = int(pd.to_datetime('2025-01-01').timestamp())

response = requests.get(f'https://sensornet.nl/dataserver3/event/collection/nina_events/stream?conditions%5B0%5D%5B%5D=time&conditions%5B0%5D%5B%5D=%3E%3D&conditions%5B0%5D%5B%5D={start_date}&conditions%5B1%5D%5B%5D=time&conditions%5B1%5D%5B%5D=%3C&conditions%5B1%5D%5B%5D={end_date}&conditions%5B2%5D%5B%5D=label&conditions%5B2%5D%5B%5D=in&conditions%5B2%5D%5B2%5D%5B%5D=21&conditions%5B2%5D%5B2%5D%5B%5D=32&conditions%5B2%5D%5B2%5D%5B%5D=33&conditions%5B2%5D%5B2%5D%5B%5D=34&args%5B%5D=aalsmeer&args%5B%5D=schiphol&fields%5B%5D=time&fields%5B%5D=location_short&fields%5B%5D=location_long&fields%5B%5D=duration&fields%5B%5D=SEL&fields%5B%5D=SELd&fields%5B%5D=SELe&fields%5B%5D=SELn&fields%5B%5D=SELden&fields%5B%5D=SEL_dB&fields%5B%5D=lasmax_dB&fields%5B%5D=callsign&fields%5B%5D=type&fields%5B%5D=altitude&fields%5B%5D=distance&fields%5B%5D=winddirection&fields%5B%5D=windspeed&fields%5B%5D=label&fields%5B%5D=hex_s&fields%5B%5D=registration&fields%5B%5D=icao_type&fields%5B%5D=serial&fields%5B%5D=operator&fields%5B%5D=tags')

colnames = pd.DataFrame(response.json()['metadata'])

data = pd.DataFrame(response.json()['rows'])

data.columns = colnames.headers

data['time'] = pd.to_datetime(data['time'], unit = 's')
data['type'] = data['type'].replace('Embraer ERJ190-100STD', 'Embraer ERJ 190-100 STD')

# Load the temperature data, skipping metadata lines (usually around 50 lines)
temp_df = pd.read_csv("etmgeg_240.txt", skiprows=50)
temp_df['date'] = pd.to_datetime(temp_df['YYYYMMDD'], format='%Y%m%d')
# Strip whitespace van kolomnamen
temp_df.columns = temp_df.columns.str.strip()
# Optioneel: kolomnamen naar lowercase en underscores

data['date'] = pd.to_datetime(data['time']).dt.date
temp_df['date'] = temp_df['date'].dt.date
temp_df['TG_C'] = temp_df['TG'] / 10.0  # Daily mean temperature
data = data.merge(temp_df[['date', 'TG_C']], on='date', how='left')
############################################################################################
# Laad data
types_df = pd.read_csv("Vliegtuigtypes_met_aangevuld_gewicht.csv")
data['type'] = data['type'].replace('Embraer ERJ190-100STD', 'Embraer ERJ 190-100 STD')

# Bereken gemiddelde SEL_dB
avg_sel = data.groupby('type')['SEL_dB'].mean().reset_index(name='gemiddelde_SEL_dB')
merged = avg_sel.merge(types_df[['type', 'gewicht_kg']], on='type', how='left').dropna()

gewicht_range = st.slider("Filter op gewicht (kg)", 0, 600000, (0, 600000))
filtered = merged[(merged['gewicht_kg'] >= gewicht_range[0]) & (merged['gewicht_kg'] <= gewicht_range[1])]

# Plotly interactieve scatterplot
fig = px.scatter(
    merged,
    x="gewicht_kg",
    y="gemiddelde_SEL_dB",
    hover_name="type",
    labels={"gewicht_kg": "Gewicht (kg)", "gemiddelde_SEL_dB": "Gem. SEL_dB"},
    title="✈️ Geluidsoverlast vs Gewicht van vliegtuigtypes in 2024"
)

fig.update_traces(marker=dict(size=8, opacity=0.7))

# Streamlit output
st.plotly_chart(fig, use_container_width=True)

gewicht_range = st.slider(
    "Filter op gewicht (kg)", 
    0, 600000, (0, 600000), 
    key="gewicht_slider"
)
filtered = merged[(merged['gewicht_kg'] >= gewicht_range[0]) & (merged['gewicht_kg'] <= gewicht_range[1])]

# -- 1. Laad en bewerk data --
types_df = pd.read_csv("Vliegtuigtypes_met_aangevuld_gewicht.csv")
data['type'] = data['type'].replace('Embraer ERJ190-100STD', 'Embraer ERJ 190-100 STD')

# Gemiddelde SEL per type
avg_sel = data.groupby('type')['SEL_dB'].mean().reset_index(name='SEL_dB')

# Merge met gewicht
merged = avg_sel.merge(types_df[['type', 'gewicht_kg']], on='type', how='left').dropna()

# Gewichtsklassen
bins = [0, 50000, 100000, 150000, 200000, 300000, 400000, 600000]
labels = ['0–50k', '50–100k', '100–150k', '150–200k', '200–300k', '300–400k', '400k+']
merged['gewichtsklasse'] = pd.cut(merged['gewicht_kg'], bins=bins, labels=labels)

# -- 2. Boxplot tonen --
st.title("📦 Spreiding van geluidsoverlast (SEL_dB) per gewichtsklasse")

fig_box, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(data=merged, x='gewichtsklasse', y='SEL_dB', ax=ax)
ax.set_xlabel("Gewichtsklasse (kg)")
ax.set_ylabel("Gemiddelde SEL_dB")
st.pyplot(fig_box)

# -- 3. Interactieve selectie van gewichtsklasse --
selected_klasse = st.radio("🔍 Selecteer een gewichtsklasse voor detailweergave:",
                           options=labels,
                           index=None,
                           horizontal=True)

# -- 4. Als geselecteerd, toon scatterplot voor die klasse --
if selected_klasse:
    klasse_data = merged[merged['gewichtsklasse'] == selected_klasse]

    st.markdown(f"### ✈️ Vliegtuigtypes in gewichtsklasse **{selected_klasse}**")

    fig_scatter, ax = plt.subplots(figsize=(12, 5))
    sns.scatterplot(data=klasse_data, x='gewicht_kg', y='SEL_dB', hue='type', ax=ax)
    ax.set_xlabel("Gewicht (kg)")
    ax.set_ylabel("Gemiddelde SEL_dB")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    st.pyplot(fig_scatter)
############################################################################################
# Vliegtuigtypes
vliegtuigtypes = [
    "Boeing 737-800",
    "Embraer ERJ 170-200 STD",
    "Embraer ERJ 190-100 STD",
    "Airbus A320 214",
    "Boeing 737-700",
    "Boeing 777-300ER",
    "Airbus A319-111",
    "Boeing 737-900",
    "Boeing 777-200",
    "Embraer ERJ190-100LR"
]

# Titel en selectie
st.title("📊 Geluid en Temperatuur per Vliegtuigtype in 2024")
selected_type = st.selectbox("✈️ Kies een vliegtuigtype:", vliegtuigtypes)

# Veronderstel dat je 'data' al hebt geladen met de juiste kolommen
data['date'] = pd.to_datetime(data['time']).dt.date
filtered = data[data['type'] == selected_type].copy()

# Aggregatie per dag
daggemiddelden = filtered.groupby('date').agg(
    gemiddelde_temp=('TG_C', 'mean'),
    gemiddelde_geluid=('SEL_dB', 'mean'),
    aantal_vluchten=('type', 'count')
).reset_index()

# Toon aantal vluchten
st.markdown(f"✈️ **Aantal vluchten van {selected_type} in 2024**: `{len(filtered)}`")

# Plotly dubbele y-as grafiek
fig = go.Figure()

# Temperatuur lijn
fig.add_trace(go.Scatter(
    x=daggemiddelden['date'],
    y=daggemiddelden['gemiddelde_temp'],
    mode='lines+markers',
    name='Temperatuur (°C)',
    line=dict(color='blue'),
    yaxis='y1',
    hovertemplate='%{y:.1f} °C dB<br>Vluchten: %{customdata}x',
    customdata=daggemiddelden[['aantal_vluchten']]
))

# Geluid lijn
fig.add_trace(go.Scatter(
    x=daggemiddelden['date'],
    y=daggemiddelden['gemiddelde_geluid'],
    mode='lines+markers',
    name='SEL_dB',
    line=dict(color='red'),
    yaxis='y2',
    hovertemplate='%{y:.1f} dB <br> Vluchten: %{customdata}x',
    customdata=daggemiddelden[['aantal_vluchten']]
))


# Layout met correcte attributen
fig.update_layout(
    title=f"{selected_type} – Gemiddelde dagwaarden in 2024",
    xaxis_title="Datum",
    yaxis=dict(
        title="Temperatuur (°C)",
        tickfont=dict(color="blue"),
    ),
    yaxis2=dict(
        title="SEL_dB",
        overlaying='y',
        side='right',
        tickfont=dict(color="red"),
    ),
    legend=dict(x=0.01, y=0.99),
    hovermode='x unified',
    height=600
)

# Toon in Streamlit
st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def load_data():
    # Load or process your data here
    return data

df = load_data()

# Group and aggregate
grouped = (
    df.groupby('type')
    .agg(
        lasmax_dB=('lasmax_dB', 'mean'),
        SEL_dB=('SEL_dB', 'mean'),
        avg_duration=('duration', 'mean'),
        count=('type', 'count')
    )
    .reset_index()
)

# Title
st.title("Top 10 Aircraft Types by Noise and Duration Metrics")

# Sidebar selection
sort_column = st.selectbox(
    "Sort by column:",
    ['lasmax_dB', 'SEL_dB', 'avg_duration', 'count'],
    index=1
)

sort_order = st.radio("Sort order:", ["Descending", "Ascending"]) == "Descending"

# Sort and display
top_10 = grouped.sort_values(by=sort_column, ascending=not sort_order).head(10)
st.dataframe(top_10, use_container_width=True)

############################################################################################
@st.cache_data
def load_data():
    # Your data loading logic here
    return data
# --- Define filter lists ---
allowed_types = [
    "Boeing 737-800",
    "Embraer ERJ 170-200 STD",
    "Embraer ERJ 190-100 STD",
    "Embraer ERJ190-100STD",
    "Airbus A320 214",
    "Boeing 737-700",
    "Boeing 777-300ER",
    "Airbus A319-111",
    "Boeing 737-900",
    "Boeing 777-200"
]

allowed_tags = [
    "Zwanenburgerbaan18C_T",
    "Zwanenburgerbaan36C_L",
    "Aalsmeerbaan18L_T",
    "Aalsmeerbaan36R_L",
    "Oostbaan22_T",
    "Oostbaan04_L",
    "Kaagbaan24_T",
    "Kaagbaan06_L"
]

# --- Selectable axes and grouping ---
numeric_colsx=  ['lasmax_dB','SEL_dB']
numeric_colsy = ['distance', 'altitude', 
                'windspeed', 'winddirection', 'duration','TG_C']
grouping_cols = ['type', 'tags']

# --- App title ---
st.title("Filtered Scatterplot of Aircraft Noise & Flight Data")

# --- Axes selection ---
x_axis = st.selectbox("Select X-axis", numeric_colsx, index=0)
y_axis = st.selectbox("Select Y-axis", numeric_colsy, index=1)
group_col = st.selectbox("Color by:", grouping_cols)

# --- Filter dataset ---
filtered_data = data[
    data['type'].isin(allowed_types) & 
    data['tags'].isin(allowed_tags)
]

# --- Drop NA for selected columns ---
plot_data = filtered_data[[x_axis, y_axis, group_col]].dropna()

# --- Create scatterplot ---
fig = px.scatter(
    plot_data,
    x=x_axis,
    y=y_axis,
    color=group_col,
    title=f"{y_axis} vs {x_axis} by {group_col}",
    opacity=0.7,
    height=600
)

# --- Display plot ---
st.plotly_chart(fig, use_container_width=True)

#################################################################################################
# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("nl.csv")
    return df.dropna(subset=['lat', 'lng', 'population'])

df = load_data()

# --- Create Folium Map ---
m = folium.Map(location=[52.2606, 4.7594], zoom_start=11)

# --- Add Heatmap ---
heat_data = [[row['lat'], row['lng'], row['population']] for index, row in df.iterrows()]
HeatMap(heat_data, radius=60, blur=15, max_zoom=10).add_to(m)

# --- Add Schiphol marker ---
folium.Marker(
    [52.3105, 4.7683],
    popup='Schiphol',
    icon=folium.Icon(icon='plane', prefix='fa', color='orange')
).add_to(m)

# --- Add Measurement Stations ---
stations = {
    'Aalsmeerderweg': [52.271712, 4.772485],
    'Darwinstraat': [52.234322, 4.759092],
    'Uiterweg': [52.263306, 4.732377],
    'Copierstraat': [52.228925, 4.739004],
    'Kudelstaartseweg': [52.234885, 4.748050],
    'Blaauwstraat': [52.264531, 4.774595],
    'Hornweg': [52.268712, 4.787693]
}

for station, coords in stations.items():
    folium.Marker(coords, popup=station, icon=folium.Icon(icon='m', color='red')).add_to(m)
    folium.Circle(
        location=coords,
        radius=4000,
        color='green',
    ).add_to(m)

# --- Add Runways ---
runways = [
    ([(52.362423, 4.711867), (52.328537, 4.708903)], 'Polderbaan'),  
    ([(52.316809, 4.745912), (52.318347, 4.797313)], 'Buitenveldertbaan'),  
    ([(52.321423, 4.780239), (52.291093, 4.777364)], 'Aalsmeerbaan'),  
    ([(52.300436, 4.783654), (52.313843, 4.802885)], 'Oostbaan'),    
    ([(52.302213, 4.737438), (52.331181, 4.740188)], 'Zwanenburgbaan'),  
    ([(52.288182, 4.734371), (52.304496, 4.777672)], 'Kaagbaan'),
]

for runway_coords, runway_name in runways:
    folium.PolyLine(runway_coords, color='yellow', weight=5).add_to(m)
    midpoint = [
        (runway_coords[0][0] + runway_coords[1][0]) / 2,
        (runway_coords[0][1] + runway_coords[1][1]) / 2
    ]
    folium.Marker(midpoint, popup=runway_name, icon=folium.Icon(icon='plane', prefix='fa', color='red')).add_to(m)

# --- Streamlit Page ---
st.title("Aalsmeer Region: Population Density & Schiphol Map")

# --- Display Folium Map in Streamlit ---
st_data = st_folium(m, width=1000, height=700)


#################################################################################
