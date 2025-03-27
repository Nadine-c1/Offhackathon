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

types_df = pd.read_csv("Vliegtuigtypes_met_aangevuld_gewicht.csv")
data['type'] = data['type'].replace('Embraer ERJ190-100STD', 'Embraer ERJ 190-100 STD')

# Stap 1: Dataset voorbereiden
types_df = pd.read_csv("Vliegtuigtypes_met_aangevuld_gewicht.csv")
data['type'] = data['type'].replace('Embraer ERJ190-100STD', 'Embraer ERJ 190-100 STD')

# Bereken gemiddelde SEL_dB per type
avg_sel = data.groupby('type')['SEL_dB'].mean().reset_index(name='SEL_dB')
merged = avg_sel.merge(types_df[['type', 'gewicht_kg']], on='type', how='left').dropna()

# Gewichtsklassen
bins = [0, 50000, 100000, 150000, 200000, 300000, 400000, 600000]
labels = ['0–50k', '50–100k', '100–150k', '150–200k', '200–300k', '300–400k', '400k+']
merged['gewichtsklasse'] = pd.cut(merged['gewicht_kg'], bins=bins, labels=labels)

# ✅ Interactieve Boxplot met Plotly
st.title("📦 Spreiding van geluidsoverlast per gewichtsklasse")

fig_box = px.box(
    merged,
    x="gewichtsklasse",
    y="SEL_dB",
    points="outliers",  # of 'all' om alle punten te tonen
    labels={"SEL_dB": "Gemiddelde SEL_dB", "gewichtsklasse": "Gewichtsklasse (kg)"},
    title="Boxplot met interactieve hover"
)
st.plotly_chart(fig_box, use_container_width=True)

# Selecteer via radio
klasse_options = ['-- Geen selectie --'] + labels
selected_klasse = st.radio("🔍 Selecteer een gewichtsklasse:", options=klasse_options, horizontal=True)

# Alleen weergeven als het NIET 'geen' is
# --- Session state toggle-trick ---
# Init state
if 'selected_klasse' not in st.session_state:
    st.session_state.selected_klasse = None

st.markdown("### 🔘 Selecteer een gewichtsklasse (checkbox-stijl toggle)")

# UI
cols = st.columns(len(labels))
for i, label in enumerate(labels):
    with cols[i]:
        # checkbox = True als deze klasse actief is
        checked = st.checkbox(label, key=f"checkbox_{label}", value=(st.session_state.selected_klasse == label))

        # Als deze werd aangeklikt en nog niet actief was → selecteer
        if checked and st.session_state.selected_klasse != label:
            st.session_state.selected_klasse = label
            # Reset andere checkboxes
            for l in labels:
                if l != label:
                    st.session_state[f"checkbox_{l}"] = False

        # Als deze werd uitgezet → deselecteer
        if not checked and st.session_state.selected_klasse == label:
            st.session_state.selected_klasse = None

# Toon scatterplot als er iets is geselecteerd
selected_klasse = st.session_state.selected_klasse
if selected_klasse:
    klasse_data = merged[merged['gewichtsklasse'] == selected_klasse]

    st.markdown(f"### ✈️ Vliegtuigtypes in gewichtsklasse **{selected_klasse}**")

    fig_scatter = px.scatter(
        klasse_data,
        x="gewicht_kg",
        y="SEL_dB",
        color="type",
        hover_name="type",
        hover_data={"gewicht_kg": True, "SEL_dB": True},
        labels={"gewicht_kg": "Gewicht (kg)", "SEL_dB": "Gemiddelde SEL_dB"},
        title=f"Scatterplot: Gewicht vs Geluid voor {selected_klasse}",
    )
    fig_scatter.update_layout(showlegend=False)
    st.plotly_chart(fig_scatter, use_container_width=True)