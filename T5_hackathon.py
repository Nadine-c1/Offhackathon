## imports
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import numpy as np
from collections import Counter
import plotly.express as px
from folium import plugins
from folium.plugins import HeatMap
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go

## set to wide mode
st.set_page_config(layout="wide")  

#---------------------------------------#
## data cache
@st.cache_data
def load_data_een_jaar():
    df_plane = pd.read_csv('Data1jaar.csv')
    df_plane['time'] = pd.to_datetime(df_plane['time'])
    df_plane['landen_stijgen'] = np.where((df_plane['tags'].str.split('_').str[1])=='T','stijgt','land')
    df_plane['datum'] = df_plane['time'].dt.date
    df_plane['type'] = df_plane['type'].replace('Embraer ERJ190-100STD', 'Embraer ERJ 190-100 STD')

    # locaties groeperen
    location_map = {
        'Aa' : 'Aalsmeer',
        'Bl' : 'Aalsmeer',
        'Ho' : 'Aalsmeer',
        'Da' : 'Kuddelstaart',
        'Co' : 'Kuddelstaart',
        'Ku' : 'Kuddelstaart',
        'Ui' : 'Water'
    }
    df_plane['distance ground'] = np.sqrt(df_plane['distance']**2 - df_plane['altitude']**2)
    df_plane['locatie_Cat'] = df_plane['location_short'].map(location_map)
    df_plane['gem_geluid_loc']= df_plane.groupby(['callsign','locatie_Cat','datum'])['SEL_dB'].transform('mean')
    df_plane['windrichting'] = pd.cut(df_plane['winddirection'], 
                                    bins=[0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5,360], 
                                    labels=['N', 'NO', 'O', 'ZO', 'Z', 'ZW', 'W', 'NW','N'], 
                                    right=False,ordered=False)
    df_plane.drop(columns=['id','location_long','SELd','SELe','SELn','SELden','SEL','label',
                           'windspeed','winddirection','callsign','duration','hex_s', 
                           'registration', 'icao_type', 'serial'],inplace=True)
    return df_plane

@st.cache_data
def load_data_deze_week():
    df_plane = pd.read_csv('Data24h.csv') 
    df_plane['time'] = pd.to_datetime(df_plane['time'])
    df_plane = df_plane[df_plane['time'].dt.date==pd.to_datetime('2025-03-24').date()]
    df_plane['Datum'] = pd.to_datetime(df_plane['time'].dt.date)
    df_plane['type'] = df_plane['type'].str.replace(" ","",regex=False)
    df_extra = pd.DataFrame([
        ['Aalsmeerderweg', 52.271712, 4.772485],
        ['Darwinstraat', 52.234322, 4.759092],
        ['Uiterweg', 52.263306, 4.732377],
        ['Copierstraat', 52.228925, 4.739004],
        ['Kudelstaartseweg', 52.234885, 4.748050],
        ['Blaauwstraat', 52.264531, 4.774595],
        ['Hornweg', 52.268712, 4.787693]
        ], columns=['locatie', 'lat', 'lon'])
    df_plane = df_plane.merge(df_extra, left_on='location_long', right_on='locatie', how='left')
    df_plane['landen_stijgen'] = np.where((df_plane['tags'].str.split('_').str[1])=='T','stijgt','land')
    df_plane.drop(columns=['id','location_short','location_long','SELd','SELe','SELn','SELden','SEL',
                           'distance','altitude', 'label','windspeed','winddirection','duration',
                           'hex_s', 'registration', 'icao_type', 'serial', 'operator'],inplace=True)
    return df_plane

@st.cache_data
def load_vluchten_data():
    df_vluchten = pd.read_csv('flights_today_master1.csv')
    df_vluchten['Datum'] = pd.to_datetime(df_vluchten['Time (EDT)'], format='%a %I:%M:%S %p')
    df_vluchten['volledige_tijd'] = '2025-03-24 ' + df_vluchten['Time (EDT)'].str.split(' ').str[1] + ' ' + df_vluchten['Time (EDT)'].str.split(' ').str[2]
    df_vluchten['time'] = pd.to_datetime(df_vluchten['volledige_tijd'], format='%Y-%m-%d %I:%M:%S %p') + timedelta(hours=4)
    df_vluchten.drop(columns=['Course','Speed_kts','Speed_mph','Altitude_feet','ClimbRate',
                              'ReportingFacility','ScrapeTime','Time (EDT)'], inplace=True)
    return df_vluchten

@st.cache_data
def load_temperature_data():
    temp_df = pd.read_csv("etmgeg_240.txt", skiprows=50)
    temp_df['date'] = pd.to_datetime(temp_df['YYYYMMDD'], format='%Y%m%d')
    temp_df.columns = temp_df.columns.str.strip()
    temp_df['date'] = temp_df['date'].dt.date
    temp_df['TG_C'] = temp_df['TG'] / 10.0 
    return temp_df[['date', 'TG_C']].dropna()

@st.cache_data
def load_kg_cap_data():
    types_df = pd.read_csv("Vliegtuigtypes_met_aangevuld_gewicht.csv")
    return types_df
#---------------------------------------#
## data inladen
vluchten_data = load_vluchten_data()
meetstation_data = load_data_deze_week()
data_1jaar = load_data_een_jaar()
temperature_data = load_temperature_data()
types_df=load_kg_cap_data()

data_1jaar_boeing737800 = data_1jaar[data_1jaar['type'] == 'Boeing 737-800']
data_1jaar = data_1jaar.merge(
    temperature_data,
    how='left',
    left_on='datum',
    right_on='date'
)

# Sidebar voor paginaselectie
pagina = st.sidebar.radio(
    "Selecteer een pagina:",
    ('Introductie', 'Data verkenning', 'Analyse meetstations en vliegroutes', 'Analyse geluid', 'Conclusie')
)
##########################################################################################################################################################################################
if pagina == 'Introductie':
    st.title("✈️ Analyse van Geluidsmetingen & Vluchtdata rond Schiphol")
    st.subheader("Team 5 - Nadine, Pol, Quinn & Sanne")
    st.markdown("""
    Welkom bij deze interactieve applicatie waarmee we vluchtgegevens, geluidsmetingen en weersinformatie combineren om inzicht te krijgen in de invloed van vliegverkeer op geluidsoverlast in regio Aalsmeer en omstreken.

    ### 📁 Gebruikte datasets
    - **Flight data** (live & historisch)
    - **Meetstationdata** (24 uur geluidsmetingen op locaties rond Aalsmeer)
    - **Temperatuurdata** (KNMI – dagelijkse temperatuur in 2024)
    - **Vliegtuigtypes** met bijbehorend **gewicht en passagierscapaciteit**

    ### 🧭 Navigatie
    Gebruik de zijbalk om door de volgende onderdelen te bladeren:
    - **Data verkenning**: ontdek de meest voorkomende vliegtuigen en hun geluidsprofielen
    - **Analyse meetstations en vliegroutes**: bekijk kaarten met vliegroutes en meetstations
    - **Analyse geluid**: diepgaande analyse van geluid per type, operator, locatie en weersinvloed
    - **Conclusie**: samenvatting van de belangrijkste inzichten
    """)
##########################################################################################################################################################################################

if pagina == 'Data verkenning':
    st.title('Data verkenning')
    # Bar plot van meest gebruikte vliegtuigtype
    st.subheader('Meest gebruikt vliegtuigtype')
    st.bar_chart(data_1jaar['type'].value_counts().head(10))

    # barchart geluid vs type
    st.subheader('Meest voorkomende vliegtuigen en hun geluid')
    var_10_meestvoorkomende_vliegtuignamen = data_1jaar['type'].value_counts().head(10).index
    df_sensor_2024_top10_type = data_1jaar[data_1jaar['type'].isin(var_10_meestvoorkomende_vliegtuignamen)]
    grouped_data3 = df_sensor_2024_top10_type.groupby('type')['SEL_dB'].mean()

        # Stel dynamische labels en kleuren in
    bar_labels = grouped_data3.index
    bar_colors = plt.cm.get_cmap('tab10', len(bar_labels))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(bar_labels, grouped_data3, color=bar_colors(range(len(bar_labels))))
    ax.set_xlabel('')
    ax.set_ylabel('SEL dB')
    ax.set_title('SEL dB')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    st.pyplot(fig)

    # boxplot max geluid per locatie opstijgen vs landen
    st.subheader('Max geluid van van het landen vs opstijgen')
    locatie_keuze = data_1jaar['locatie_Cat'].unique()
    locatie_selectie = st.multiselect("Kies locaties:", locatie_keuze)
    filtered_locatie = data_1jaar[data_1jaar['locatie_Cat'].isin(locatie_selectie)]
    fig = px.box(filtered_locatie, x='landen_stijgen', y='lasmax_dB',
                color='locatie_Cat',title=f'Geluidsniveaus ({locatie_selectie})',
                category_orders={'landen_stijgen': ['stijgen','landen']})
    fig.update_yaxes(range=[40,100])
    st.plotly_chart(fig)

    st.subheader('Toevoegen van extra kolommen')
    st.markdown('Extra kolommen zijn aan het dataframe toegevoegd. Daarnaast zijn er bewerkingen gedaan op de tijd, zodat de de datums/tijden goed komen te staan')
    st.write('De toegevoegde kolommen zijn:')
    st.write(data_1jaar[['distance ground', 'locatie_Cat', 'gem_geluid_loc', 'windrichting']].head())
    st.write('Daarnaast zijn analyses soms op basis van het geluid gedaan voor heel 2024, andere analyses met informatie over de vluchten is gedaan op basis van 24-3-2025')


##########################################################################################################################################################################################
if pagina == 'Analyse meetstations en vliegroutes':
    st.title('Vluchten analyse')

    # Map
    m = folium.Map(location=[52.2606, 4.7594], zoom_start=11)

    # Dropdown menu to select flight type
    flight_type = st.selectbox('Selecteer vlucht type:', ['Arrivals', 'Departures'])
    filtered_vluchten_data = vluchten_data[vluchten_data['FlightType'] == flight_type]

    #lijnen en coordinaten correct maken    
    coordinaten = list(zip(filtered_vluchten_data['Latitude'], filtered_vluchten_data['Longitude']))
    frequentie = Counter(coordinaten)
    for vlucht in filtered_vluchten_data['FlightNumber'].unique():
        vlucht_data = filtered_vluchten_data[filtered_vluchten_data['FlightNumber'] == vlucht]
        locations = vlucht_data[['Latitude', 'Longitude']].values.tolist()

        # Bereken gemiddelde frequentie van deze route
        route_freq = [frequentie[tuple(coord)] for coord in locations]
        max_freq = max(route_freq)
        lijn_dikte = min(2 + max_freq * 1.5, 2)
        transparantie = max(1 - (max_freq / max(frequentie.values())) * 50, 0.1)

        folium.PolyLine(
            locations=locations,
            color='blue',
            weight=lijn_dikte,
            opacity=transparantie
        ).add_to(m)

    # Add Schiphol to the map
    folium.Marker([52.3105, 4.7683], popup='Schiphol', icon=folium.Icon(icon='plane', prefix='fa', color='orange')).add_to(m)

    # List of measurement stations with their coordinates
    stations = {
        'Aalsmeerderweg': [52.271712, 4.772485],
        'Darwinstraat': [52.234322, 4.759092],
        'Uiterweg': [52.263306, 4.732377],
        'Copierstraat': [52.228925, 4.739004],
        'Kudelstaartseweg': [52.234885, 4.748050],
        'Blaauwstraat': [52.264531, 4.774595],
        'Hornweg': [52.268712, 4.787693]
    }

    # Add the measurement stations to the map
    for station, coords in stations.items():
        folium.Marker(coords, popup=station, icon=folium.Icon(icon='m', color='red')).add_to(m)

    runways = [
        ([(52.362423, 4.711867), (52.328537, 4.708903)], 'Polderbaan'),
        ([(52.316809, 4.745912), (52.318347, 4.797313)], 'Buitenveldertbaan'),
        ([(52.321423, 4.780239), (52.291093, 4.777364)], 'Aalsmeerbaan'),
        ([(52.300436, 4.783654), (52.313843, 4.802885)], 'Oostbaan'),
        ([(52.302213, 4.737438), (52.331181, 4.740188)], 'Zwanenburgbaan'),
        ([(52.288182, 4.734371), (52.304496, 4.777672)], 'Kaagbaan'),
    ]

    # Add the runways to the map
    for runway_coords, runway_name in runways:
        folium.PolyLine(runway_coords, color='yellow', weight=5).add_to(m)
        midpoint = [(runway_coords[0][0] + runway_coords[1][0]) / 2, (runway_coords[0][1] + runway_coords[1][1]) / 2]
        folium.Marker(midpoint, popup=runway_name, icon=folium.Icon(icon='plane', prefix='fa', color='red')).add_to(m)

    # # Add a legend to the map
    # legend_html = '''
    # <div style="position: fixed; 
    #             bottom: 50px; left: 50px; width: 200px; height: 150px; 
    #             border:2px solid grey; z-index:9999; font-size:14px;
    #             background-color:white; opacity: 0.8;">
    # &emsp;<b>Legend</b><br>
    # &emsp;<i class="fa fa-plane fa-2x" style="color:orange"></i>&emsp;Schiphol<br>
    # &emsp;<i class="fa fa-map-marker fa-2x" style="color:red"></i>&emsp;Measurement Stations<br>
    # &emsp;<i class="fa fa-circle fa-2x" style="color:green"></i>&emsp;4000m Radius<br>
    # &emsp;<i class="fa fa-plane fa-2x" style="color:red"></i>&emsp;Runways<br>
    # </div>
    # '''

    # m.get_root().html.add_child(folium.Element(legend_html))

    # Display the map
    folium_static(m)

##########################################################################################################################################################################################

if pagina == 'Analyse geluid':
    st.title('Analyse op het geluid')

    # scatterplot van de afstand vs dB
    st.subheader('Afstand analyse per operator')
    fig = px.scatter(data_1jaar_boeing737800, x='distance', y='SEL_dB', color='operator')
    st.plotly_chart(fig)

    # scatterplot van de hoogte vs dB
    st.subheader('Hoogte analyse per landingsbaan')
    filtered_data_1jaar_boeing737800 = data_1jaar_boeing737800[(data_1jaar_boeing737800['distance ground'] < 50)]
    fig = px.scatter(filtered_data_1jaar_boeing737800, x='altitude', y='SEL_dB', color='tags')
    st.plotly_chart(fig)

    # analyse op de windrichting vs de dB
    st.subheader('Weer analyse')
    def plot_landingsbaan_info(data_1jaar, landingsbaan_selectie, locatie_selectie):
        # Filter de data op basis van landingsbaan en locatie
        filtered_landingsbaan_locatie = data_1jaar[data_1jaar['tags'] == landingsbaan_selectie]
        filtered_landingsbaan_locatie = filtered_landingsbaan_locatie[filtered_landingsbaan_locatie['locatie_Cat'] == locatie_selectie]

        # Groepeer per windrichting en bereken het gemiddelde geluid in dB
        filtered_landingsbaan_avg = filtered_landingsbaan_locatie.groupby(['windrichting'])['gem_geluid_loc'].mean().reset_index()

        # Plot 1: Geluid in dB (logaritmische schaal)
        fig = px.bar(
            data_frame=filtered_landingsbaan_avg, 
            x='windrichting', 
            y='gem_geluid_loc',
            title=f"Gemiddelde geluid (dB) per windrichting voor {locatie_selectie}"
        )
        st.plotly_chart(fig)

        # Converteer dB naar een lineaire schaal.
        # Bij geluidsdrukniveaus wordt vaak gewerkt met de formule: lineaire waarde = 10^(dB/20)
        filtered_landingsbaan_avg['linear_geluid'] = 20e-6 * 10 ** (filtered_landingsbaan_avg['gem_geluid_loc'] / 20)

        # Plot 2: Geluid op lineaire schaal (niet-logaritmisch)
        fig_linear = px.bar(
            data_frame=filtered_landingsbaan_avg, 
            x='windrichting', 
            y='linear_geluid',
            title=f"Gemiddelde lineaire geluidwaarde per windrichting voor {locatie_selectie}"
        )
        st.plotly_chart(fig_linear)

    # Gebruikersinterface
    landingsbaan_keuze = {
        'Zwanenburgbaan18C_T', 'Zwanenburgbaan36C_L', 'Aalsmeerbaan36R_L',
        'Aalsmeerbaan18L_T', 'Oostbaan22_T', 'Oostbaan04_L',
        'Kaagbaan24_T', 'Kaagbaan06_L'
    }
    landingsbaan_selectie = st.selectbox('Selecteer een landingsbaan:', list(landingsbaan_keuze))

    locatie_keuze = data_1jaar['locatie_Cat'].unique()
    locatie_selectie = st.selectbox("Kies locatie:", locatie_keuze)

   

    # Roep de functie aan voor de geselecteerde landingsbaan en locatie
    plot_landingsbaan_info(data_1jaar, landingsbaan_selectie, locatie_selectie)

 # Analyse: Geluid en Temperatuur per vliegtuigtype
    st.subheader("Geluid en Temperatuur per Vliegtuigtype in 2024")

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

    selected_type = st.selectbox("✈️ Kies een vliegtuigtype:", vliegtuigtypes)
    info_type = types_df[types_df['type'] == selected_type]
    gewicht = info_type['gewicht_kg'].values[0] if not info_type.empty else 'Onbekend'
    capaciteit = info_type['capaciteit_passagiers'].values[0] if not info_type.empty else 'Onbekend'

    data_1jaar['date'] = pd.to_datetime(data_1jaar['time']).dt.date
    filtered = data_1jaar[data_1jaar['type'] == selected_type].copy()

    daggemiddelden = filtered.groupby('date').agg(
        gemiddelde_temp=('TG_C', 'mean'),
        gemiddelde_geluid=('SEL_dB', 'mean'),
        aantal_vluchten=('type', 'count')
    ).reset_index()

    st.markdown(f"""
    ✈️ **Aantal vluchten van {selected_type} in 2024**: `{len(filtered)}`  
    ⚖️ **Gewicht**: `{gewicht} kg`  
    👥 **Capaciteit passagiers**: `{capaciteit}`
    """)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=daggemiddelden['date'],
        y=daggemiddelden['gemiddelde_temp'],
        mode='lines+markers',
        name='Temperatuur (°C)',
        line=dict(color='blue'),
        yaxis='y1',
        hovertemplate='%{y:.1f} °C<br>Vluchten: %{customdata}x',
        customdata=daggemiddelden[['aantal_vluchten']]
    ))

    fig.add_trace(go.Scatter(
        x=daggemiddelden['date'],
        y=daggemiddelden['gemiddelde_geluid'],
        mode='lines+markers',
        name='SEL_dB',
        line=dict(color='red'),
        yaxis='y2',
        hovertemplate='%{y:.1f} dB<br>Vluchten: %{customdata}x',
        customdata=daggemiddelden[['aantal_vluchten']]
    ))

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

    st.plotly_chart(fig, use_container_width=True)

    # Bereken gemiddelde SEL_dB per type
    avg_sel = data_1jaar.groupby('type')['SEL_dB'].mean().reset_index(name='SEL_dB')
    merged = avg_sel.merge(types_df[['type', 'gewicht_kg']], on='type', how='left').dropna()

    # Gewichtsklassen
    bins = [0, 50000, 100000, 150000, 200000, 300000, 400000, 600000]
    labels = ['0–50k', '50–100k', '100–150k', '150–200k', '200–300k', '300–400k', '400k+']
    merged['gewichtsklasse'] = pd.cut(merged['gewicht_kg'], bins=bins, labels=labels)

    # ✅ Interactieve Boxplot met Plotly
    st.title('Gewicht analyse')
    st.subheader("Spreiding van geluidsoverlast per gewichtsklasse o.b.v. 2024")

    fig_box = px.box(
        merged,
        x="gewichtsklasse",
        y="SEL_dB",
        points="outliers",
        labels={"SEL_dB": "Gemiddelde SEL_dB", "gewichtsklasse": "Gewichtsklasse (kg)"},
        category_orders={"gewichtsklasse": labels}
    )
    st.plotly_chart(fig_box, use_container_width=True)


    st.subheader('Heatmap geluid stations')
    vluchten_boeing737800 = [
    "KLM1248", "KLM1602", "KLM1776", "KLM1902", "KLM1314", "KLM1368", 
    "KLM1960", "KLM1782", "KLM1298", "KLM1316", "TRA5426", "TRA6866", 
    "TRA6904", "TRA5752", "KLM1266", "KLM1930", "KLM1750", "KLM1042", 
    "KLM1512", "TRA6148", "KLM1504", "KLM1622", "KLM1170", "KLM1036", 
    "KLM1142", "KLM1226"]

    vluchten_boeing737800.append('Selecteer alles')
    selected_vluchten_boeing737800 = st.multiselect('Selecteer 1 of meerdere vluchten Boeing 737-800:', vluchten_boeing737800)
    if 'Selecteer alles' in selected_vluchten_boeing737800:
        selected_vluchten_boeing737800 = vluchten_boeing737800[:-1]

    vluchten_EmbraerERJ190100STD = [
        'KLM1814', 'KLM982', 'KLM1850', 'KLM1788', 'KLM1944', 'KLM1830', 
        'KLM1654', 'KLM916', 'KLM1780', 'KLM1016', 'KLM1126', 'KLM978', 
        'KLM980', 'KLM1318', 'KLM1440', 'KLM1936', 'KLM1154', 'KLM1943']

    vluchten_EmbraerERJ190100STD.append('Selecteer alles')
    selected_vluchten_EmbraerERJ190100STD = st.multiselect('Selecteer 1 of meerdere vluchten EmbraerERJ 190-100STD:', vluchten_EmbraerERJ190100STD)
    if 'Selecteer alles' in selected_vluchten_EmbraerERJ190100STD:
        selected_vluchten_EmbraerERJ190100STD = vluchten_EmbraerERJ190100STD[:-1]

    # Create a blank map centered around the average coordinates of Aalsmeer
    m = folium.Map(location=[52.2606, 4.7594], zoom_start=11)

    # tijd slider van half uur toevoegen
    start_time = meetstation_data['time'].min().to_pydatetime()
    end_time = meetstation_data['time'].max() .to_pydatetime()
    custom_start_time = datetime(2025, 3, 24, 0, 30) 
    custom_end_time = datetime(2025, 3, 24, 23, 0)

    time_range = st.slider(
        'Selecteer tijdsframe (per half uur)',
        min_value=start_time,
        max_value=end_time,
        value=(custom_start_time,custom_end_time),
        step=timedelta(minutes=30),
        format="HH:mm"
    )

    # Zet de tijden om naar pandas Timestamps voor filtering
    start_time_filtered = pd.to_datetime(time_range[0])
    end_time_filtered = pd.to_datetime(time_range[1])

    # kaart van de geluidsniveau's toevoegen
    def vluchtgeluid_kaart(vlucht, start_time, end_time):
        """
        Voeg een vluchtpad en geluidsscore-heatmap toe aan de folium kaart.
        
        Parameters:
            callsign (str): Vluchtnummer (bijv. 'KLM1368').
            start_time (datetime): Starttijd van het tijdsframe.
            end_time (datetime): Eindtijd van het tijdsframe
        """

        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)

        callsign_data = vluchten_data[
            (vluchten_data['FlightNumber'] == vlucht)&
            (vluchten_data['time']>=start_time)&
            (vluchten_data['time']<=end_time)
            ]
        
        if callsign_data.empty:
            return
        folium.PolyLine(
            locations=callsign_data[['Latitude', 'Longitude']].values.tolist(),
            color='blue',
            weight=2.5,
            opacity=1
        ).add_to(m)

        # Filter meetstationdata voor de geselecteerde vlucht en tijdsframe
        meetstation_data_callsign = meetstation_data[
            (meetstation_data['callsign'] == vlucht) &
            (meetstation_data['time'] >= start_time) & 
            (meetstation_data['time'] <= end_time)
        ]
        
        if meetstation_data_callsign.empty:
            return
        
        # Voeg de heatmap toe
        heatdata = meetstation_data_callsign[['lat', 'lon', 'SEL_dB']].values.tolist()
        HeatMap(heatdata).add_to(folium.FeatureGroup(name='Heat Map').add_to(m))

    # Add Schiphol to the map
    folium.Marker([52.3105, 4.7683], popup='Schiphol', icon=folium.Icon(icon='plane', prefix='fa', color='orange')).add_to(m)

    # List of measurement stations with their coordinates
    stations = {
        'Aalsmeerderweg': [52.271712, 4.772485],
        'Darwinstraat': [52.234322, 4.759092],
        'Uiterweg': [52.263306, 4.732377],
        'Copierstraat': [52.228925, 4.739004],
        'Kudelstaartseweg': [52.234885, 4.748050],
        'Blaauwstraat': [52.264531, 4.774595],
        'Hornweg': [52.268712, 4.787693]
    }

    # Add the measurement stations to the map
    for station, coords in stations.items():
        folium.Marker(coords, popup=station, icon=folium.Icon(icon='m', color='red')).add_to(m)

    # Draw a circle with a radius of 4000 meters around all stations
    for i in stations:
        folium.Circle(
            location=stations[i],
            radius=4000,
            color='green'
        ).add_to(m)

    runways = [
        ([(52.362423, 4.711867), (52.328537, 4.708903)], 'Polderbaan'),  
        ([(52.316809, 4.745912), (52.318347, 4.797313)], 'Buitenveldertbaan'),  
        ([(52.321423, 4.780239), (52.291093, 4.777364)], 'Aalsmeerbaan'),  
        ([(52.300436, 4.783654), (52.313843, 4.802885)], 'Oostbaan'),    
        ([(52.302213, 4.737438), (52.331181, 4.740188)], 'Zwanenburgbaan'),  
        ([(52.288182, 4.734371), (52.304496, 4.777672)], 'Kaagbaan'), 
    ]

    # Add the runways to the map
    for runway_coords, runway_name in runways:
        folium.PolyLine(runway_coords, color='yellow', weight=5).add_to(m)
        midpoint = [(runway_coords[0][0] + runway_coords[1][0]) / 2, (runway_coords[0][1] + runway_coords[1][1]) / 2]
        folium.Marker(midpoint, popup=runway_name, icon=folium.Icon(icon='plane', prefix='fa', color='red')).add_to(m)

    # # Add a legend to the map
    # legend_html = '''
    # <div style="position: fixed; 
    #             bottom: 50px; left: 50px; width: 200px; height: 150px; 
    #             border:2px solid grey; z-index:9999; font-size:14px;
    #             background-color:white; opacity: 0.8;">
    # &emsp;<b>Legend</b><br>
    # &emsp;<i class="fa fa-plane fa-2x" style="color:orange"></i>&emsp;Schiphol<br>
    # &emsp;<i class="fa fa-map-marker fa-2x" style="color:red"></i>&emsp;Measurement Stations<br>
    # &emsp;<i class="fa fa-circle fa-2x" style="color:green"></i>&emsp;4000m Radius<br>
    # &emsp;<i class="fa fa-plane fa-2x" style="color:red"></i>&emsp;Runways<br>
    # </div>
    # '''

    # m.get_root().html.add_child(folium.Element(legend_html))

    # kaart map keuze toevoegen
    for vlucht in selected_vluchten_boeing737800:
        vluchtgeluid_kaart(vlucht,start_time_filtered,end_time_filtered)

    for vlucht in selected_vluchten_EmbraerERJ190100STD:
        vluchtgeluid_kaart(vlucht,start_time_filtered,end_time_filtered)

    # Display the map
    folium_static(m)
##########################################################################################################################################################################################

if pagina == 'Conclusie':
    st.title('Belangrijkste bevindingen en conclusies')
    st.write('* Boeing 737-800 is veruit het meest gebruikte vliegtuigtype in 2024, met meer dan 100.000 vluchten van en naar schiphol')
    st.write('* Uit de boxplot blijkt een duidelijke positieve correlatie tussen gewicht en geluidsproductie. Hogere gewichtsklassen (zoals 300–400k en 400k+) veroorzaken gemiddeld meer geluid.')
    st.write('* Op logaritmische schaal lijken windrichtingen weinig invloed te hebben, maar op lineaire schaal zie je wél duidelijke verschillen.')
    st.write('* Er is geen directe, consistente relatie gevonden tussen temperatuur en geluid (SEL_dB) per dag. Beide variabelen schommelen, maar niet synchroon.')
    st.write('* De heatmap visualisaties laten zien dat geluidspieken zich vaak concentreren rond specifieke meetpunten (zoals Aalsmeerderweg of Kudelstaartseweg).')
