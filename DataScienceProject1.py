import streamlit as st
import pandas as pd
import numpy
import pydeck as pdk
import plotly.express as px


DATA_URL = ("Motor.csv")
st.title("Vehicle Crash Information (New York City)")
# st.markdown("This is Streamlit dashboard")

@st.cache(persist=True)
def load(nr):
    data = pd.read_csv(DATA_URL, nrows=nr,parse_dates=[['CRASH DATE', 'CRASH TIME']])
    data.dropna(subset=['LATITUDE','LONGITUDE'], inplace= True)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data.rename(columns={'crash date_crash time': 'date/time'}, inplace=True)
    return data


data = load(10000)


st.header("Injured people")
injured_people = st.slider("No of ppl injured", 0, 19)


st.map(data)


st.header("How many collisions occur in a day")
hour = st.slider("Hour to look at", 0, 23)
data = data[data['date/time'].dt.hour == hour]


st.markdown("Vehicle collisions between %i:00 and %i:00" % (hour, (hour + 1) % 24))
midpoint = (numpy.average(data['latitude']), numpy.average(data['longitude']))
st.write(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state={
        'latitude': midpoint[0],
        'longitude': midpoint[1],
        'zoom': 11,
        'pitch': 50,

    },
    layers=[
        pdk.Layer(
            "HexagonLayer",
            data=data[['date/time', 'latitude', 'longitude']],
            get_position = ['longitude', 'latitude'],
            radius = 100,
            extruded = True,
            pickable = True,
            elevation_scale = 4,
            elevation_range = [0, 1000],

        ),
    ],
))


st.subheader("Breakdown by minute %i:00 and %i:00" % (hour, (hour + 1) % 24))
filtered = data[
    (data['date/time'].dt.hour >= hour) & (data['date/time'].dt.hour < (hour + 1))
]
hist = numpy.histogram(filtered['date/time'].dt.minute, bins=60, range=(0, 60))[0]

chart_data = pd.DataFrame({'minute': range(60), 'crashes': hist})
fig = px.bar(chart_data, x='minute', y='crashes', hover_data=['minute', 'crashes'], height=400)
st.write(fig)


if st.checkbox("Show data", False):
    st.subheader("Raw data")
    st.write(data)

