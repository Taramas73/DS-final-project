import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from os import path

# Il faudrait mettre les constantes (par exemple les chemins) dans des venv.
# Pour l'instant, je répertoire de travail est la racine du projet.

@st.cache_data
def load_data():
    df = pd.read_csv(path.join('data', 'train_data.csv'))

    return df

st.title('Past disaster analysis')

tab1, tab2 = st.tabs(["General information", "Statistics"])

with tab1:
    st.header("Training data")
    st.dataframe(load_data().rename(columns={load_data().columns[0]: "index"}), use_container_width=True, hide_index=True)

    st.header("Localisation")

    fig = px.scatter_geo(data_frame=load_data().dropna(), 
                        lat=load_data().dropna()['lat'],
                        lon=load_data().dropna()['lgt'],
                        color=load_data().dropna()['disaster_type'],
                        projection='equirectangular', 
                        hover_name=load_data().dropna()['disaster'],
                        size=load_data().dropna()['severity'], 
                        animation_frame=load_data().dropna()['moment'],
                        width=1500, height=768)
    st.plotly_chart(fig)

with tab2:
    st.text("Pie charts here")