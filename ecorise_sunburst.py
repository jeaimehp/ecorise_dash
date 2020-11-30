# Libraries
# Data
import os # Operating system library
import numpy
import pandas as pd # Dataframe manipulations
import pathlib # file paths

# Dash App
# from jupyter_dash import JupyterDash # for running in a Jupyter Notebook
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State, ALL, MATCH

# Data Visualization
import plotly.express as px
import plotly.graph_objects as go

# Geojson loading
from urllib.request import urlopen
import json

# DATA LOADING AND CLEANING
## Load data
data_filepath = pathlib.Path(__file__).parent.absolute()
programs = pd.read_csv(os.path.join(data_filepath,'data','Programs.csv'))
organizations = pd.read_csv(os.path.join(data_filepath,'data','Organizations.csv'))

# Build sunburst
sb = organizations[['Sector','Education_Service_Center','Organization']].dropna()
sb['count'] = 1

fig = px.sunburst(sb, path=['Sector','Education_Service_Center','Organization'], values = 'count',maxdepth=2)


external_stylesheets = [dbc.themes.LITERA]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    dcc.Graph(id='chart-sunburst',figure=fig)
])

# RUN app
if __name__ == '__main__':
    app.run_server(debug=True, port=8070)
