# Libraries

# Data
import os # Operating system library
import numpy
import pandas as pd # Dataframe manipulations
import geopandas as gpd
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
geo = pd.read_excel(os.path.join(data_filepath,'data','Geodata.xlsx'))

# Merge Organization data with geodata
orgs = organizations.merge(geo, left_on='Organization', right_on='ORGANIZATION', how='left')

# Education Service Center Geospatial data layer - use geojson simplified to 0.01 threshold on QGIS
## load education service ceters geojson
esc_simple_geojson = os.path.join(data_filepath,'data','esc_simple.geojson')

with open(esc_simple_geojson) as json_file:
    esc_geojson = json.load(json_file)

# Load ESC Centroid data as point data (file generated using QGIS Centroid feature)
centroids = pd.read_csv(os.path.join(data_filepath,'data','centroids.csv'))

# df['unique_id'] = df.longstrings.map(hash)
orgs['orgID'] = orgs.index + 1
programs['programID'] = programs.index + 1

# Add OrgID column to Programs, merging on Organization
org_cols = ['Organization','orgID']
programs_org = orgs[org_cols]
programs = programs.merge(programs_org, how='left', on='Organization')
programs['orgID'] = programs['orgID'].astype('Int64')
# load filter dictionary data
d_filter = pd.read_csv(os.path.join(data_filepath,'data','filter_dict.csv'))

# Create dictionary of filter options {tablename:{columnname:{'display_name':display_name,'options'{data_column:display_name}}}}
filter_dict = {}
for t in d_filter['table_name'].unique():
    t_df = d_filter[d_filter['table_name']==t][['column_name','display_name']].drop_duplicates()
    t_df = t_df.set_index('column_name')
    t_dict = t_df.to_dict('index')
    filter_dict[t] = t_dict
    for k in filter_dict[t].keys():
        tk_df = d_filter[(d_filter['table_name']==t) & (d_filter['column_name']==k)][['term','display_term']].drop_duplicates()
        tk_dict = tk_df.set_index('term').T.to_dict('records')
        filter_dict[t][k]['options'] = tk_dict

# APP Functions
def make_dropdown(i, options, placeholder, multi = True):
    ''' Create a dropdown taking id, option and placeholder values as inputs. '''

    # Handle either list or options as inputs
    if isinstance(options, dict):
        opts = [{'label': options[k], 'value': k}
                            for k in sorted(options)]
    else:
        opts = [{'label': c, 'value': c}
                            for c in sorted(options)]

    # Return actual dropdown component
    return dcc.Dropdown(
                id = f"{i}",
                options=opts,
                multi=multi,
                placeholder=placeholder,
                )

def build_directory_table(table_id, df, display_cols):
    ''' Function to create the structure and style elements of both the Organization and Programs tables'''
    # Checks to add:
#     * input dataframe
#     * display_cols are in list of columsn in dataframe

    data_table = dash_table.DataTable(
                    id=table_id,
                    columns=[{"name": i, "id": i} for i in df[display_cols].columns],
                    data=df.to_dict('records'),
                    sort_action="native",
                    sort_mode="multi",
                    page_action="native",
                    page_current= 0,
                    page_size= 5,
                    css=[{'selector': '.row', 'rule': 'margin: 0; flex-wrap: nowrap'},
                        {'selector':'.export','rule':'position:absolute;left:0px;bottom:-35px'}],
                    fixed_columns={'headers': True, 'data': 1},
                    style_as_list_view=True,
                    style_cell={'padding': '5px',
                        'maxWidth': '300px',
                        'textAlign': 'left',
                        'height': 'auto',
                        'whiteSpace': 'normal'
                               },
                    style_header={
                        'backgroundColor': 'white',
                        'fontWeight': 'bold'
                    },
                    style_data={'padding-left':'15px'},
                    style_table={'minWidth': '100%', 'maxWidth': 'none !important','overflowX': 'auto'},
                    export_format="xlsx",
                    export_headers="display"
                                    )
    return data_table

# STYLING
#STYLES
# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "20rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

# LISTS (TODO: MAKE THIS A DYNAMIC PULL)
Education_Service_Centers =['All Regions',
'Region 1 – Edinburg',
'Region 2 – Corpus Christi',
'Region 3 – Victoria',
'Region 4 – Houston',
'Region 5 – Beaumont',
'Region 6 – Huntsville',
'Region 7 – Kilgore',
'Region 8 – Mount Pleasant',
'Region 9 – Wichita Falls',
'Region 10 – Richardson',
'Region 11 – Fort Worth',
'Region 12 – Waco',
'Region 13 – Austin',
'Region 14 – Abilene',
'Region 15 – San Angelo',
'Region 17 – Lubbock',
'Region 18 – Midland',
'Region 19 – El Paso',
'Region 20 – San Antonio']

directory_org_cols = [
'Organization',
'City',
'Education_Service_Center',
'Full-Time_Staff',
'General_Email',
'Mission',
'Part-Time_Staff',
'Phone',
'Primary_Email',
'Primary_First_Name',
'Primary_Last_Name',
'Primary_Role',
'Custom_Region',
'Sector',
'Service_Area',
'Stakeholder_Category',
'State',
'Street_Address',
'Other_Staff/Contractors',
'Work_Terms',
'Website',
'ZIP_Code']

directory_program_cols =[
'Program',
'Organization',
'Title_I_School_Participants',
'Status',
'COVID-19_Adaptations',
'Description',
'Environmental_Themes',
'Regions_Served',
'Impact_Evaluation',
'Groups_Served',
'Program_Locations',
'Other_Languages',
'Services_&_Resources',
'Rural_Communities_Focus',
'Engaging_Rural_Communities',
'Schools_Served',
'Academic_Standards_Alignment',
'Student_Engagement_Frequency',
'Students_Served',
'Teacher/Administrator_Engagement_Frequency',
'Teachers/Administrators_Served',
'Program_Times',
'Title_I_Schools_&_Low_Socioeconomic_Background_Focus',
'Participants_Served',
'Academic_Standards']

# COMPONENTS
# Get dictionaries to use for Org and Program filter lists
# Organizations
org_filter_list = ['Custom_Region', 'Education_Service_Center', 'Sector', 'Service_Area']
org_filter_dict = {k: filter_dict['Organizations'].get(k, None) for k in (org_filter_list)}

# Programs
pg_filter_list = ['Regions_Served','Environmental_Themes','Services_&_Resources','Academic_Standards_Alignment',
                 'Program_Locations','Program_Times','Rural_Communities_Focus','Groups_Served','Title_I_School_Participants']
pg_filter_dict = {k: filter_dict['Programs'].get(k, None) for k in (pg_filter_list)}

# Build components

overview_msg = html.Div([
    html.H5(id='overview_msg')
])

dds_orgs =  html.Div(
        [make_dropdown(f'dd-org-{k}', org_filter_dict[k]['options'][0],
                       org_filter_dict[k]['display_name'])  for k in org_filter_dict
         ]
)

dds_programs =  html.Div(
        [make_dropdown(f'dd-pg-{k}', pg_filter_dict[k]['options'][0],pg_filter_dict[k]['display_name'])
         for k in pg_filter_dict
         ]
)

# LAYOUT PIECES
sidebar = html.Div(
    [
        html.H4('Filter on'),
        html.H5('Organization Data'),
        dds_orgs,
        html.H5('Program Data'),
        dds_programs,
        html.Div(id='div-overview_msg')
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div([
    html.Div(id="out-all-types"),
    html.Div([
        dcc.Tabs([
            dcc.Tab(label='Organizations', id='tab-orgs'),
            dcc.Tab(label='Programs', id='tab-programs'),
        ])
    ])
],
    id="page-content", style=CONTENT_STYLE)

# APP LAYOUT
# Build App
external_stylesheets = [dbc.themes.LITERA]
# # For Jupyter notebook
# app = JupyterDash(external_stylesheets=external_stylesheets)
# for running an actual Dash App
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True
app.layout = html.Div([sidebar, content])

## CALLBACKS
@app.callback(
    [Output('out-all-types','children'),Output('tab-orgs','children'),Output('tab-programs','children')
    ],
    [Input(f'dd-org-{dd}', "value") for dd in org_filter_list]+
    [Input(f'dd-pg-{dd}', "value") for dd in pg_filter_list ],
)
def dd_values(*vals):
    # initial dataframes
    df_o = orgs
    df_p = programs

    #iterate through organization columns and filter data
    i = 0
    for v in vals[0:len(org_filter_list)]:
        if(v):
            df_o = df_o[df_o[org_filter_list[i]].isin(v)]
        i += 1

    df_p = programs[programs['orgID'].isin(df_o['orgID'])]
    p = i - len(org_filter_list)
    p_count = 0
    p_msg = 'programs: '
    for v in vals[len(org_filter_list):]:
        if(v):
            for i in v:
                p_msg = p_msg + i
            df_p = df_p[df_p[pg_filter_list[p]].isin(v)]
            p_count += 1
        p += 1

    if p_count > 0:
        df_o = df_o[df_o['orgID'].isin(df_p['orgID'])]

    # calculate Org message
    org_count = len(df_o)
    pg_count = len(df_p)
    orgs_msg = str(org_count) + ' Organizations / ' + str(pg_count) + ' Programs'

    # Build Directory tables
    orgs_tab = html.Div([
        build_directory_table('table-orgs', df_o, directory_org_cols)
        ],style={'width':'100%'})

    programs_tab = html.Div([
          build_directory_table('table-programs', df_p, directory_program_cols)
        ],style={'width':'100%'})

    # return values
    return [orgs_msg], orgs_tab, programs_tab

## TO DO
## Add themes once the data processing is worked out
## Output: Output('chart_theme', 'figure'),

# RUN app
if __name__ == '__main__':
    app.run_server(debug=True, port=8040)
