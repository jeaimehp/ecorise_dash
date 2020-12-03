# Libraries
# Data
import os # Operating system library
import numpy as np
import pandas as pd # Dataframe manipulations
import geopandas as gpd
import  pathlib # file paths

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

# Roll up Environmental theme data
# List of standardized themes from dataset
# **Added to standardized list from spreadsheet: Conservation (Wildlife/Habitat), Green Building,  Outdoor Learning
themes_list = ['Outdoor learning',
'Waste',
'Water',
'Energy',
'Green building',
'Food Systems/Nutrition',
'Transportation',
'Air Quality',
'Conservation (wildlife/habitat)',
'Workforce Development',
'STEM']
add_themes = ['Conservation (Wildlife/Habitat)', 'Green Building',  'Outdoor Learning']
full_list = themes_list + add_themes

def count_env_themes(programs, theme_list):
    themes = programs[['Program','Environmental_Themes']]
    expanded = themes['Environmental_Themes'].str.get_dummies(', ')
    # Replace columns that aren't in list with 'Other'
    expanded = expanded.rename(lambda x:  x if x in theme_list else 'Other', axis=1)
    # Group 'Other' Columns together
    expanded = expanded.groupby(expanded.columns, axis=1).sum()
    # Merge the themes and expanded dataframes
    themes = pd.concat([themes, expanded], axis=1)

    # Get Count of Programs per theme
    theme_count = pd.DataFrame(expanded.sum())
    theme_count.reset_index(inplace=True)
    theme_count.columns = ['Theme','Count']
    theme_count['Percent'] = round(100 * theme_count['Count'] / len(programs),0)
    theme_count = theme_count.sort_values(by=['Count'])
    return theme_count

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

# STYLING
# Set Color scales for Figures using the EcoRise brande color palette
fulltint = ['#00A887','#B9D535','#FFC600','#FF8F12','#FF664B']
tint_75 = ['#40BEA5','#CBE068','#FFC400','#FFAB4D','#FF8C78']
tint_50 = ['#80D4C3','#DCEA9A','#FFE380','#FFC789','#FFB3A5']
eco_color = fulltint + tint_75 + tint_50
eco_color_r = fulltint[::-1] + tint_75[::-1] + tint_50[::-1]
eco_color_desc = eco_color[::-1]
eco_color_desc_r = eco_color_r[::-1]

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

# figure Functions
def make_groupby_pie_chart(df,col, textinfo = None, groupby_column = 'Organization', color_scale = eco_color,showlegend=False ):
    df = pd.DataFrame(df.groupby(col)[groupby_column].count())
    df.reset_index(level=0, inplace=True)
    fig = px.pie(df, values=groupby_column, names=col, color_discrete_sequence=color_scale)
    fig.update_traces(textposition='inside', textinfo=textinfo)
    fig.update_layout(showlegend=showlegend,height=250,   margin=dict(l=20, r=20, t=0, b=0))
    return fig

def make_bar(df,xaxis,yaxis,label, orientation='h', textposition='inside', marker_color=eco_color_desc):
    fig = px.bar(df, x=xaxis, y=yaxis, orientation=orientation, text=label)
    fig.update_traces(marker_color=marker_color, texttemplate='%{text}', textposition=textposition)
    fig.update_yaxes(visible=False, showticklabels=False)
    fig.update_xaxes(visible=False, showticklabels=False)
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=0, b=0),
        height=250)
    fig.update_layout(uniformtext_minsize=12)
    fig.update_traces(
       hovertemplate=None,
       hoverinfo='skip'
    )
    return fig

def make_map(orgdata, choro_geojson, featureidkey, choro_df, choro_df_location, choro_df_value, zoom = 5):
    # Design point layer
    scatter_fig_hover_template = '<b>%{hovertext}</b><br>Education Service Center: %{customdata[4]}'
    scatter_fig = px.scatter_mapbox(orgdata, lat="LATITUDE", lon="LONGITUDE",
                             hover_name="Organization", hover_data=["Custom_Region", 'Stakeholder_Category', 'Sector', 'Service_Area','Education_Service_Center'])
    scatter_fig.update_traces(hovertemplate=scatter_fig_hover_template)

    # Build choropleth layer
    fig = px.choropleth_mapbox(choro_df, geojson=choro_geojson,
              featureidkey=featureidkey,
              locations=choro_df_location,
              color=choro_df_value,
              color_continuous_scale = tint_50,
              opacity = 0.25,
               zoom=zoom,
              center= {'lat': 31.9686, 'lon': -99.9018},
              mapbox_style='carto-positron')
    fig.update_traces(hovertemplate='ESC: %{location}<br>Organizations: %{z}')

    # add pt layer to map
    for item in range(0,len(scatter_fig.data)):
        fig.add_trace(scatter_fig.data[item])
    fig.update_layout(mapbox_style="open-street-map") # Ensure don't need token
    fig.update_layout(
        showlegend=False,
        autosize=True,
        height=350,
        margin=dict(l=20, r=20, t=20, b=0))

    return fig

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

dashboard = html.Div([
    dbc.Row([
        dbc.Col([
            dcc.Graph(id = 'map')
        ],md=6), 
        dbc.Col([
            dcc.Graph(id = 'treemap')
        ],md=6),   
    ],
    ),
    dbc.Row([  
        dbc.Col([
            dcc.Graph(id='chart_sector'),
            dcc.Dropdown(
                id = 'dd-pie',
                options = [{'label': c, 'value': c}
                            for c in sorted(['Sector','Service_Area'])],
                value='Sector'
                ),
        ],md=6),     
        dbc.Col([
            dcc.Graph(id='chart_theme'),
        ],md=6),        
    ])

])

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
            dcc.Tab([dashboard],label='Dashboard', id='tab-dashboard'),
            dcc.Tab(id='tab-orgs'),
            dcc.Tab(id='tab-programs'),
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
application = app.server
app.config.suppress_callback_exceptions = True

app.layout = html.Div([sidebar, content])

## CALLBACKS
# Update the underlying data when filters are changed, and rebuild pie chart if selection changes
# TO DO: mode data to local session store, and update pie chart using a data state reference
@app.callback(
    [Output('out-all-types','children')
#      ,Output('tab-dashboard','children')
         ,Output('tab-orgs','children'),Output('tab-orgs','label')
         ,Output('tab-programs','children') ,Output('tab-programs','label')
         ,Output('map', 'figure'),Output('chart_theme', 'figure')
         ,Output('treemap', 'figure'),Output('chart_sector', 'figure')
    ],
    [Input('dd-pie','value')]+
    [Input(f'dd-org-{dd}', "value") for dd in org_filter_list]+
    [Input(f'dd-pg-{dd}', "value") for dd in pg_filter_list ],
)
def dd_values(pie,*vals):
    # initial dataframes
    df_o = orgs
    df_p = programs

    #iterate through organization columns and filter data
    i = 0
    for v in vals[0:len(org_filter_list)]:
        if(v):
            df_o = df_o[df_o[org_filter_list[i]].isin(v)]
        i += 1

    # select only programs in the org list, then filter on program filters
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

    # if any program filter is selected, filter orgs to this
    if p_count > 0:
        df_o = df_o[df_o['orgID'].isin(df_p['orgID'])]

    # calculate Org message
    org_count = len(df_o)
    pg_count = len(df_p)
    orgs_msg = 'Organization Records (' + str(org_count) + ')'
    pg_msg  = 'Program Records (' + str(pg_count) + ')'

    # Build Directory tables
    orgs_tab = html.Div([
        build_directory_table('table-orgs', df_o, directory_org_cols)
        ],style={'width':'100%'})

    programs_tab = html.Div([
          build_directory_table('table-programs', df_p, directory_program_cols)
        ],style={'width':'100%'})

    # Build Figures
    # Calculate theme split
    theme_count = count_env_themes(df_p, full_list)
    theme_count['Label'] = theme_count['Theme'] + ' - ' + theme_count['Percent'].astype(str) + '%'

    # build map
    # Get Count of entities per esc
    esc_count = pd.DataFrame(df_o['Education_Service_Center'].value_counts())
    esc_count = esc_count.reset_index().rename(columns={"index": "ESC", "Education_Service_Center": "Organizations"})
    map_fig = make_map(df_o, esc_geojson, 'properties.FID', esc_count, 'ESC', 'Organizations', zoom=4)

    # build treemap
    pg_count = pd.DataFrame(df_p['orgID'].value_counts())
    pg_count = pg_count.reset_index().rename(columns={"index": "orgID", "orgID": "Program_Count"})
    pg_count = pg_count.merge(orgs[['Organization','orgID','Education_Service_Center','Sector','Service_Area']], how='inner', on='orgID')
    path=['Education_Service_Center','Sector','Organization']
    sb = pg_count.dropna(subset=path)
    sb['Education_Service_Center'] = 'ESC: ' + sb['Education_Service_Center'].astype('int').astype('str')

    tree_fig = px.sunburst(sb, path=path, values = 'Program_Count',
                           color_discrete_sequence=eco_color,
                           maxdepth=2)

    # Test section
    test_msg = ''

    # return values
    return (test_msg, orgs_tab, orgs_msg, programs_tab, pg_msg,
            map_fig,
            make_bar(theme_count, 'Percent','Theme','Label'),
            tree_fig,
            make_groupby_pie_chart(df_o,pie,showlegend=True)
           )


## TO DO
## Add themes once the data processing is worked out
## Output: Output('chart_theme', 'figure'),

# RUN app
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8070)
