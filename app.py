
import streamlit as st
from streamlit_authenticator import Authenticate, SafeLoader, LoginError
from srtm.height_map_collection import Srtm1HeightMapCollection
import plotly.graph_objects as go
from shapely.geometry import Point
from urllib import request
import http.cookiejar as cookielib
from pathlib import Path
import geopandas as gp
import grispy as gsp
from shapely import wkt
import pandas as pd
import numpy as np
import yaml
import os
import sys



hide_menu = """
<style>
#MainMenu {
    visibility:hidden;}

footer{
    visibility:hidden;
}

footer:after{
    content:'Copyright @ 2022 - Giga';
    visibility: visible;
    display:block;
    position:relative;
    color:tomato;
    padding:5px;
    top:3px;
}
.appview-container .main .block-container{{
        padding-top: {padding_top}rem;    }}
</style>
"""



####CONFIGS

path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(path, 'data')
srtm1_path = os.path.join(data_path, 'srtm')
# srtm base url
srtm_base_url = 'https://e4ftl01.cr.usgs.gov//DP133/SRTM/SRTMGL1.003/2000.02.11/'
# srtm dict
srtm_bbox_filename = 'srtm30m_bounding_boxes.geojson.json'
# radius earth
R = 6371
# km to degrees
km2deg = np.rad2deg(1/R)




def prepare_school_data(schools):
    schools.rename(columns= {'latitude': 'lat', 'longitude': 'lon'}, inplace = True)
    schools[['lat', 'lon']].describe()
    schools = df_to_gdf(schools)
    return schools

def prepare_tower_data(towers):
    towers.loc[towers.Longitude == '30.0.3644', 'Longitude'] = 30.03644
    towers.drop(columns= 'Unnamed: 0', inplace= True)
    towers['Longitude'] = towers['Longitude'].astype(float)
    towers['Latitude'] = towers['Latitude'].astype(float)
    towers.loc[[853,854, 898], 'Latitude'] = towers.loc[[853,854, 898], 'Latitude']/(100000)
    towers.loc[898, 'Longitude'] = towers.loc[898, 'Longitude']/(100000)
    towers.loc[916, 'Longitude'] = towers.loc[916, 'Longitude']/(1000000)
    towers = df_to_gdf(towers)
    towers.loc[towers['Tower Height'] == 'IBS', 'Tower Height'] = round(towers.loc[towers['Tower Height'] != 'IBS']['Tower Height'].astype(int).mean())
    towers['Tower Height'] = towers['Tower Height'].astype(int)
    towers.rename(columns={'Latitude': 'lat', 'Longitude': 'lon', 'Tower Height': 'height', 'Technology': 'tech'}, inplace = True)
    return towers

@st.cache_data
def load_school_data(file_object):
    
    if file_object is not None:
        file_ext = file_object.name.split('.')[-1]
        if file_ext == 'xlsx':
            schools = pd.read_excel(file_object)
        elif file_ext == 'csv':
            schools = pd.read_csv(file_object)
        else:
            st.warning('File type is not supported!')
        
        schools = prepare_school_data(schools)
        return schools

@st.cache_data
def load_tower_data(file_object2):
    
    if file_object2 is not None:
        file_ext = file_object2.name.split('.')[-1]
        if file_ext == 'xlsx':
            towers = pd.read_excel(file_object2)
        elif file_ext == 'csv':
            towers = pd.read_csv(file_object2)
        else:
            st.warning('File type is not supported!')
        
        towers = prepare_tower_data(towers)
        return towers


@st.cache_data
def initialize_bubble_search(_schools, _towers, tower_service_range):
    grid = gsp.GriSPy(_towers[['lon', 'lat']].to_numpy(), metric='vincenty')
    upper_radii = tower_service_range*km2deg
    bubble_dist, bubble_ind = grid.bubble_neighbors(_schools[['lon', 'lat']].to_numpy(), sorted = True, distance_upper_bound=upper_radii)
    return bubble_dist, bubble_ind


@st.cache_data
def get_neighbor_towers(_towers, bubble_dist, bubble_ind, school_pos):
    tower_match = pd.DataFrame(zip(bubble_ind[school_pos], bubble_dist[school_pos]), columns = ['tower_pos', 'dist'])
    tower_match.reset_index(drop=True, inplace=True)
    tower_match[['lat', 'lon', 'height', 'tech']] = tower_match.tower_pos.apply(lambda x: _towers.iloc[x][['lat', 'lon', 'height', 'tech']])
    tower_match['idx'] = tower_match.tower_pos.apply(lambda x: _towers.iloc[x].name)
    tower_match['dist_km'] = tower_match.dist.apply(lambda x: np.deg2rad(x)*R)
    return tower_match


def df_to_gdf(df, crs = 'EPSG:4326'):

    if 'geometry' not in df:
        df = initialize_geometry_column(df)
    else:
        df['geometry'] = df['geometry'].apply(wkt.loads)

    return gp.GeoDataFrame(df, crs = crs)

def initialize_geometry_column(df):

    if 'geometry' in df:
        return print('Geometry column has already been initialized!')
        
    geo_cols = list(filter(lambda a: a.casefold() in ['lat', 'lon', 'longitude', 'latitude', 'x', 'y'], df.columns))

    if len(geo_cols)==2:
        geo_cols.sort(reverse=True)

        try:    
            df['geometry'] = [Point(i, j) for i,j in zip(df[geo_cols[0]], df[geo_cols[1]]) if i]
        except:
            df[geo_cols[0]] = df[geo_cols[0]].astype(float)
            df[geo_cols[1]] = df[geo_cols[1]].astype(float)
            df['geometry'] = [Point(i, j) for i,j in zip(df[geo_cols[0]], df[geo_cols[1]]) if i]

    elif len(geo_cols)>2:
        print('Please keep only the relevant geo refernce columns. We find multiple matches: ' + str(geo_cols))
        sys.exit(1)
        
    else:
        print('Dataframe does not include any of the "lat", "latitude" or "y" in the column set!')
        sys.exit(1)
    
    return df


def download_srtm_data(username, password, url, path):

    password_manager = request.HTTPPasswordMgrWithDefaultRealm()
    password_manager.add_password(None, "https://urs.earthdata.nasa.gov", username, password)

    cookie_jar = cookielib.CookieJar()

    # Install all the handlers.

    opener = request.build_opener(
        request.HTTPBasicAuthHandler(password_manager),
        #request.HTTPHandler(debuglevel=1),    # Uncomment these two lines to see
        #request.HTTPSHandler(debuglevel=1),   # details of the requests/responses
        request.HTTPCookieProcessor(cookie_jar))
    request.install_opener(opener)


    # retrieve the file from url
    request.urlretrieve(url, path)



def main():

    #st.title("Giga Visibility Analysis Tool")

    tool_tab, docs_tab, tab3 = st.tabs(["Tool", "Documentation", "About"])

    with docs_tab:
        intro = st.markdown(open('/Users/utkucanozturk/Documents/st-demo-app/docs/intro.md').read())

    with tool_tab:

        st.subheader('Before you start...')

        st.markdown('''Make sure you have a Nasa EarthData account; if you don't have one please go to [this page](http://urs.earthdata.nasa.gov/) and register!''')
        

        if not os.path.exists(data_path):
            st.error('Data path does not exist! Please create a data folder in current directory!')
        
        elif not os.path.exists(srtm1_path):
            st.error('SRTM path does not exist! Please create a srtm folder under data folder!')
        else:

            st.write('''### School Data Preparation''')

            if 'school_ready' not in st.session_state:
                st.session_state.school_ready = False

            schools = None
            uploaded_school_file = st.file_uploader("Choose a school file", type=["xlsx",'csv'])
            if uploaded_school_file:
                with st.spinner('Reading and processing school data...'):
                    schools = load_school_data(uploaded_school_file)
                st.info(f"{len(schools)} schools where read and preprocessed with **success**!")
                st.dataframe(schools[['lon', 'lat']])
                st.session_state.school_ready = True

            if 'tower_ready' not in st.session_state:
                st.session_state.tower_ready = False
            
            if st.session_state.school_ready:
                st.write('''### Tower Data Preparation''')

                towers = None
                uploaded_tower_file = st.file_uploader("Choose a tower file", type="csv")
                if uploaded_tower_file:
                    with st.spinner('Reading and processing tower data...'):
                        towers = load_tower_data(uploaded_tower_file)
                    st.info(f"{len(towers)} towers where read and preprocessed with **success**!")
                    st.dataframe(towers[['lon', 'lat', 'height', 'tech']])
                    st.session_state.tower_ready = True

            if 'srtm_ready' not in st.session_state:
                st.session_state.srtm_ready = False

            if st.session_state.school_ready and st.session_state.tower_ready and st.session_state.srtm_ready == False:
                st.write('''### SRTM Data Download''')


                ed_username = st.text_input('Earthdata Username')
                ed_password = st.text_input('Earthdata Password', type='password')
                if st.button('Save & Download'):
                    st.spinner('Finding SRTM tiles containing school and tower locations...')
                    srtm_dict = gp.read_file(os.path.join(path, 'assets', srtm_bbox_filename))
                    area_srtm_files = pd.concat([schools[['geometry']], towers[['geometry']]]).sjoin(srtm_dict, how='left', predicate='within')
                    st.info('Number of unmatched locations: ' + str(area_srtm_files.dataFile.isnull().sum()))
                    files_to_download = area_srtm_files.dataFile.unique()

                    st.write('Downloading corresponding SRTM tiles...')
                    my_bar = st.progress(0)
                    n_files = len(files_to_download)
                    for idx, file in enumerate(files_to_download):
                        file_path = os.path.join(srtm1_path, file)
                        if not os.path.exists(file_path):
                            download_srtm_data(ed_username, ed_password, srtm_base_url + file, file_path)
                        my_bar.progress((idx+1)/n_files)
                    
                    st.session_state.srtm_ready  = all(file in os.listdir(srtm1_path) for file in files_to_download)

            if 'nn_ready' not in st.session_state:
                st.session_state.nn_ready = False
            
            if st.session_state.srtm_ready:
                st.success('SRTM data download is complete!')
                st.write(''' ### Neares Neighbor Search''')

                tower_service_range = st.slider('Maximum cell tower range in kilometers', 0,100, 35)
                st.info('Tower service range is set as ' + str(tower_service_range) + '!')

                bubble_dist, bubble_ind = initialize_bubble_search(schools, towers, tower_service_range)
                st.session_state.nn_ready = True
                st.success('Bubble search implemented!')
            
            if st.session_state.nn_ready:
                st.write('''### Visibility Analysis''')
                #with st.form("Input Parameters"):
                st.write("Input Parameters")
                los_correction = st.slider('Line of sight acceptable height difference in meters', 0,100, 5)
                school_building_height = st.slider('Average school building height', 0, 100, 15)
                option = st.selectbox('How do you want to proceed?', ('Visualize line of sight for a school-tower pair', 'Run visibility analysis for all schools!'))

                #submitted = st.form_submit_button("Start Visibility Analysis")

                    #if submitted:
                srtm1_data = Srtm1HeightMapCollection(auto_build_index=True, hgt_dir=Path(srtm1_path))
                if option == 'Visualize line of sight for a school-tower pair':
                    school_idx = st.selectbox('Choose a school id:', schools.index.tolist())
                    school_pos = schools.index.get_loc(school_idx)
                    tower_match = get_neighbor_towers(towers, bubble_dist, bubble_ind, school_pos)
                    st.dataframe(tower_match)

                    tower_idx = st.selectbox('Select a neighbor tower idx:', tower_match.idx.tolist())
                    twr = towers.loc[tower_idx]

                    e_profile, d_profile = zip(*[(i.elevation, i.distance) for i in srtm1_data.get_elevation_profile(schools.iloc[school_pos]['lat'], schools.iloc[school_pos]['lon'], twr.lat, twr.lon)])
                    df_elev = pd.DataFrame(zip(np.linspace(e_profile[0] + school_building_height, e_profile[-1] + twr.height, len(e_profile)), e_profile, d_profile), columns = ['los', 'dep', 'dist'])
                    df_elev['dif'] = df_elev.los - df_elev.dep
                    t_visible = np.all(df_elev.dif > -los_correction)
                    if t_visible:
                        st.success('School is visible by the tower!')
                    else: 
                        st.error('School is not visible by the tower!')
                    
                    fig = go.Figure()

                    fig.add_trace(go.Scatter(x=d_profile, y=e_profile,
                                        mode='lines',
                                        name='Elevation Profile', fill='tonext', fillcolor = 'rgba(255, 0, 0, 0.1)'))
                    fig.add_trace(go.Scatter(x=d_profile, y=df_elev.los,
                                        mode='lines',
                                        name='Line of Sight', line = dict(color='firebrick', dash='dash'), fill = 'tonexty', fillcolor = 'rgba(0, 255, 0, 0.1)'))

                    for p in df_elev[df_elev.dif < -los_correction].itertuples():
                        #fig.add_vline(x=p.dist, line_dash = 'dot', annotation_text = np.round(p.dif, 0))
                        fig.add_trace(go.Scatter(x=[p.dist, p.dist], y=[min(e_profile), p.dep], mode= 'lines', name = 'Exceeds ' + str(np.round(p.dif, 1)) + 'm', line_dash = 'dot'))

                    fig.add_trace(go.Scatter(x=[d_profile[0]], y=[e_profile[0]+school_building_height], mode='markers', name = 'School', marker_color = 'yellow', marker_symbol = 'arrow-bar-up-open', marker_size = school_building_height/2))
                    fig.add_trace(go.Scatter(x=[d_profile[-1]], y=[e_profile[-1]+twr.height], mode='markers', name = 'Neighbor Tower 0', marker_symbol = 'arrow-bar-up-open', marker_size = twr.height/2))

                    fig.update_layout(template='presentation', legend_traceorder= 'normal')
                    fig.update_layout(legend=dict(itemsizing='constant', orientation = 'h'))

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info('Under development!')
        

st.set_page_config(
    page_title='Giga Visibility Tool',
    page_icon=os.path.join(path, 'assets', 'giga_blue.jpeg'),
    layout ='centered'
)

st.image(os.path.join(path, 'assets', 'GIGA_lockup_white_horizontal.png'), width=300)

st.title("Visibility Analysis Tool")

st.markdown(hide_menu, unsafe_allow_html=True)



with open(os.path.join(path, 'assets', 'user_config.yaml')) as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    #config['preauthorized']
)

try:
    authenticator.login()
except LoginError as e:
    st.error(e)

#name, authentication_status, username = authenticator.login('Login', 'main')

if st.session_state['authentication_status']:
    authenticator.logout()
    st.write(f'Welcome *{st.session_state["name"]}*')
    main()
elif st.session_state['authentication_status'] is False:
    st.error('Username/password is incorrect')
elif st.session_state['authentication_status'] is None:
    st.warning('Please enter your username and password')
