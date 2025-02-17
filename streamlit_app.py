#imports

import streamlit as st
import pandas as pd
import math
from pathlib import Path
import sklearn.model_selection
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
import streamlit as st
import folium
import numpy as np
import rasterio
from streamlit_folium import folium_static
from folium.plugins import HeatMap
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, MimeType, BBox, CRS
from streamlit.components.v1 import html

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Malaria Outbreak Predictor',
    page_icon=':mosquito', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def get_data():
    """ Get the NDVI AND NDWI VALUE INFORMATION AND INPUT IT INTO THE MODEL
    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    """

    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    DATA_FILENAME = Path(__file__).parent/'ndvi_ndwi_values_2000_2023_gen.csv'
    df = pd.read_csv(DATA_FILENAME)


    df = df.drop(columns= 'Unnamed: 0')

    one_hots = pd.get_dummies(df['GEO_NAME_SHORT'])

    final_df = pd.concat([df, one_hots], axis=1)

    return final_df

#create the dataframe from the csv file
df = get_data()

#train the model on the data with the hyperparameter tuned model specifications
def model(df):

    X = df[[['ndvi_value','ndwi_value','month_x','Chad','Ethiopia','South Sudan','Sudan']]]
    y = df['outbreak']

    classifier = LogisticRegression(solver='liblinear', max_iter=100, C=1)

    res = classifier.fit(X , y)

    return res

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# Malaria Outbreak Predictor
# :mosquito
This model predicts the outbreak of malaria in Sudan

'''

''' 
below is a test to see if the data is uploading

'''

st.write(df.head())
st.write(df.shape)
# Add some spacing
''
''


ndvi = st.number_input('please provide the ndvi value of Sudan')
st.write(ndvi)

ndwi = st.number_input('please provide the ndwi value of Sudan')
st.write(ndwi)

month = st.number_input('please provide the month of the year you are looking for (1-12)')
year = st.number_input('please provide the year that you are looking for')

inputs = np.array([float(ndvi), float(ndwi), int(month), 0 ,0 ,0 ,1])
st.write(inputs)




# SentinelHub API Config
config = SHConfig()
config.sh_client_id = "sh-dcb370a5-45c8-493f-b0fb-b189adb4be33"  # Replace with your SentinelHub Client ID
config.sh_client_secret = "F2bbee2gwHyeFiKMkhuzXZunycBI21op"  # Replace with your SentinelHub Client Secret

# Sudan Bounding Box
SUDAN_BBOX = BBox(bbox=(21.8, 8.7, 39.0, 23.1), crs=CRS.WGS84)

# NDVI Evalscript
EVALSCRIPT_NDVI = """
function setup() {
    return {
        input: ["B04", "B08"],
        output: { bands: 1 }
    };
}
function evaluatePixel(sample) {
    return [(sample.B08 - sample.B04) / (sample.B08 + sample.B04)];
}
"""

# NDWI Evalscript
EVALSCRIPT_NDWI = """
function setup() {
    return {
        input: ["B08", "B11"],
        output: { bands: 1 }
    };
}
function evaluatePixel(sample) {
    return [(sample.B08 - sample.B11) / (sample.B08 + sample.B11)];
}
"""



def get_satellite_index(bbox, evalscript, month, year):
    #"""Fetch NDVI or NDWI data from SentinelHub"""
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(f'{int(month)}-01-{int(year)}', f'{int(month)}-28-{int(year)}'),  # Adjust as needed
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        config=config
    )
    response = request.get_data()
    return np.squeeze(response) if response else None  # Convert to NumPy array

# Streamlit App
st.title("🌍 NDVI & NDWI Map of Sudan")
st.sidebar.header("Select Map Type")

# Get NDVI and NDWI data
ndvi_data = None
ndwi_data = None
ndvi_norm = None
ndwi_norm = None

if month and year:
    ndvi_data = get_satellite_index(SUDAN_BBOX, EVALSCRIPT_NDVI)
    ndwi_data = get_satellite_index(SUDAN_BBOX, EVALSCRIPT_NDWI)

# Normalize Data
    ndvi_norm = (ndvi_data - np.min(ndvi_data)) / (np.max(ndvi_data) - np.min(ndvi_data))
    ndwi_norm = (ndwi_data - np.min(ndwi_data)) / (np.max(ndwi_data) - np.min(ndwi_data))

# Select visualization type
map_type = st.sidebar.radio("Choose Map Type:", ["NDVI", "NDWI", "Both"])

# Create Folium Map
m = folium.Map(location=[15.5, 32.5], zoom_start=5)

# Add NDVI as Heatmap
if map_type in ["NDVI", "Both"]:
    HeatMap(ndvi_norm, name="NDVI", gradient={0: "red", 0.5: "yellow", 1: "green"}).add_to(m)

# Add NDWI as Contour or Heatmap
if map_type in ["NDWI", "Both"]:
    HeatMap(ndwi_norm, name="NDWI", gradient={0: "white", 0.5: "blue", 1: "darkblue"}, radius=10).add_to(m)

# Add Layer Control
folium.LayerControl().add_to(m)

# Display the Map
folium_static(m)
