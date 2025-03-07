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
from datetime import date, timedelta
from prophet import Prophet

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

    X = df[['ndvi_value','ndwi_value','month_x','Chad','Ethiopia','South Sudan','Sudan']]
    y = df['outbreak']

    classifier = LogisticRegression(solver='liblinear', max_iter=100, C=1, class_weight='balanced')

    res = classifier.fit(X , y)

    return res

model = model(df)



# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# Malaria Outbreak Predictor

This model predicts the outbreak of malaria in Sudan

'''

''' 
This model used the time of year to predict if a malaria outbreak will occur

'''


# Add some spacing
''
''


month = st.number_input('please provide the month of the year you are looking for (1-12)')
year = st.number_input('please provide the year you are looking for')

def forecaster(month, year, df):

    date = f'{int(year)}-{int(month)}-15'

    #select the date as 'ds' and values as 'y' for the prophet model to work 
    ndvi_df = df[['ds','ndvi_value']]
    ndvi_df = ndvi_df.rename(columns={'ndvi_value':'y'})

    ndwi_df = df[['ds','ndwi_value']]
    ndwi_df = ndwi_df.rename(columns={'ndwi_value':'y'})
    
    #create an instance of a prophet forecaster
    ndvi_fore = Prophet()

    ndvi_fore.fit(ndvi_df)

    ndvi_pred = ndvi_fore.predict(pd.DataFrame(data={'ds': date}, index=[0]))

    ndvi_pred.head()

    ndvi_val = ndvi_pred.loc[0, 'yhat']
    
    #now create an instance for the ndwi value

    ndwi_fore = Prophet()

    ndwi_fore.fit(ndwi_df)

    ndwi_pred = ndwi_fore.predict(pd.DataFrame(data={'ds': date}, index=[0]))

    ndwi_val = ndwi_pred.loc[0, 'yhat']


    return ndvi_val, ndwi_val

inputs = None

if month and year:
    forecast = forecaster(month, year, df)

    st.write(forecast)

    inputs = np.array([forecast[0], forecast[1], int(month), 0 ,0 ,0 ,1])
    
    st.write(inputs)
    inputs = inputs.reshape(1,-1)
    st.write(inputs.shape)

    pred = model.predict(inputs)
    pred_proba = model.predict_proba(inputs)
    st.write(pred, pred_proba)



    #verbally declare the prediction value's interpretation
    #verbally declare the probablity of the prediction
    
    if pred == 0:
        
        st.write('the prediction is that there will not be a malaria outbreak')

    elif pred == 1:
        
        st.write('the prediction is that there will be a malaria outbreak')


    if pred_proba is not None:

        st.write(f'the probability of a malaria outbreak not occuring is {pred_proba[0,0]}')
        st.write(f'the probability of a malaria outbreak occuring is {pred_proba[0,1]}')



